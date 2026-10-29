//! Queries that exist for the browser UI: a filtered, counted, paged
//! session list and session-id prefix resolution. Everything else the web
//! layer needs is served by the same methods the MCP tools use.

use rusqlite::params;
use rusqlite::types::Value as SqlValue;

use super::sessions::SessionRow;
use super::{Database, DbResult};

/// Filters for [`Database::web_list_sessions`]. Empty strings mean "no
/// filter", matching the rest of the db layer.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct WebListFilter<'a> {
    /// Case-insensitive substring match over title, first prompt, summary
    /// and keywords. Not FTS — a sidebar filter, typed one character at a
    /// time, wants prefix-of-a-word behaviour that FTS5 tokens don't give.
    pub q: Option<&'a str>,
    /// `project_path` prefix.
    pub project: Option<&'a str>,
    /// Exact `cli_source`.
    pub source: Option<&'a str>,
    /// Only sessions modified within the last N days.
    pub days: Option<i64>,
}

/// Sort orders for the browser list. Whitelisted names, never raw SQL from
/// the client; all descending — "biggest/newest first" is the browsing
/// order for every one of these.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum WebSort {
    #[default]
    Modified,
    Created,
    Duration,
    Messages,
}

impl WebSort {
    /// Parse a query-string value; unknown values fall back to Modified.
    pub fn parse(name: Option<&str>) -> WebSort {
        match name.unwrap_or("") {
            "created" => WebSort::Created,
            "duration" => WebSort::Duration,
            "messages" => WebSort::Messages,
            _ => WebSort::Modified,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            WebSort::Modified => "modified",
            WebSort::Created => "created",
            WebSort::Duration => "duration",
            WebSort::Messages => "messages",
        }
    }

    /// The sort column. Combined with a `session_id` tie-break everywhere so
    /// keyset cursors are total-ordered even when many rows share a value.
    fn column(self) -> &'static str {
        match self {
            WebSort::Modified => "modified_at",
            WebSort::Created => "created_at",
            WebSort::Duration => "duration_ms",
            WebSort::Messages => "message_count",
        }
    }

    /// ORDER BY clause. SQLite sorts NULLs last under DESC, which is what a
    /// browser list wants (sessions missing the field sink to the bottom).
    fn order_sql(self) -> String {
        format!("{col} DESC, session_id DESC", col = self.column())
    }

    /// The cursor value for a row under this sort (`None` = SQL NULL).
    fn cursor_value(self, row: &SessionRow) -> Option<SqlValue> {
        match self {
            WebSort::Modified => row.modified_at.clone().map(SqlValue::Text),
            WebSort::Created => row.created_at.clone().map(SqlValue::Text),
            WebSort::Duration => row.duration_ms.map(SqlValue::Integer),
            WebSort::Messages => Some(SqlValue::Integer(row.message_count)),
        }
    }
}

/// An opaque keyset cursor: "resume after the row whose sort value was `v`
/// and whose id was `id`". Serialized as hex-encoded JSON so clients treat
/// it as a token; `parse` rejects anything malformed with `None` (the
/// caller then falls back to the first page rather than erroring).
#[derive(Debug, Clone, PartialEq)]
pub struct WebCursor {
    /// Sort-column value at the boundary row; `None` when it was NULL.
    pub value: Option<serde_json::Value>,
    pub id: String,
}

impl WebCursor {
    pub fn from_row(sort: WebSort, row: &SessionRow) -> WebCursor {
        let value = match sort.cursor_value(row) {
            Some(SqlValue::Text(t)) => Some(serde_json::Value::String(t)),
            Some(SqlValue::Integer(i)) => Some(serde_json::json!(i)),
            _ => None,
        };
        WebCursor {
            value,
            id: row.session_id.clone(),
        }
    }

    pub fn encode(&self) -> String {
        let json = serde_json::json!({ "v": self.value, "id": self.id }).to_string();
        json.bytes().fold(String::new(), |mut out, b| {
            out.push_str(&format!("{b:02x}"));
            out
        })
    }

    pub fn parse(token: &str) -> Option<WebCursor> {
        if token.is_empty() || !token.len().is_multiple_of(2) || token.len() > 2048 {
            return None;
        }
        let bytes: Option<Vec<u8>> = (0..token.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&token[i..i + 2], 16).ok())
            .collect();
        let value: serde_json::Value = serde_json::from_slice(&bytes?).ok()?;
        Some(WebCursor {
            value: match value.get("v") {
                None | Some(serde_json::Value::Null) => None,
                Some(v) => Some(v.clone()),
            },
            id: value.get("id")?.as_str()?.to_string(),
        })
    }

    /// SQL predicate placing this row strictly after the cursor under
    /// `sort`'s DESC-with-NULLs-last order.
    fn predicate(&self, sort: WebSort) -> (String, Vec<SqlValue>) {
        let col = sort.column();
        match &self.value {
            Some(v) => {
                let sql_v = match v {
                    serde_json::Value::String(t) => SqlValue::Text(t.clone()),
                    other => SqlValue::Integer(other.as_i64().unwrap_or(0)),
                };
                (
                    format!(
                        " AND ({col} < ? OR ({col} = ? AND session_id < ?) OR {col} IS NULL)"
                    ),
                    vec![sql_v.clone(), sql_v, SqlValue::Text(self.id.clone())],
                )
            }
            None => (
                format!(" AND {col} IS NULL AND session_id < ?"),
                vec![SqlValue::Text(self.id.clone())],
            ),
        }
    }
}

/// Split a filter query into AND terms: bare words are individual terms,
/// double-quoted runs are phrase terms with spaces preserved. An
/// unterminated quote (someone mid-typing) takes the rest of the input as
/// a phrase rather than failing — this runs per keystroke. Empty terms
/// are dropped.
pub fn filter_terms(q: &str) -> Vec<String> {
    let mut terms = Vec::new();
    let mut current = String::new();
    let mut in_phrase = false;
    for c in q.chars() {
        match c {
            '"' => {
                // Quote toggles phrase mode; the accumulated run (word or
                // phrase) ends either way.
                if !current.trim().is_empty() {
                    terms.push(current.trim().to_string());
                }
                current.clear();
                in_phrase = !in_phrase;
            }
            c if c.is_whitespace() && !in_phrase => {
                if !current.is_empty() {
                    terms.push(current.clone());
                    current.clear();
                }
            }
            c => current.push(c),
        }
    }
    if !current.trim().is_empty() {
        terms.push(current.trim().to_string());
    }
    terms
}

/// One page of sessions, the total the filter matches, and per-source
/// counts under the same filter (minus the source facet itself, so the
/// facet always shows what picking each source would yield).
#[derive(Debug, Clone, PartialEq)]
pub struct WebSessionPage {
    pub sessions: Vec<SessionRow>,
    pub total: i64,
    pub source_counts: Vec<(String, i64)>,
    /// Keyset token for the page after this one; `None` on the last page.
    pub next_cursor: Option<String>,
}

impl Database {
    /// Sessions for the browser's list, newest first, with the total count
    /// for pagination. Empty sessions (`message_count = 0`) are left out —
    /// they are parser artefacts, not something to browse.
    /// `cursor`, when given, wins over `offset`: keyset pagination is
    /// stable while the indexer inserts rows mid-browse, where offsets
    /// drift. Offset stays supported for direct API use.
    pub fn web_list_sessions(
        &self,
        filter: &WebListFilter<'_>,
        sort: WebSort,
        limit: i64,
        offset: i64,
        cursor: Option<&WebCursor>,
    ) -> DbResult<WebSessionPage> {
        // The WHERE clause is built twice: once with the source facet (the
        // page and total) and once without it (the per-source counts).
        let build_where = |with_source: bool| -> (String, Vec<SqlValue>) {
            let mut where_sql = String::from(" WHERE message_count > 0");
            let mut sql_params: Vec<SqlValue> = Vec::new();
            // Every term (word or quoted phrase) must appear somewhere in
            // the row — terms AND, columns OR per term, so `azure` in the
            // title and `backoff` in the summary still match together.
            if let Some(q) = filter.q.map(str::trim).filter(|s| !s.is_empty()) {
                for term in filter_terms(q) {
                    let pattern = format!("%{}%", like_escape(&term));
                    where_sql.push_str(
                        " AND (title LIKE ? ESCAPE '\\' OR first_prompt LIKE ? ESCAPE '\\' \
                         OR summary LIKE ? ESCAPE '\\' OR keywords LIKE ? ESCAPE '\\')",
                    );
                    for _ in 0..4 {
                        sql_params.push(SqlValue::Text(pattern.clone()));
                    }
                }
            }
            if let Some(project) = filter.project.filter(|s| !s.is_empty()) {
                where_sql.push_str(" AND project_path LIKE ? ESCAPE '\\'");
                sql_params.push(SqlValue::Text(format!("{}%", like_escape(project))));
            }
            if with_source && let Some(source) = filter.source.filter(|s| !s.is_empty()) {
                where_sql.push_str(" AND cli_source = ?");
                sql_params.push(SqlValue::Text(source.to_string()));
            }
            if let Some(days) = filter.days.filter(|d| *d > 0) {
                // modified_at is stored as ISO-8601 text; compare against a
                // cutoff in the same shape. Rows without a timestamp drop
                // out of a time-bounded view, which is the honest reading.
                where_sql.push_str(" AND modified_at >= ?");
                let cutoff = chrono::Utc::now() - chrono::Duration::days(days);
                sql_params.push(SqlValue::Text(cutoff.to_rfc3339()));
            }
            (where_sql, sql_params)
        };

        // One connection for everything, so total, page and facet counts
        // describe the same instant even while an index pass is writing.
        let conn = self.connection()?;
        let (where_sql, sql_params) = build_where(true);

        let total: i64 = conn.query_row(
            &format!("SELECT COUNT(*) FROM sessions{where_sql}"),
            rusqlite::params_from_iter(sql_params.iter()),
            |row| row.get(0),
        )?;

        let mut page_where = where_sql.clone();
        let mut page_params = sql_params;
        let offset = if let Some(cursor) = cursor {
            let (pred, mut params) = cursor.predicate(sort);
            page_where.push_str(&pred);
            page_params.append(&mut params);
            0
        } else {
            offset.max(0)
        };
        // Fetch one extra row: its existence — not a full page — is what
        // proves there is a next page, so the last page never dangles a
        // cursor to an empty one.
        let limit = limit.max(0);
        page_params.push(SqlValue::Integer(limit + 1));
        page_params.push(SqlValue::Integer(offset));
        let mut stmt = conn.prepare(&format!(
            "SELECT * FROM sessions{page_where} ORDER BY {} LIMIT ? OFFSET ?",
            sort.order_sql()
        ))?;
        let mut sessions = stmt
            .query_map(
                rusqlite::params_from_iter(page_params.iter()),
                SessionRow::from_row,
            )?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        let has_more = sessions.len() as i64 > limit;
        sessions.truncate(limit as usize);
        let next_cursor = if has_more {
            sessions.last().map(|row| WebCursor::from_row(sort, row).encode())
        } else {
            None
        };

        let (facet_where, facet_params) = build_where(false);
        let mut stmt = conn.prepare(&format!(
            "SELECT cli_source, COUNT(*) FROM sessions{facet_where} GROUP BY cli_source ORDER BY cli_source"
        ))?;
        let source_counts = stmt
            .query_map(rusqlite::params_from_iter(facet_params.iter()), |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?))
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;

        Ok(WebSessionPage {
            sessions,
            total,
            source_counts,
            next_cursor,
        })
    }

    /// Session ids starting with `prefix`, at most `limit` of them. Used to
    /// resolve the short ids people copy out of the UI and the MCP tools.
    pub fn find_session_ids_by_prefix(&self, prefix: &str, limit: i64) -> DbResult<Vec<String>> {
        let conn = self.connection()?;
        let mut stmt = conn.prepare(
            "SELECT session_id FROM sessions WHERE session_id LIKE ? ESCAPE '\\' \
             ORDER BY session_id LIMIT ?",
        )?;
        let rows = stmt.query_map(params![format!("{}%", like_escape(prefix)), limit], |row| {
            row.get::<_, String>(0)
        })?;
        rows.collect::<rusqlite::Result<Vec<_>>>()
            .map_err(Into::into)
    }
}

/// Escape `%`, `_` and the escape char itself for a `LIKE … ESCAPE '\'`.
fn like_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        if matches!(c, '%' | '_' | '\\') {
            out.push('\\');
        }
        out.push(c);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::SessionMetadata;

    fn db_with(rows: &[(&str, &str, &str, &str, i64)]) -> (Database, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let db = Database::new(dir.path().join("t.db")).unwrap();
        for (id, source, project, title, messages) in rows {
            let mut meta = SessionMetadata::new(*id, *source, format!("/t/{id}.jsonl"));
            meta.project_path = Some(project.to_string());
            meta.title = Some(title.to_string());
            meta.message_count = *messages;
            meta.modified_at = Some(chrono::Utc::now().fixed_offset());
            db.upsert_session_with_files(&meta, &[]).unwrap();
        }
        (db, dir)
    }

    #[test]
    fn list_filters_counts_and_pages() {
        let (db, _d) = db_with(&[
            (
                "aaa1",
                "claude_code",
                "/home/u/proj-a",
                "Fix auth 100% bug",
                3,
            ),
            (
                "aaa2",
                "claude_code",
                "/home/u/proj-b",
                "Refactor search",
                5,
            ),
            ("bbb1", "vscode_copilot", "/home/u/proj-a", "Write docs", 2),
            ("ccc1", "claude_code", "/home/u/proj-a", "Empty", 0),
        ]);

        let all = db
            .web_list_sessions(&WebListFilter::default(), WebSort::Modified, 50, 0, None)
            .unwrap();
        assert_eq!(all.total, 3, "empty session excluded");
        assert_eq!(all.sessions.len(), 3);

        let page = db
            .web_list_sessions(&WebListFilter::default(), WebSort::Modified, 2, 0, None)
            .unwrap();
        assert_eq!((page.total, page.sessions.len()), (3, 2));
        let page = db
            .web_list_sessions(&WebListFilter::default(), WebSort::Modified, 2, 2, None)
            .unwrap();
        assert_eq!((page.total, page.sessions.len()), (3, 1));

        let f = WebListFilter {
            project: Some("/home/u/proj-a"),
            ..Default::default()
        };
        assert_eq!(db.web_list_sessions(&f, WebSort::Modified, 50, 0, None).unwrap().total, 2);

        let f = WebListFilter {
            source: Some("vscode_copilot"),
            ..Default::default()
        };
        let r = db.web_list_sessions(&f, WebSort::Modified, 50, 0, None).unwrap();
        assert_eq!(r.sessions[0].session_id, "bbb1");

        // LIKE wildcards in the query are literal.
        let f = WebListFilter {
            q: Some("100%"),
            ..Default::default()
        };
        assert_eq!(db.web_list_sessions(&f, WebSort::Modified, 50, 0, None).unwrap().total, 1);
        let f = WebListFilter {
            q: Some("10%"),
            ..Default::default()
        };
        assert_eq!(db.web_list_sessions(&f, WebSort::Modified, 50, 0, None).unwrap().total, 0);

        let f = WebListFilter {
            days: Some(1),
            ..Default::default()
        };
        assert_eq!(db.web_list_sessions(&f, WebSort::Modified, 50, 0, None).unwrap().total, 3);
    }

    #[test]
    fn filter_terms_tokenize_words_and_phrases() {
        assert_eq!(filter_terms("azure backoff"), vec!["azure", "backoff"]);
        assert_eq!(filter_terms(r#""429 backoff" azure"#), vec!["429 backoff", "azure"]);
        assert_eq!(filter_terms(r#"a "b c" d"#), vec!["a", "b c", "d"]);
        // Unterminated quote: rest of input is one phrase.
        assert_eq!(filter_terms(r#"x "azure ba"#), vec!["x", "azure ba"]);
        // Empty quotes and stray whitespace vanish.
        assert_eq!(filter_terms(r#"  ""  a  "#), vec!["a"]);
        assert!(filter_terms("   ").is_empty());
        assert!(filter_terms(r#""""#).is_empty());
    }

    #[test]
    fn multi_word_filter_is_and_of_terms_across_fields() {
        let dir = tempfile::tempdir().unwrap();
        let db = Database::new(dir.path().join("t.db")).unwrap();
        // Word in title, other word only in summary — cross-field AND.
        let mut meta = SessionMetadata::new("s1", "claude_code", "/t/s1.jsonl");
        meta.title = Some("Add Azure OpenAI support".to_string());
        meta.message_count = 3;
        meta.modified_at = Some(chrono::Utc::now().fixed_offset());
        db.upsert_session_with_files(&meta, &[]).unwrap();
        db.update_summary("s1", "Add Azure OpenAI support", "fixed the 429 backoff schedule")
            .unwrap();
        let mut other = SessionMetadata::new("s2", "claude_code", "/t/s2.jsonl");
        other.title = Some("Azure networking only".to_string());
        other.message_count = 3;
        other.modified_at = Some(chrono::Utc::now().fixed_offset());
        db.upsert_session_with_files(&other, &[]).unwrap();

        let hits = |q: &str| {
            let f = WebListFilter { q: Some(q), ..Default::default() };
            db.web_list_sessions(&f, WebSort::Modified, 50, 0, None).unwrap().total
        };
        assert_eq!(hits("azure backoff"), 1, "AND across fields");
        assert_eq!(hits("backoff azure"), 1, "order-independent");
        assert_eq!(hits("azure"), 2);
        assert_eq!(hits(r#""azure backoff""#), 0, "phrase must be contiguous");
        assert_eq!(hits(r#""429 backoff""#), 1);
        assert_eq!(hits("azure nothere"), 0);
        assert_eq!(hits("back azu"), 1, "terms are substrings, not whole words");
    }

    #[test]
    fn cursor_pages_are_stable_across_ties_and_nulls() {
        let dir = tempfile::tempdir().unwrap();
        let db = Database::new(dir.path().join("t.db")).unwrap();
        // Ten rows: identical modified_at (tie-break by id) and, under the
        // duration sort, all NULL durations (the NULL branch).
        let ts = chrono::Utc::now().fixed_offset();
        for i in 0..10 {
            let mut m = SessionMetadata::new(format!("s{i:02}"), "claude_code", "/t/x.jsonl");
            m.title = Some(format!("t{i}"));
            m.message_count = 1 + i as i64;
            m.modified_at = Some(ts);
            db.upsert_session_with_files(&m, &[]).unwrap();
        }
        let all_ids = |sort: WebSort| {
            let mut ids = Vec::new();
            let mut cursor: Option<WebCursor> = None;
            loop {
                let page = db
                    .web_list_sessions(&WebListFilter::default(), sort, 3, 0, cursor.as_ref())
                    .unwrap();
                ids.extend(page.sessions.iter().map(|s| s.session_id.clone()));
                match page.next_cursor {
                    Some(tok) => cursor = Some(WebCursor::parse(&tok).expect("valid token")),
                    None => break,
                }
            }
            ids
        };
        // Every row exactly once, no overlaps, no gaps — for a tied text
        // sort, an integer sort, and an all-NULL sort column.
        for sort in [WebSort::Modified, WebSort::Messages, WebSort::Duration] {
            let ids = all_ids(sort);
            assert_eq!(ids.len(), 10, "{sort:?}: {ids:?}");
            let unique: std::collections::HashSet<_> = ids.iter().collect();
            assert_eq!(unique.len(), 10, "{sort:?} duplicated rows: {ids:?}");
        }
        // Cursor survives its wire encoding, and garbage parses to None.
        let page = db
            .web_list_sessions(&WebListFilter::default(), WebSort::Messages, 3, 0, None)
            .unwrap();
        let tok = page.next_cursor.unwrap();
        assert_eq!(WebCursor::parse(&tok).unwrap().encode(), tok);
        assert!(WebCursor::parse("zz").is_none());
        assert!(WebCursor::parse("").is_none());
        assert!(WebCursor::parse("abc").is_none());
    }

    #[test]
    fn sort_orders_apply() {
        let (db, _d) = db_with(&[
            ("aaa1", "claude_code", "/p", "few", 2),
            ("aaa2", "claude_code", "/p", "many", 9),
        ]);
        let r = db
            .web_list_sessions(&WebListFilter::default(), WebSort::Messages, 50, 0, None)
            .unwrap();
        assert_eq!(r.sessions[0].session_id, "aaa2");
        assert_eq!(WebSort::parse(Some("nonsense")), WebSort::Modified);
        assert_eq!(WebSort::parse(Some("duration")), WebSort::Duration);
    }

    #[test]
    fn prefix_resolution() {
        let (db, _d) = db_with(&[
            ("aaa1", "claude_code", "/p", "x", 1),
            ("aaa2", "claude_code", "/p", "y", 1),
            ("bbb1", "claude_code", "/p", "z", 1),
        ]);
        assert_eq!(
            db.find_session_ids_by_prefix("aaa", 5).unwrap(),
            vec!["aaa1", "aaa2"]
        );
        assert_eq!(
            db.find_session_ids_by_prefix("bbb1", 5).unwrap(),
            vec!["bbb1"]
        );
        assert!(db.find_session_ids_by_prefix("zzz", 5).unwrap().is_empty());
        assert!(db.find_session_ids_by_prefix("a%", 5).unwrap().is_empty());
    }

    #[test]
    fn like_escape_escapes_wildcards() {
        assert_eq!(like_escape("a%b_c\\d"), "a\\%b\\_c\\\\d");
    }
}
