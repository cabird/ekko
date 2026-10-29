//! `recent_activity` — the "what was I working on?" briefing.

use serde_json::{Map, Value, json};

use super::{SessionFilter, EkkoMemory, opt_str, title_or_first_prompt};
use crate::db::DbResult;

/// Arguments for [`EkkoMemory::recent_activity`], with their defaults.
#[derive(Debug, Clone)]
pub struct RecentActivityArgs<'a> {
    /// Scope to sessions in/under this directory.
    pub directory: Option<&'a str>,
    /// Include sessions in subdirectories.
    pub include_subdirs: bool,
    /// Look back this many days.
    pub days: i64,
    /// Maximum sessions to include.
    pub max_sessions: usize,
}

impl Default for RecentActivityArgs<'_> {
    fn default() -> Self {
        RecentActivityArgs {
            directory: None,
            include_subdirs: true,
            days: 7,
            max_sessions: 10,
        }
    }
}

/// How many aggregated files `recent_activity` reports.
const TOP_FILES: usize = 15;

impl EkkoMemory {
    /// Quick briefing on recent work: recent session summaries plus the files
    /// they touched.
    ///
    /// # Errors
    ///
    /// Propagates database errors.
    pub fn recent_activity(&self, args: &RecentActivityArgs<'_>) -> DbResult<Value> {
        let all_sessions = self.db().get_sessions_with_summaries(None)?;
        let mut sessions = self.filter_sessions(
            all_sessions,
            &SessionFilter::new()
                .directory(args.directory, args.include_subdirs)
                .days(Some(args.days)),
        );
        sessions.truncate(args.max_sessions);

        if sessions.is_empty() {
            let days = args.days;
            let mut summary = format!("No sessions found in the last {days} days");
            if let Some(dir) = args.directory.filter(|d| !d.is_empty()) {
                summary.push_str(&format!(" under {dir}"));
            }
            return Ok(json!({
                "sessions": [],
                "files_touched": [],
                "summary": summary,
            }));
        }

        // Aggregate files touched (one batch query).
        let session_ids: Vec<String> = sessions.iter().map(|s| s.session_id.clone()).collect();
        let all_files = self.db().get_files_for_sessions(&session_ids)?;

        // Paths are accumulated in first-seen order, which the stable sort
        // below preserves for ties.
        let mut order: Vec<String> = Vec::new();
        let mut stats: std::collections::HashMap<String, (i64, Vec<String>)> =
            std::collections::HashMap::new();
        for (_session_id, files) in &all_files {
            for f in files {
                let entry = stats.entry(f.file_path.clone()).or_insert_with(|| {
                    order.push(f.file_path.clone());
                    (0, Vec::new())
                });
                entry.0 += 1;
                // Operations are kept in first-seen order so the rendered
                // list is deterministic and testable.
                if !f.operation.is_empty() && !entry.1.contains(&f.operation) {
                    entry.1.push(f.operation.clone());
                }
            }
        }

        let mut ranked: Vec<&String> = order.iter().collect();
        // `sorted(key=lambda x: -x['count'])` — stable, so ties keep
        // first-seen order.
        ranked.sort_by_key(|path| -stats[*path].0);
        let top_files: Vec<Value> = ranked
            .into_iter()
            .take(TOP_FILES)
            .map(|path| {
                let (count, operations) = &stats[path];
                json!({"path": path, "count": count, "operations": operations})
            })
            .collect();

        let formatted: Vec<Value> = sessions
            .iter()
            .map(|s| {
                let mut m = Map::new();
                m.insert("session_id".into(), json!(s.session_id));
                m.insert("title".into(), json!(title_or_first_prompt(s)));
                m.insert("summary".into(), opt_str(s.summary.as_deref()));
                m.insert("project_path".into(), opt_str(s.project_path.as_deref()));
                m.insert("git_branch".into(), opt_str(s.git_branch.as_deref()));
                m.insert("cli_source".into(), json!(s.cli_source));
                m.insert("modified_at".into(), opt_str(s.modified_at.as_deref()));
                m.insert("message_count".into(), json!(s.message_count));
                m.insert(
                    "transcript_path".into(),
                    opt_str(s.transcript_path.as_deref()),
                );
                Value::Object(m)
            })
            .collect();

        Ok(json!({
            "sessions": formatted,
            "files_touched": top_files,
            // Room for an LLM narrative; always null today.
            "summary": Value::Null,
        }))
    }
}
