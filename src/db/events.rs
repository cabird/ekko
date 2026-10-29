//! Structured transcript events, stored capped.
//!
//! The database is the source of truth for the transcript view: parse once,
//! cap once (here, at ingest), and everything downstream serves the stored
//! JSON verbatim. The caps deliberately discard the tails of oversized
//! payloads — a `Write` carrying a whole file body, a tool result dumping a
//! build log — because keeping them would bloat the archive for content
//! nobody re-reads. Measured on a real 121 MB / 86-transcript archive, the
//! capped events total 7.3 MB.
//!
//! Each stored event is the JSON object the web transcript route serves,
//! with `truncated: true` and `full_len` recorded whenever a cap bit.

use rusqlite::params;
use serde_json::{Value, json};

use super::{Database, DbResult};
use crate::models::Event;

/// Cap for message text (chars). Long messages are usually pasted logs;
/// the conversation itself comfortably fits.
pub const EVENT_CONTENT_CAP: usize = 10_000;
/// Cap for a tool call's serialized input (chars) — enough for any command
/// line, file path, or an edit's old/new context.
pub const EVENT_TOOL_INPUT_CAP: usize = 2_000;
/// Cap for a tool result (chars) — the head of any log.
pub const EVENT_TOOL_OUTPUT_CAP: usize = 4_000;

/// Truncate to `max` chars, reporting `(text, was_cut, original_len)`.
/// Shared with the web layer's display truncation.
pub(crate) fn cap(s: &str, max: usize) -> (String, bool, usize) {
    let len = s.chars().count();
    if len <= max {
        (s.to_string(), false, len)
    } else {
        (s.chars().take(max).collect(), true, len)
    }
}

/// One parsed event as its stored (capped) JSON object.
pub fn event_to_stored_json(e: &Event) -> Value {
    let (content, content_cut, content_len) = cap(&e.content, EVENT_CONTENT_CAP);
    // Tool input is capped on its serialized form; a truncated JSON string
    // is no longer JSON, so it is stored (and served) as text from then on.
    let (tool_input, input_cut, input_len) = match &e.tool_input {
        Some(v) => {
            let serialized = match v {
                Value::String(s) => s.clone(),
                other => serde_json::to_string_pretty(other).unwrap_or_else(|_| other.to_string()),
            };
            let (text, cut, len) = cap(&serialized, EVENT_TOOL_INPUT_CAP);
            if cut {
                (Some(Value::String(text)), true, len)
            } else {
                // Untruncated inputs keep their JSON structure.
                (Some(v.clone()), false, len)
            }
        }
        None => (None, false, 0),
    };
    let (tool_output, output_cut, output_len) = match &e.tool_output {
        Some(o) => {
            let (text, cut, len) = cap(o, EVENT_TOOL_OUTPUT_CAP);
            (Some(text), cut, len)
        }
        None => (None, false, 0),
    };
    let truncated = content_cut || input_cut || output_cut;
    // `full_len` describes the field that was cut (the largest one when
    // several were), so the UI's "N chars originally" is about the payload
    // the reader is missing — not the message text next to it.
    let full_len = [
        (content_cut, content_len),
        (input_cut, input_len),
        (output_cut, output_len),
    ]
    .into_iter()
    .filter(|(cut, _)| *cut)
    .map(|(_, len)| len)
    .max()
    .unwrap_or(0);
    let mut out = json!({
        "sequence": e.sequence,
        "type": e.event_type.as_str(),
        "timestamp": e.timestamp.map(|t| t.to_rfc3339()),
        "role": e.role.as_ref().map(|r| r.as_str()),
        "content": content,
        "tool_name": e.tool_name,
        "tool_input": tool_input,
        "tool_call_id": e.tool_call_id,
        "tool_output": tool_output,
        "tool_success": e.tool_success,
        "artifact_path": e.artifact_path,
        "artifact_type": e.artifact_type,
        "truncated": truncated,
    });
    if truncated && let Some(map) = out.as_object_mut() {
        map.insert("full_len".into(), json!(full_len));
    }
    out
}

impl Database {
    /// Replace a session's stored events with the capped rendering of
    /// `events`. One transaction; ordering is parse order (the parser's
    /// `sequence` field is not unique — see the schema comment).
    pub fn store_events(&self, session_id: &str, events: &[Event]) -> DbResult<()> {
        let mut conn = self.connection()?;
        let tx = conn.transaction()?;
        tx.execute(
            "DELETE FROM session_events WHERE session_id = ?",
            params![session_id],
        )?;
        {
            let mut stmt =
                tx.prepare("INSERT INTO session_events (session_id, pos, event) VALUES (?, ?, ?)")?;
            for (pos, e) in events.iter().enumerate() {
                stmt.execute(params![
                    session_id,
                    pos as i64,
                    event_to_stored_json(e).to_string()
                ])?;
            }
        }
        tx.commit()?;
        Ok(())
    }

    /// A session's stored events in sequence order, exactly as stored.
    pub fn get_events(&self, session_id: &str) -> DbResult<Vec<Value>> {
        let conn = self.connection()?;
        let mut stmt =
            conn.prepare("SELECT event FROM session_events WHERE session_id = ? ORDER BY pos")?;
        let rows = stmt.query_map(params![session_id], |row| row.get::<_, String>(0))?;
        let mut out = Vec::new();
        for row in rows {
            let text = row?;
            out.push(
                serde_json::from_str(&text).map_err(|source| super::DbError::Json {
                    context: "session_events.event",
                    source,
                })?,
            );
        }
        Ok(out)
    }

    /// Whether any events are stored for this session.
    pub fn has_events(&self, session_id: &str) -> DbResult<bool> {
        let conn = self.connection()?;
        let n: i64 = conn.query_row(
            "SELECT EXISTS(SELECT 1 FROM session_events WHERE session_id = ?)",
            params![session_id],
            |row| row.get(0),
        )?;
        Ok(n != 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{EventType, MessageRole};

    fn event(seq: i64) -> Event {
        Event {
            id: format!("e{seq}"),
            sequence: seq,
            event_type: EventType::Message,
            timestamp: None,
            role: Some(MessageRole::User),
            content: "hello".to_string(),
            tool_name: None,
            tool_input: None,
            tool_call_id: None,
            tool_output: None,
            tool_success: None,
            artifact_path: None,
            artifact_type: None,
            raw_ref: None,
        }
    }

    #[test]
    fn caps_bite_at_ingest_and_are_recorded() {
        let mut e = event(0);
        e.content = "x".repeat(EVENT_CONTENT_CAP + 5);
        let v = event_to_stored_json(&e);
        assert_eq!(
            v["content"].as_str().unwrap().chars().count(),
            EVENT_CONTENT_CAP
        );
        assert_eq!(v["truncated"], true);
        assert_eq!(v["full_len"], (EVENT_CONTENT_CAP + 5) as i64);

        let mut e = event(1);
        e.event_type = EventType::ToolCall;
        e.tool_name = Some("Write".to_string());
        e.tool_input =
            Some(serde_json::json!({"file_path": "/x", "content": "y".repeat(3_000_000)}));
        let v = event_to_stored_json(&e);
        let input = v["tool_input"]
            .as_str()
            .expect("truncated input becomes text");
        assert_eq!(input.chars().count(), EVENT_TOOL_INPUT_CAP);
        assert_eq!(v["truncated"], true);

        // Small structured inputs stay structured.
        let mut e = event(2);
        e.tool_input = Some(serde_json::json!({"file_path": "/x"}));
        let v = event_to_stored_json(&e);
        assert_eq!(v["tool_input"]["file_path"], "/x");
        assert_eq!(v["truncated"], false);

        let mut e = event(3);
        e.tool_output = Some("z".repeat(EVENT_TOOL_OUTPUT_CAP * 2));
        let v = event_to_stored_json(&e);
        assert_eq!(
            v["tool_output"].as_str().unwrap().chars().count(),
            EVENT_TOOL_OUTPUT_CAP
        );
    }

    #[test]
    fn full_len_reports_the_field_that_was_actually_cut() {
        // Tiny message, huge tool output: the number shown to the reader
        // must be the output's size, not the message's.
        let mut e = event(0);
        e.content = "hi".to_string();
        e.tool_output = Some("z".repeat(EVENT_TOOL_OUTPUT_CAP * 3));
        let v = event_to_stored_json(&e);
        assert_eq!(v["truncated"], true);
        assert_eq!(v["full_len"], (EVENT_TOOL_OUTPUT_CAP * 3) as i64);

        // Oversized tool input, same rule.
        let mut e = event(1);
        e.tool_input = Some(serde_json::json!({ "content": "y".repeat(50_000) }));
        let v = event_to_stored_json(&e);
        let serialized_len = serde_json::to_string_pretty(&e.tool_input.clone().unwrap())
            .unwrap()
            .chars()
            .count();
        assert_eq!(v["full_len"], serialized_len as i64);

        // Nothing cut: no full_len key at all.
        let v = event_to_stored_json(&event(2));
        assert!(v.get("full_len").is_none());
    }

    #[test]
    fn store_load_roundtrip_and_replace() {
        let dir = tempfile::tempdir().unwrap();
        let db = Database::new(dir.path().join("t.db")).unwrap();
        assert!(!db.has_events("s1").unwrap());
        // Duplicate parser sequences are legal (message + its tool calls
        // share one) — storage must not reject them.
        db.store_events("s1", &[event(0), event(1), event(1)])
            .unwrap();
        assert!(db.has_events("s1").unwrap());
        let got = db.get_events("s1").unwrap();
        assert_eq!(got.len(), 3);
        assert_eq!(got[0]["sequence"], 0);
        assert_eq!(got[1]["content"], "hello");
        // Re-store replaces, never appends.
        db.store_events("s1", &[event(7)]).unwrap();
        let got = db.get_events("s1").unwrap();
        assert_eq!(got.len(), 1);
        assert_eq!(got[0]["sequence"], 7);
        // delete_session clears events too.
        let mut meta = crate::models::SessionMetadata::new("s1", "claude_code", "/t/x.jsonl");
        meta.message_count = 1;
        db.upsert_session_with_files(&meta, &[]).unwrap();
        db.delete_session("s1").unwrap();
        assert!(!db.has_events("s1").unwrap());
    }
}
