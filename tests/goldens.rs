//! Byte-fidelity golden tests for the prompt pipeline.
//!
//! The expected bytes in `tests/goldens/*.json` are pinned expected output:
//!
//! - `templates.json` — every prompt rendered through
//!   [`ekko::prompts::render`] for a fixed set of inputs.
//! - `inline.json` — the map/reduce prompts assembled inside the summarize
//!   and snapshot pipelines, captured for the same fixed inputs.
//!
//! Exactly what reaches the LLM is behaviour, so these are pinned to the
//! byte. Update them deliberately, as part of a change that intends to alter
//! a prompt — never to make a surprised test pass.

use ekko::prompts;
use ekko::snapshot::{self, SnapshotMeta};
use ekko::summarize;
use serde_json::Value;

const TEMPLATES_JSON: &str = include_str!("goldens/templates.json");
const INLINE_JSON: &str = include_str!("goldens/inline.json");

fn templates() -> Vec<Value> {
    serde_json::from_str(TEMPLATES_JSON).expect("templates.json parses")
}

fn inline() -> Value {
    serde_json::from_str(INLINE_JSON).expect("inline.json parses")
}

/// Show the first differing byte, so a failure names where the two texts
/// part company instead of dumping two 5 KB prompts.
fn assert_bytes_eq(id: &str, actual: &str, expected: &str) {
    if actual == expected {
        return;
    }
    let pos = actual
        .as_bytes()
        .iter()
        .zip(expected.as_bytes())
        .position(|(a, b)| a != b)
        .unwrap_or(actual.len().min(expected.len()));
    let from = pos.saturating_sub(60);
    panic!(
        "golden `{id}` diverged at byte {pos} (actual {} bytes, golden {} bytes)\n\
         actual: …{:?}\n\
         golden: …{:?}",
        actual.len(),
        expected.len(),
        actual.get(from..(pos + 60).min(actual.len())).unwrap_or(""),
        expected
            .get(from..(pos + 60).min(expected.len()))
            .unwrap_or(""),
    );
}

// ---------------------------------------------------------------------------
// Template renders
// ---------------------------------------------------------------------------

#[test]
fn every_template_golden_matches_pinned_bytes() {
    let cases = templates();
    assert!(
        cases.len() >= 12,
        "expected at least 12 golden renders, got {}",
        cases.len()
    );

    for case in &cases {
        let id = case["id"].as_str().expect("id");
        let template = case["template"].as_str().expect("template");
        let ctx = case["context"].clone();
        let expected = case["expected"].as_str().expect("expected");
        let actual = prompts::render(template, &ctx)
            .unwrap_or_else(|e| panic!("golden `{id}` failed to render: {e}"));
        assert_bytes_eq(id, &actual, expected);
    }
}

#[test]
fn goldens_cover_every_wired_template_and_the_tricky_scenarios() {
    let cases = templates();
    let ids: Vec<&str> = cases.iter().map(|c| c["id"].as_str().unwrap()).collect();
    let used: std::collections::HashSet<&str> = cases
        .iter()
        .map(|c| c["template"].as_str().unwrap())
        .collect();

    for name in prompts::list_prompts() {
        assert!(
            used.contains(name.as_str()),
            "no golden for template {name}"
        );
    }

    for required in [
        "summarize_basic",
        "summarize_missing_var",
        "context_snapshot_full",
        "context_snapshot_none_branch",
        "semantic_search_basic",
        "qa_transcript_with_summary",
        "qa_transcript_no_summary_key",
        "qa_transcript_empty_summary",
        "qa_synthesize_zero",
        "qa_synthesize_one",
        "qa_synthesize_three",
        "qa_from_summaries_basic",
    ] {
        assert!(
            ids.contains(&required),
            "missing golden scenario {required}"
        );
    }
}

/// The template-engine properties the goldens are meant to pin, asserted
/// directly against the golden bytes so a future refactor cannot quietly
/// "fix" them.
#[test]
fn goldens_confirm_template_semantics() {
    let cases = templates();
    let get = |id: &str| -> String {
        cases
            .iter()
            .find(|c| c["id"] == id)
            .unwrap_or_else(|| panic!("no golden {id}"))["expected"]
            .as_str()
            .unwrap()
            .to_string()
    };

    // A rendered prompt does not end in a newline: the template file's final
    // "\n" is stripped.
    let sem = get("semantic_search_basic");
    assert!(sem.ends_with("Return up to 10 results."), "{sem:?}");

    // A missing variable renders as the empty string, never an error.
    let missing = get("summarize_missing_var");
    assert!(
        missing.ends_with("## Conversation\n\n\n\nReturn only valid JSON."),
        "{:?}",
        &missing[missing.len() - 60..]
    );
    let limitless = get("semantic_search_empty_sessions");
    assert!(limitless.contains("Return up to 1 results."));
    let qa_missing = get("qa_from_summaries_missing");
    assert!(qa_missing.contains("## Question\n\n\"\"\n"));

    // A null value renders as the four bytes "None".
    let none_branch = get("context_snapshot_none_branch");
    assert!(
        none_branch.contains("**Git Branch:** None\n"),
        "{none_branch}"
    );
    assert!(none_branch.contains("**Session Date:** None\n"));
    let none_qa = get("qa_transcript_none_summary");
    assert!(none_qa.contains("- **Project:** None\n- **Date:** None\n"));
    // …but `{% if summary %}` is still falsy for a null value.
    assert!(!none_qa.contains("**Summary:**"));

    // A missing variable leaves the field blank rather than printing "None".
    let missing_vars = get("context_snapshot_missing_vars");
    assert!(missing_vars.contains("**Git Branch:** \n"));

    // Empty string == missing for `{% if %}`.
    assert_eq!(
        get("qa_transcript_empty_summary"),
        get("qa_transcript_no_summary_key")
    );

    // No HTML escaping for .md templates.
    let basic = get("summarize_basic");
    assert!(basic.contains("Fix the <auth> bug & make it 'work'"));
    assert!(!basic.contains("&amp;"));

    // trim_blocks/lstrip_blocks: the newline after `{% endif %}` is eaten,
    // which is why Title and Summary end up on the same line.
    let three = get("qa_synthesize_three");
    assert!(
        three.contains("- **Title:** Deploy to Azure- **Summary:** Outcome: shipped v2.\n"),
        "{three}"
    );
    // loop.index is 1-based and keeps counting across entries with missing keys.
    assert!(three.contains("### Session 3: None (None)\n"));
    // Zero answers: the for-body contributes nothing at all.
    assert!(get("qa_synthesize_zero").contains("## Session Answers\n\n\n---\n"));
}

// ---------------------------------------------------------------------------
// Inline map-reduce prompts
// ---------------------------------------------------------------------------

#[test]
fn summarize_map_prompts_match_pinned_bytes() {
    let data = inline();
    let cases = data["summarize_map"].as_array().expect("summarize_map");
    assert_eq!(cases.len(), 3);
    for case in cases {
        let index = case["index"].as_u64().unwrap() as usize;
        let num_chunks = case["num_chunks"].as_u64().unwrap() as usize;
        let chunk = case["chunk"].as_str().unwrap();
        let expected = case["expected"].as_str().unwrap();
        let actual = summarize::map_prompt(index, num_chunks, chunk).unwrap();
        assert_bytes_eq(&format!("summarize_map[{index}]"), &actual, expected);
    }
}

#[test]
fn summarize_reduce_prompt_matches_pinned_bytes() {
    let data = inline();
    for case in data["summarize_reduce"]
        .as_array()
        .expect("summarize_reduce")
    {
        let num_chunks = case["num_chunks"].as_u64().unwrap() as usize;
        let parts: Vec<Value> = case["parts"].as_array().unwrap().clone();
        let expected = case["expected"].as_str().unwrap();
        let actual = summarize::reduce_prompt(&parts, num_chunks);
        assert_bytes_eq("summarize_reduce", &actual, expected);

        // The fixture exercises the conditional fields: a part with
        // title+topics, one with neither, one with empty strings.
        assert!(expected.contains("--- Part 1 of 3 ---\nTitle: Part one title\nSummary: Part one summary.\nTopics: a, b, c\n\n"));
        assert!(expected.contains("--- Part 2 of 3 ---\nSummary: Part two summary.\n\n"));
        assert!(expected.contains("--- Part 3 of 3 ---\nTitle: Part three title\nSummary: \n\n"));
    }
}

#[test]
fn summarize_mapreduce_result_post_processing_is_pinned() {
    let data = inline();
    let expected = &data["summarize_result"];
    // The reduce step takes {"title": "Merged", …} and appends
    // "\n\nTopics: x, y" to the summary.
    let actual = summarize::summary_from_response(&serde_json::json!({
        "title": "Merged", "summary": "Merged summary", "topics": "x, y"
    }));
    assert_eq!(actual.title.as_deref(), expected["title"].as_str());
    assert_eq!(actual.summary.as_deref(), expected["summary"].as_str());
    assert_eq!(
        actual.summary.as_deref(),
        Some("Merged summary\n\nTopics: x, y")
    );
}

fn meta_from_json(v: &Value) -> SnapshotMeta {
    SnapshotMeta {
        project_path: v["project_path"].as_str().unwrap().to_string(),
        cli_source: v["cli_source"].as_str().unwrap().to_string(),
        git_branch: v["git_branch"].as_str().unwrap().to_string(),
        session_date: v["session_date"].as_str().unwrap().to_string(),
        brief_title: v["brief_title"].as_str().unwrap().to_string(),
    }
}

#[test]
fn snapshot_map_prompts_match_pinned_bytes() {
    let data = inline();
    let cases = data["snapshot_map"].as_array().expect("snapshot_map");
    assert_eq!(cases.len(), 2);
    for case in cases {
        let index = case["index"].as_u64().unwrap() as usize;
        let num_chunks = case["num_chunks"].as_u64().unwrap() as usize;
        let chunk = case["chunk"].as_str().unwrap();
        let meta = meta_from_json(&case["meta"]);
        let expected = case["expected"].as_str().unwrap();
        let actual = snapshot::map_prompt(index, num_chunks, chunk, &meta).unwrap();
        assert_bytes_eq(&format!("snapshot_map[{index}]"), &actual, expected);
    }
}

#[test]
fn snapshot_reduce_prompt_matches_pinned_bytes() {
    let data = inline();
    for case in data["snapshot_reduce"].as_array().expect("snapshot_reduce") {
        let num_chunks = case["num_chunks"].as_u64().unwrap() as usize;
        let parts: Vec<String> = case["parts"]
            .as_array()
            .unwrap()
            .iter()
            .map(|p| p.as_str().unwrap().to_string())
            .collect();
        let meta = meta_from_json(&case["meta"]);
        let expected = case["expected"].as_str().unwrap();
        let actual = snapshot::reduce_prompt(&parts, num_chunks, &meta);
        assert_bytes_eq("snapshot_reduce", &actual, expected);
        assert!(expected.contains("=== Part 1 of 2 ===\n# Snapshot part 1\n- a\n\n"));
        assert!(expected.contains("**Git Branch:** main\n"));
    }
}

/// The single-shot prompts are the map prompts minus the NOTE preamble; pin
/// that relationship so a change to one cannot silently skip the other.
#[test]
fn single_shot_prompts_are_the_map_prompts_without_the_note() {
    let data = inline();

    let case = &data["summarize_map"].as_array().unwrap()[0];
    let chunk = case["chunk"].as_str().unwrap();
    let expected = case["expected"].as_str().unwrap();
    let single = summarize::summarize_prompt(chunk).unwrap();
    assert_eq!(
        expected,
        format!(
            "NOTE: This is part 1 of 3 of a large session transcript. \
Summarize only what is in this part.\n\n{single}"
        )
    );

    let case = &data["snapshot_map"].as_array().unwrap()[0];
    let chunk = case["chunk"].as_str().unwrap();
    let meta = meta_from_json(&case["meta"]);
    let expected = case["expected"].as_str().unwrap();
    let single = snapshot::snapshot_prompt(chunk, &meta).unwrap();
    assert_eq!(
        expected,
        format!(
            "NOTE: This is part 1 of 2 of a large session transcript. \
Generate a context snapshot covering only what is in this part.\n\n{single}"
        )
    );
}
