//! Prompt loading and rendering.
//!
//! The `prompts/*.md` templates are embedded at compile time and rendered with
//! [minijinja], configured so that: `trim_blocks` and `lstrip_blocks` are on;
//! autoescaping is off (every template is Markdown, so `<`, `&` and `'` pass
//! through verbatim); undefined variables are lenient — a missing variable
//! renders as the empty string, is falsy in `{% if %}` alongside `null` and
//! `""`, and iterating one yields an empty loop rather than an error; rendered
//! output does not end in a trailing newline even though every template file
//! does; and a null value renders as the literal `None`. Sequences and maps
//! interpolated directly render through [`container_formatter`]. The rendered
//! bytes are pinned by goldens in `tests/goldens.rs`.

use std::sync::LazyLock;

use minijinja::value::ValueKind;
use minijinja::{Environment, Error as JinjaError, ErrorKind, Output, State, Value as JinjaValue};
use serde::Serialize;

/// Every prompt template, embedded at compile time. Keys carry the `.md`
/// extension; callers name templates without it.
pub const TEMPLATES: &[(&str, &str)] = &[
    (
        "context_snapshot.md",
        include_str!("../prompts/context_snapshot.md"),
    ),
    (
        "qa_from_summaries.md",
        include_str!("../prompts/qa_from_summaries.md"),
    ),
    (
        "qa_synthesize.md",
        include_str!("../prompts/qa_synthesize.md"),
    ),
    (
        "qa_transcript.md",
        include_str!("../prompts/qa_transcript.md"),
    ),
    (
        "semantic_search.md",
        include_str!("../prompts/semantic_search.md"),
    ),
    ("summarize.md", include_str!("../prompts/summarize.md")),
];

/// Failure rendering a prompt. Rendering cannot fail for the inputs ekko
/// passes; these variants exist so a template/typo bug surfaces instead of
/// producing a truncated prompt.
#[derive(Debug, thiserror::Error)]
pub enum PromptError {
    /// No template with this name is embedded.
    #[error("unknown prompt template: {0}")]
    Unknown(String),
    /// The template exists but rendering or context serialization failed.
    #[error("failed to render prompt '{name}': {source}")]
    Render {
        /// Template name (without `.md`).
        name: String,
        /// Underlying minijinja error.
        #[source]
        source: JinjaError,
    },
}

/// Render a directly interpolated sequence or map in single-quoted style
/// (`['a', 'b']`), not the engine's JSON-ish default (`["a", "b"]`).
///
/// No wired template interpolates a container today; the formatter exists so a
/// future caller cannot silently change the rendered bytes. Scalars (`none`,
/// bools, numbers, strings) go through the engine's own formatter.
fn container_formatter(out: &mut Output, state: &State, value: &JinjaValue) -> Result<(), JinjaError> {
    match value.kind() {
        ValueKind::Seq | ValueKind::Map => {
            let json: serde_json::Value = serde_json::to_value(value)
                .map_err(|e| JinjaError::new(ErrorKind::BadSerialization, e.to_string()))?;
            out.write_str(&crate::models::display_str(&json))?;
            Ok(())
        }
        _ => minijinja::escape_formatter(out, state, value),
    }
}

/// The process-wide prompt environment; every template is parsed once.
static ENV: LazyLock<Environment<'static>> = LazyLock::new(|| {
    let mut env = Environment::new();
    env.set_trim_blocks(true);
    env.set_lstrip_blocks(true);
    // Every template is Markdown, so nothing is ever HTML-escaped.
    env.set_auto_escape_callback(|_name| minijinja::AutoEscape::None);
    // A missing variable renders as "" and is falsy, rather than erroring.
    env.set_undefined_behavior(minijinja::UndefinedBehavior::Lenient);
    env.set_formatter(container_formatter);
    for (name, source) in TEMPLATES {
        env.add_template(name, source)
            .expect("embedded prompt templates must compile");
    }
    env
});

/// Render a prompt template with the given context.
///
/// `name` omits the `.md` extension, e.g. `render("summarize", ctx)`.
/// The context may be any [`Serialize`] value that produces a map —
/// [`minijinja::context!`], a `serde_json::Value::Object`, a `HashMap`, or a
/// `#[derive(Serialize)]` struct.
///
/// # Errors
///
/// [`PromptError::Unknown`] if no such template is embedded,
/// [`PromptError::Render`] if rendering fails.
pub fn render<S: Serialize>(name: &str, ctx: S) -> Result<String, PromptError> {
    let file = format!("{name}.md");
    let template = ENV
        .get_template(&file)
        .map_err(|_| PromptError::Unknown(name.to_string()))?;
    template.render(ctx).map_err(|source| PromptError::Render {
        name: name.to_string(),
        source,
    })
}

/// All available prompt template names, without the `.md` extension, sorted.
pub fn list_prompts() -> Vec<String> {
    let mut names: Vec<String> = TEMPLATES
        .iter()
        // `strip_suffix`, not `trim_end_matches`: the latter strips repeated
        // suffixes (`a.md.md` -> `a`).
        .map(|(n, _)| n.strip_suffix(".md").unwrap_or(n).to_string())
        .collect();
    names.sort();
    names
}

/// True if a template with this name (no `.md`) is embedded.
pub fn has_prompt(name: &str) -> bool {
    TEMPLATES.iter().any(|(n, _)| *n == format!("{name}.md"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn lists_all_prompts() {
        let names = list_prompts();
        assert_eq!(
            names,
            vec![
                "context_snapshot",
                "qa_from_summaries",
                "qa_synthesize",
                "qa_transcript",
                "semantic_search",
                "summarize",
            ]
        );
    }

    #[test]
    fn unknown_template_errors() {
        let err = render("nope", json!({})).unwrap_err();
        assert!(matches!(err, PromptError::Unknown(ref n) if n == "nope"));
        assert!(!has_prompt("nope"));
        assert!(has_prompt("summarize"));
    }

    #[test]
    fn trailing_newline_is_stripped() {
        // Every source file ends with "\n"; rendering drops exactly one.
        let src = TEMPLATES
            .iter()
            .find(|(n, _)| *n == "semantic_search.md")
            .unwrap()
            .1;
        assert!(src.ends_with("results.\n"));
        let out = render(
            "semantic_search",
            json!({"query": "q", "sessions": "s", "limit": 5}),
        )
        .unwrap();
        assert!(out.ends_with("Return up to 5 results."));
        assert!(!out.ends_with('\n'));
    }

    #[test]
    fn missing_variable_renders_empty() {
        let out = render("semantic_search", json!({"query": "q", "sessions": "s"})).unwrap();
        assert!(out.ends_with("Return up to  results."), "{out:?}");
    }

    #[test]
    fn none_renders_as_the_literal_none() {
        let out = render(
            "qa_transcript",
            json!({"question": "q", "project": null, "date": null, "conversation": "c"}),
        )
        .unwrap();
        assert!(
            out.contains("- **Project:** None\n- **Date:** None\n"),
            "{out}"
        );
    }

    #[test]
    fn empty_string_is_falsy_in_if() {
        let with = render(
            "qa_transcript",
            json!({"question": "q", "project": "p", "date": "d", "summary": "S", "conversation": "c"}),
        )
        .unwrap();
        let without = render(
            "qa_transcript",
            json!({"question": "q", "project": "p", "date": "d", "summary": "", "conversation": "c"}),
        )
        .unwrap();
        let missing = render(
            "qa_transcript",
            json!({"question": "q", "project": "p", "date": "d", "conversation": "c"}),
        )
        .unwrap();
        assert!(with.contains("- **Summary:** S"));
        assert!(!without.contains("**Summary:**"));
        assert_eq!(without, missing);
    }

    #[test]
    fn markdown_is_not_html_escaped() {
        let out = render(
            "semantic_search",
            json!({"query": "<a & b> 'q'", "sessions": "\"x\"", "limit": 1}),
        )
        .unwrap();
        assert!(out.contains("\"<a & b> 'q'\""), "{out}");
        assert!(!out.contains("&amp;"));
    }

    #[test]
    fn sequences_render_with_container_repr() {
        let out = render(
            "semantic_search",
            json!({"query": ["a", "b"], "sessions": "s", "limit": 1}),
        )
        .unwrap();
        assert!(
            out.starts_with("Find sessions relevant to this query: \"['a', 'b']\""),
            "{out}"
        );
    }

    #[test]
    fn loop_index_is_one_based() {
        let out = render(
            "qa_synthesize",
            json!({"question": "q", "answers": [
                {"project": "p1", "date": "d1", "answer": "a1"},
                {"project": "p2", "date": "d2", "answer": "a2"}
            ]}),
        )
        .unwrap();
        assert!(out.contains("### Session 1: p1 (d1)"));
        assert!(out.contains("### Session 2: p2 (d2)"));
    }
}
