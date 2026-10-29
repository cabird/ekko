//! Tokenizer helpers: special-token escaping, token counting and
//! token-budgeted text splitting.
//!
//! The splitter lives here rather than in [`crate::utils`] because it is
//! built on `count_tokens`.

use regex::Regex;
use std::sync::LazyLock;
use tiktoken_rs::CoreBPE;

// The word class is spelled out as `[\p{L}\p{N}_]` on purpose: a special
// token's body is alphanumerics (categories L* and N*, including Nl/No such
// as '²') plus the literal underscore, and nothing else. Rust's default `\w`
// ([\p{Alphabetic}\p{M}\p{Nd}\p{Pc}\p{Join_Control}]) would also match
// combining marks, other connector punctuation (e.g. '‿') and join controls,
// which must not be escaped.
static SPECIAL_TOKEN_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"<\|([\p{L}\p{N}_]+)\|>").expect("valid regex"));

static BPE: LazyLock<CoreBPE> =
    LazyLock::new(|| tiktoken_rs::cl100k_base().expect("cl100k_base tokenizer loads"));

/// Escape special tokens like `<|endoftext|>` to prevent LLM confusion.
pub fn escape_special_tokens(text: &str) -> String {
    SPECIAL_TOKEN_RE.replace_all(text, "<| $1 |>").into_owned()
}

/// Count tokens in text using tiktoken (cl100k_base), escaping special
/// tokens first.
pub fn count_tokens(text: &str) -> usize {
    BPE.encode_ordinary(&escape_special_tokens(text)).len()
}

/// Split text into chunks that each fit within `max_tokens`, on line
/// boundaries.
///
/// Accumulates lines until adding another would exceed the budget, then
/// starts a new chunk. Returns at least one chunk; a single line larger
/// than the budget becomes its own oversized chunk.
pub fn split_text_by_tokens(text: &str, max_tokens: usize) -> Vec<String> {
    let mut chunks: Vec<String> = Vec::new();
    let mut buf: Vec<&str> = Vec::new();
    let mut buf_tokens: usize = 0;

    for line in text.split('\n') {
        let line_tokens = count_tokens(&format!("{line}\n"));
        if !buf.is_empty() && buf_tokens + line_tokens > max_tokens {
            chunks.push(buf.join("\n"));
            buf.clear();
            buf_tokens = 0;
        }
        buf.push(line);
        buf_tokens += line_tokens;
    }

    if !buf.is_empty() {
        chunks.push(buf.join("\n"));
    }

    if chunks.is_empty() {
        vec![text.to_string()]
    } else {
        chunks
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // The expected values below are pinned; update them deliberately when
    // behavior changes intentionally.

    #[test]
    fn escape_basic() {
        assert_eq!(escape_special_tokens("<|endoftext|>"), "<| endoftext |>");
        assert_eq!(escape_special_tokens("a<|x|>b<|y|>c"), "a<| x |>b<| y |>c");
        assert_eq!(
            escape_special_tokens("<|fim_prefix|><|fim_suffix|>"),
            "<| fim_prefix |><| fim_suffix |>"
        );
        assert_eq!(escape_special_tokens("no tokens"), "no tokens");
    }

    #[test]
    fn escape_non_matches() {
        // Space is not a word char; empty group requires +.
        assert_eq!(escape_special_tokens("<|a b|>"), "<|a b|>");
        assert_eq!(escape_special_tokens("<||>"), "<||>");
    }

    #[test]
    fn escape_unicode_word_chars() {
        // All L*/N* categories count as word characters here.
        assert_eq!(escape_special_tokens("<|日本語|>"), "<| 日本語 |>");
        assert_eq!(escape_special_tokens("<|²|>"), "<| ² |>"); // No category
        assert_eq!(escape_special_tokens("<|_a1|>"), "<| _a1 |>");
        // '‿' (Pc) is not a word character here — must not be escaped.
        assert_eq!(escape_special_tokens("<|‿|>"), "<|‿|>");
    }

    #[test]
    fn token_counts_are_pinned() {
        assert_eq!(count_tokens(""), 0);
        assert_eq!(count_tokens("hello world"), 2);
        // Escaped before counting: "<| endoftext |>" is 6 tokens.
        assert_eq!(count_tokens("<|endoftext|>"), 6);
        assert_eq!(
            count_tokens("The quick brown fox jumps over the lazy dog."),
            10
        );
        assert_eq!(count_tokens("línea con acentos y 日本語のテキスト"), 16);
        assert_eq!(count_tokens("fn main() {\n    println!(\"hi\");\n}\n"), 10);
    }

    #[test]
    fn split_empty_text() {
        // Splitting "" on '\n' yields one empty line → one empty chunk.
        assert_eq!(split_text_by_tokens("", 10), vec!["".to_string()]);
    }

    #[test]
    fn split_single_huge_line() {
        // A single line over budget is never split — one oversized chunk.
        let text = "word ".repeat(200);
        let chunks = split_text_by_tokens(&text, 10);
        assert_eq!(chunks, vec![text]);
    }

    #[test]
    fn split_multi_line() {
        // "line {i}\n" is 4 tokens; budget 8 → two lines per chunk.
        let text = (0..10)
            .map(|i| format!("line {i}"))
            .collect::<Vec<_>>()
            .join("\n");
        assert_eq!(
            split_text_by_tokens(&text, 8),
            vec![
                "line 0\nline 1",
                "line 2\nline 3",
                "line 4\nline 5",
                "line 6\nline 7",
                "line 8\nline 9",
            ]
        );
    }

    #[test]
    fn split_trailing_newline() {
        // "a\nb\n" splits into ['a', 'b', '']; the empty tail joins the last
        // chunk, reproducing the trailing newline.
        assert_eq!(
            split_text_by_tokens("a\nb\n", 3),
            vec!["a".to_string(), "b\n".to_string()]
        );
    }

    #[test]
    fn split_exact_boundary() {
        // Each "aa\n"/"bb\n" is 2 tokens; budget 2 → adding the second line
        // would exceed (2+2 > 2) → flush per line.
        assert_eq!(
            split_text_by_tokens("aa\nbb", 2),
            vec!["aa".to_string(), "bb".to_string()]
        );
        // Budget 4 fits both exactly (2+2 == 4, not >).
        assert_eq!(
            split_text_by_tokens("aa\nbb", 4),
            vec!["aa\nbb".to_string()]
        );
    }
}
