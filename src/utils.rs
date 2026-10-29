//! Shared utility functions.
//!
//! `split_text_by_tokens` lives in [`crate::tokens`] because it depends on
//! the tokenizer.

use chrono::{DateTime, Duration, FixedOffset, NaiveDate, NaiveDateTime, Timelike, Utc};
use rand::RngExt;
use std::collections::HashSet;
use std::path::{Component, Path, PathBuf};

/// Generate a unique short tag (like `S_AB12`).
///
/// Uses 2 uppercase letters + 2 digits for readability. Checks against
/// `used_tags` to ensure uniqueness within a single prompt context.
pub fn generate_unique_tag(used_tags: &HashSet<String>) -> String {
    let mut rng = rand::rng();
    loop {
        let letters: String = (0..2)
            .map(|_| rng.random_range(b'A'..=b'Z') as char)
            .collect();
        let numbers: String = (0..2)
            .map(|_| rng.random_range(b'0'..=b'9') as char)
            .collect();
        let tag = format!("S_{letters}{numbers}");
        if !used_tags.contains(&tag) {
            return tag;
        }
    }
}

/// Best-effort resolution of a path that need not exist: absolutize against
/// the cwd, resolve symlinks for path prefixes that exist, and normalize
/// `.`/`..` lexically for the rest.
pub fn resolve_path(p: &str) -> Option<PathBuf> {
    if p.contains('\0') {
        return None;
    }
    let path = Path::new(p);
    let abs = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir().ok()?.join(path)
    };
    if let Ok(c) = abs.canonicalize() {
        return Some(c);
    }
    let mut out = PathBuf::new();
    for comp in abs.components() {
        match comp {
            Component::Prefix(_) | Component::RootDir => out.push(comp.as_os_str()),
            Component::CurDir => {}
            Component::ParentDir => {
                out.pop();
            }
            Component::Normal(name) => {
                out.push(name);
                if let Ok(c) = out.canonicalize() {
                    out = c;
                }
            }
        }
    }
    Some(out)
}

/// Check if `project_path` is under `base_path` (or equal to it).
///
/// The project path is resolved; `base_path` is compared as given.
pub fn is_path_under(project_path: &str, base_path: &Path) -> bool {
    if project_path.is_empty() {
        return false;
    }
    match resolve_path(project_path) {
        // Path::starts_with is component-wise, so it covers both equality and
        // "base is an ancestor" without matching partial path segments.
        Some(proj) => proj.starts_with(base_path),
        None => false,
    }
}

/// Check if `project_path` equals `base_path` (project path resolved, base
/// compared as given).
pub fn paths_equal(project_path: &str, base_path: &Path) -> bool {
    if project_path.is_empty() {
        return false;
    }
    match resolve_path(project_path) {
        Some(proj) => proj == base_path,
        None => false,
    }
}

/// Parse an ISO 8601 datetime string in any of the shapes transcripts and
/// config actually carry: offset-aware, naive, or date-only, with `T` or a
/// space separator, extended or compact.
///
/// Returns a timezone-aware datetime (naive inputs are assumed UTC),
/// or `None` if parsing fails.
pub fn parse_datetime(value: &str) -> Option<DateTime<FixedOffset>> {
    if value.is_empty() {
        return None;
    }
    // Every 'Z' becomes '+00:00', not just a trailing one.
    let value = value.replace('Z', "+00:00");

    // Offset-aware formats. %#z accepts +HH, +HHMM, and +HH:MM.
    const AWARE: [&str; 5] = [
        "%Y-%m-%dT%H:%M:%S%.f%#z",
        "%Y-%m-%d %H:%M:%S%.f%#z",
        "%Y-%m-%dT%H:%M%#z",
        "%Y-%m-%d %H:%M%#z",
        "%Y%m%dT%H%M%S%.f%#z",
    ];
    for fmt in AWARE {
        if let Ok(dt) = DateTime::parse_from_str(&value, fmt) {
            return Some(dt);
        }
    }
    // Naive formats → assume UTC.
    const NAIVE: [&str; 5] = [
        "%Y-%m-%dT%H:%M:%S%.f",
        "%Y-%m-%d %H:%M:%S%.f",
        "%Y-%m-%dT%H:%M",
        "%Y-%m-%d %H:%M",
        "%Y%m%dT%H%M%S",
    ];
    for fmt in NAIVE {
        if let Ok(dt) = NaiveDateTime::parse_from_str(&value, fmt) {
            return Some(dt.and_utc().fixed_offset());
        }
    }
    // Date-only → midnight UTC.
    for fmt in ["%Y-%m-%d", "%Y%m%d"] {
        if let Ok(d) = NaiveDate::parse_from_str(&value, fmt) {
            return Some(d.and_hms_opt(0, 0, 0)?.and_utc().fixed_offset());
        }
    }
    None
}

/// Resolve `days`/`start_date`/`end_date` into a `(start_dt, end_dt)` tuple.
///
/// If `start_date` or `end_date` are provided (non-empty) they take
/// precedence over `days`. Date-only strings are treated as midnight UTC
/// (start) or end-of-day 23:59:59 (end).
pub fn parse_date_filter(
    days: Option<i64>,
    start_date: Option<&str>,
    end_date: Option<&str>,
) -> (Option<DateTime<FixedOffset>>, Option<DateTime<FixedOffset>>) {
    // An empty string counts as absent, not as an unparseable date.
    let start_date = start_date.filter(|s| !s.is_empty());
    let end_date = end_date.filter(|s| !s.is_empty());

    if start_date.is_some() || end_date.is_some() {
        let mut start_dt = None;
        let mut end_dt = None;
        if let Some(s) = start_date {
            start_dt = parse_datetime(s);
            // Date-only string (no 'T') → midnight UTC
            if start_dt.is_none() && s.chars().count() <= 10 {
                start_dt = parse_datetime(&format!("{s}T00:00:00+00:00"));
            }
        }
        if let Some(e) = end_date {
            end_dt = parse_datetime(e);
            if end_dt.is_none() && e.chars().count() <= 10 {
                // Date-only string → end of day
                end_dt = parse_datetime(&format!("{e}T23:59:59+00:00"));
            } else if end_dt.is_some() && e.chars().count() <= 10 {
                // Successfully parsed but was date-only — bump to end of day
                end_dt = end_dt
                    .and_then(|d| d.with_hour(23))
                    .and_then(|d| d.with_minute(59))
                    .and_then(|d| d.with_second(59));
            }
        }
        return (start_dt, end_dt);
    }

    if let Some(d) = days {
        let start = Utc::now() - Duration::days(d);
        return (Some(start.fixed_offset()), None);
    }

    (None, None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    fn utc(y: i32, mo: u32, d: u32, h: u32, mi: u32, s: u32) -> DateTime<FixedOffset> {
        Utc.with_ymd_and_hms(y, mo, d, h, mi, s)
            .unwrap()
            .fixed_offset()
    }

    #[test]
    fn unique_tag_format() {
        let used = HashSet::new();
        for _ in 0..50 {
            let tag = generate_unique_tag(&used);
            let b = tag.as_bytes();
            assert_eq!(tag.len(), 6, "tag {tag}");
            assert_eq!(&tag[..2], "S_");
            assert!(b[2].is_ascii_uppercase() && b[3].is_ascii_uppercase());
            assert!(b[4].is_ascii_digit() && b[5].is_ascii_digit());
        }
    }

    #[test]
    fn unique_tag_avoids_used() {
        // Exclude everything except one specific tag; the generator must
        // eventually return the sole remaining tag.
        let mut used = HashSet::new();
        for a in b'A'..=b'Z' {
            for b in b'A'..=b'Z' {
                for n in 0..100 {
                    let tag = format!("S_{}{}{:02}", a as char, b as char, n);
                    if tag != "S_QQ07" {
                        used.insert(tag);
                    }
                }
            }
        }
        assert_eq!(generate_unique_tag(&used), "S_QQ07");
    }

    #[test]
    fn path_under_basic() {
        // Callers resolve the base before comparing (see search.rs /
        // memory/mod.rs); mirror that here or /tmp -> /private/tmp on
        // macOS makes the comparison asymmetric.
        let tmp = resolve_path("/tmp").unwrap();
        assert!(is_path_under("/tmp", Path::new("/")));
        assert!(is_path_under("/tmp", &tmp));
        assert!(!is_path_under("/tmp", &tmp.join("deeper")));
        assert!(!is_path_under("", Path::new("/")));
        // Non-existent paths still resolve lexically.
        assert!(is_path_under(
            "/no/such/dir/x/../y",
            Path::new("/no/such/dir")
        ));
        // Component-wise, not string-prefix: /ab is not under /a.
        assert!(!is_path_under("/abc", Path::new("/ab")));
    }

    #[test]
    fn paths_equal_basic() {
        let tmp = resolve_path("/tmp").unwrap();
        assert!(paths_equal("/tmp", &tmp));
        assert!(paths_equal("/tmp/../tmp", &tmp));
        assert!(!paths_equal("/tmp", Path::new("/")));
        assert!(!paths_equal("", &tmp));
        assert!(paths_equal("/no/such/a/./b", Path::new("/no/such/a/b")));
    }

    // The expected values below are pinned; update them deliberately when
    // parse_datetime / parse_date_filter behavior changes on purpose.

    #[test]
    fn parse_datetime_formats() {
        assert_eq!(
            parse_datetime("2024-01-15T10:30:00Z"),
            Some(utc(2024, 1, 15, 10, 30, 0))
        );
        // Offset preserved: 10:30+05:30 == 05:00 UTC.
        let dt = parse_datetime("2024-01-15T10:30:00+05:30").unwrap();
        assert_eq!(dt.offset().local_minus_utc(), 19800);
        assert_eq!(dt.with_timezone(&Utc), utc(2024, 1, 15, 5, 0, 0));
        // Space separator, naive → UTC.
        assert_eq!(
            parse_datetime("2024-01-15 10:30:00"),
            Some(utc(2024, 1, 15, 10, 30, 0))
        );
        // Date-only → midnight UTC.
        assert_eq!(parse_datetime("2024-01-15"), Some(utc(2024, 1, 15, 0, 0, 0)));
        // Minute precision.
        assert_eq!(
            parse_datetime("2024-01-15T10:30"),
            Some(utc(2024, 1, 15, 10, 30, 0))
        );
        // Fractional seconds.
        let dt = parse_datetime("2024-01-15T10:30:00.123456Z").unwrap();
        assert_eq!(dt.timestamp_subsec_micros(), 123456);
        // Compact offset ±HHMM.
        let dt = parse_datetime("2024-01-15T10:30:00-0800").unwrap();
        assert_eq!(dt.offset().local_minus_utc(), -28800);
        // Compact basic format, no separators.
        assert_eq!(
            parse_datetime("20240115T103000"),
            Some(utc(2024, 1, 15, 10, 30, 0))
        );
        assert_eq!(parse_datetime("not-a-date"), None);
        assert_eq!(parse_datetime(""), None);
        assert_eq!(parse_datetime("2024-13-45"), None);
    }

    #[test]
    fn date_filter_date_only_bounds() {
        // Start is midnight UTC, end is 23:59:59 UTC on the end date.
        let (s, e) = parse_date_filter(None, Some("2024-01-15"), Some("2024-01-20"));
        assert_eq!(s, Some(utc(2024, 1, 15, 0, 0, 0)));
        assert_eq!(e, Some(utc(2024, 1, 20, 23, 59, 59)));
    }

    #[test]
    fn date_filter_datetime_inputs() {
        let (s, e) = parse_date_filter(None, Some("2024-01-15T08:00:00Z"), None);
        assert_eq!(s, Some(utc(2024, 1, 15, 8, 0, 0)));
        assert_eq!(e, None);
        // Full datetime end is NOT bumped to end of day.
        let (s, e) = parse_date_filter(None, None, Some("2024-01-20T23:00:00+00:00"));
        assert_eq!(s, None);
        assert_eq!(e, Some(utc(2024, 1, 20, 23, 0, 0)));
    }

    #[test]
    fn date_filter_junk_and_empty() {
        // Unparseable start (even after the T00:00:00 retry) → (None, None).
        assert_eq!(parse_date_filter(None, Some("junk"), None), (None, None));
        // Empty strings are falsy → fall through to days/none.
        assert_eq!(parse_date_filter(None, Some(""), Some("")), (None, None));
        assert_eq!(parse_date_filter(None, None, None), (None, None));
    }

    #[test]
    fn date_filter_days() {
        let before = Utc::now() - Duration::days(7);
        let (s, e) = parse_date_filter(Some(7), None, None);
        let after = Utc::now() - Duration::days(7);
        let s = s.unwrap().with_timezone(&Utc);
        assert!(s >= before && s <= after);
        assert_eq!(e, None);
        // Dates take precedence over days.
        let (s, _) = parse_date_filter(Some(7), Some("2024-01-15"), None);
        assert_eq!(s, Some(utc(2024, 1, 15, 0, 0, 0)));
    }
}

// ---------------------------------------------------------------------------
// Text and number helpers shared across modules
// ---------------------------------------------------------------------------

/// Round `x` to `digits` decimal places: correctly-rounded decimal, ties to
/// even. Formatting does the rounding on the binary value and parsing back
/// yields the nearest double — more accurate than a naive multiply/round,
/// which double-rounds.
pub(crate) fn round_decimal(x: f64, digits: usize) -> f64 {
    format!("{x:.digits$}").parse().unwrap_or(x)
}

/// The whitespace set ekko splits and trims on: the Unicode `White_Space`
/// property PLUS the four file/group/record/unit separator controls
/// U+001C..U+001F, which Rust's [`char::is_whitespace`] excludes. Transcripts
/// carry those separators inside pasted terminal output, and treating them as
/// text would glue tokens together.
///
/// Over 0..0x100 the set is exactly
/// `{9,10,11,12,13,28,29,30,31,32,133,160}`; above U+00FF it is just
/// `White_Space` (U+1680, U+2028, U+2029, U+202F, U+205F and U+3000 are in;
/// U+180E and U+200B are not).
pub(crate) fn is_extended_whitespace(c: char) -> bool {
    c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)
}

/// Trim [`is_extended_whitespace`] from both ends.
pub(crate) fn strip_ws(s: &str) -> &str {
    s.trim_matches(is_extended_whitespace)
}

/// Trim [`is_extended_whitespace`] from the end only.
pub(crate) fn rstrip_ws(s: &str) -> &str {
    s.trim_end_matches(is_extended_whitespace)
}

#[cfg(test)]
mod text_semantics_tests {
    use super::*;

    #[test]
    fn round_decimal_ties_to_even() {
        // Midpoints in decimal round to even; the binary values sit within an
        // ulp of the tie, so this pins the format-then-parse strategy rather
        // than a naive multiply/round.
        assert_eq!(round_decimal(0.125, 2), 0.12);
        assert_eq!(round_decimal(0.375, 2), 0.38);
        assert_eq!(round_decimal(2.5, 0), 2.0);
        assert_eq!(round_decimal(3.5, 0), 4.0);
        // The two call-site precisions.
        assert_eq!(round_decimal(0.1234567, 6), 0.123457);
        assert_eq!(round_decimal(9.049_999, 2), 9.05);
        assert!(round_decimal(f64::NAN, 6).is_nan());
    }

    #[test]
    fn is_extended_whitespace_covers_the_split_set() {
        let expected: std::collections::HashSet<u32> =
            [9, 10, 11, 12, 13, 28, 29, 30, 31, 32, 133, 160].into_iter().collect();
        for code in 0u32..0x100 {
            let c = char::from_u32(code).unwrap();
            assert_eq!(is_extended_whitespace(c), expected.contains(&code), "U+{code:04X}");
        }
        assert!(is_extended_whitespace('\u{3000}'));
        assert!(!is_extended_whitespace('\u{200B}'));
    }

    #[test]
    fn strip_helpers_use_the_extended_set() {
        assert_eq!(strip_ws("\u{1c} x \u{1f}"), "x");
        assert_eq!(rstrip_ws(" x \u{1c}"), " x");
        assert_eq!(strip_ws(""), "");
    }
}
