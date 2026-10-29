//! Fuzzy project-path suggestions.
//!
//! Similarity is a longest-matching-block ratio over normalized path
//! signatures, including an autojunk heuristic that ignores over-frequent
//! elements in sequences of 200+ characters. The exact scores are
//! load-bearing — they decide the `> 0.5` cutoff and the result ordering —
//! so the algorithm is pinned by goldens in the test module.

use std::collections::HashMap;

use super::{Database, DbResult};

/// Similarity of two strings over Unicode code points: twice the total
/// size of the matching blocks divided by the combined length, so 1.0 for
/// equal strings (including two empty ones) and 0.0 for no shared runs.
pub(crate) fn sequence_ratio(a: &str, b: &str) -> f64 {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let matches = matching_total(&a, &b);
    let length = a.len() + b.len();
    if length > 0 {
        2.0 * matches as f64 / length as f64
    } else {
        1.0
    }
}

/// Sum of matching-block sizes (the `matches` term in `ratio()`).
fn matching_total(a: &[char], b: &[char]) -> usize {
    // b2j: element -> positions in b, with the autojunk heuristic: for
    // len(b) >= 200, elements occupying > 1% of b are "popular" and removed
    // from the index (they can still be crossed by the extension loops
    // below).
    let n = b.len();
    let mut b2j: HashMap<char, Vec<usize>> = HashMap::new();
    for (j, &elt) in b.iter().enumerate() {
        b2j.entry(elt).or_default().push(j);
    }
    if n >= 200 {
        let ntest = n / 100 + 1;
        b2j.retain(|_, idxs| idxs.len() <= ntest);
    }

    // Iteratively collect matching blocks: recurse on both sides of each
    // longest match, accumulating sizes.
    let mut total = 0usize;
    let mut queue: Vec<(usize, usize, usize, usize)> = vec![(0, a.len(), 0, n)];
    while let Some((alo, ahi, blo, bhi)) = queue.pop() {
        let (i, j, k) = find_longest_match(a, b, &b2j, alo, ahi, blo, bhi);
        if k > 0 {
            total += k;
            if alo < i && blo < j {
                queue.push((alo, i, blo, j));
            }
            if i + k < ahi && j + k < bhi {
                queue.push((i + k, ahi, j + k, bhi));
            }
        }
    }
    total
}

/// Longest matching block between `a[alo..ahi]` and `b[blo..bhi]`. Nothing
/// is treated as junk, so only the plain extension pair is needed; popular
/// elements skipped by `b2j` are crossed here.
fn find_longest_match(
    a: &[char],
    b: &[char],
    b2j: &HashMap<char, Vec<usize>>,
    alo: usize,
    ahi: usize,
    blo: usize,
    bhi: usize,
) -> (usize, usize, usize) {
    let (mut besti, mut bestj, mut bestsize) = (alo, blo, 0usize);
    let mut j2len: HashMap<usize, usize> = HashMap::new();
    for (i, &elt) in a.iter().enumerate().take(ahi).skip(alo) {
        let mut newj2len: HashMap<usize, usize> = HashMap::new();
        if let Some(indices) = b2j.get(&elt) {
            for &j in indices {
                if j < blo {
                    continue;
                }
                if j >= bhi {
                    break;
                }
                let k = if j == 0 {
                    1
                } else {
                    j2len.get(&(j - 1)).copied().unwrap_or(0) + 1
                };
                newj2len.insert(j, k);
                if k > bestsize {
                    besti = i + 1 - k;
                    bestj = j + 1 - k;
                    bestsize = k;
                }
            }
        }
        j2len = newj2len;
    }
    // Extend the match outwards over equal elements.
    while besti > alo && bestj > blo && a[besti - 1] == b[bestj - 1] {
        besti -= 1;
        bestj -= 1;
        bestsize += 1;
    }
    while besti + bestsize < ahi
        && bestj + bestsize < bhi
        && a[besti + bestsize] == b[bestj + bestsize]
    {
        bestsize += 1;
    }
    (besti, bestj, bestsize)
}

/// Normalize a path for fuzzy comparison: split into components (root kept
/// as `/`, `.` components dropped, duplicate/trailing slashes collapsed,
/// `..` kept), strip `-`/`_` from the final segment, lowercase everything
/// and rejoin with `/`. Because the root is itself a component, an absolute
/// path comes back with a leading `//` — a quirk, but a stable one that the
/// goldens pin.
pub(crate) fn path_signature(p: &str) -> String {
    let parts = path_parts(p);
    if parts.is_empty() {
        return p.to_lowercase();
    }
    let last = parts[parts.len() - 1]
        .replace(['-', '_'], "")
        .to_lowercase();
    let head = parts[..parts.len() - 1].join("/").to_lowercase();
    format!("{head}/{last}")
}

/// Split a stored project path into components. Uses std
/// `Path::components` for `.`/`..`/duplicate-slash handling, plus one
/// fix-up:
///
/// POSIX (IEEE 1003.1 §4.13) gives a path beginning with **exactly two**
/// slashes an implementation-defined meaning and keeps the double slash as
/// the root component. THREE or more leading slashes collapse back to a
/// single `/`. Rust's `Path::components` normalizes all of them to one
/// `RootDir`, so the two-slash case is restored here:
///
/// ```text
/// "/foo/bar"   -> ("/",  "foo", "bar")
/// "//foo/bar"  -> ("//", "foo", "bar")
/// "///foo/bar" -> ("/",  "foo", "bar")
/// "//"         -> ("//",)
/// ```
/// which makes `path_signature("//foo/bar") == "///foo/bar"`.
fn path_parts(p: &str) -> Vec<String> {
    use std::path::Component;
    // Exactly two leading slashes → the root component is "//".
    let root = if p.starts_with("//") && !p.starts_with("///") {
        "//"
    } else {
        "/"
    };
    std::path::Path::new(p)
        .components()
        .filter_map(|c| match c {
            Component::RootDir => Some(root.to_string()),
            Component::Normal(s) => Some(s.to_string_lossy().into_owned()),
            Component::ParentDir => Some("..".to_string()),
            Component::CurDir => None,
            Component::Prefix(pre) => Some(pre.as_os_str().to_string_lossy().into_owned()),
        })
        .collect()
}

impl Database {
    /// Suggest similar project paths when an exact match fails: score every
    /// known project path with [`sequence_ratio`] over normalized
    /// signatures, sort descending by `(ratio, path)`, take the first `n`,
    /// then keep only those with ratio > 0.5.
    pub fn suggest_similar_paths(&self, query_path: &str, n: usize) -> DbResult<Vec<String>> {
        let projects = self.get_projects()?;
        let known_paths: Vec<String> = projects
            .into_iter()
            .map(|p| p.project_path)
            .filter(|p| !p.is_empty())
            .collect();
        if known_paths.is_empty() {
            return Ok(Vec::new());
        }

        let query_sig = path_signature(query_path);
        let mut scored: Vec<(f64, String)> = known_paths
            .into_iter()
            .map(|p| {
                let sig = path_signature(&p);
                (sequence_ratio(&query_sig, &sig), p)
            })
            .collect();

        // Ratio descending, then path descending.
        scored.sort_by(|x, y| {
            y.0.partial_cmp(&x.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| y.1.cmp(&x.1))
        });

        Ok(scored
            .into_iter()
            .take(n)
            .filter(|(ratio, _)| *ratio > 0.5)
            .map(|(_, p)| p)
            .collect())
    }
}
