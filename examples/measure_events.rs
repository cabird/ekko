//! Measure what capped structured events would cost in the DB, per cap choice.
use std::path::Path;
use ekko::parsers;

fn cap_len(s: &str, max: usize) -> usize { s.chars().count().min(max) }

fn main() {
    let dirs = std::env::args().nth(1).unwrap();
    let mut grand = [0usize; 4]; let mut raw_total = 0u64; let mut n = 0;
    // caps: (content, tool_input, tool_output) per scenario
    let scenarios = [(10_000, 2_000, 4_000), (20_000, 2_000, 4_000), (10_000, 4_000, 10_000), (usize::MAX, usize::MAX, usize::MAX)];
    for entry in walkdir(Path::new(&dirs)) {
        let parser = parsers::parser_for("claude_code").unwrap();
        let parsed = parser.parse_transcript(&entry);
        if parsed.events.is_empty() { continue; }
        raw_total += std::fs::metadata(&entry).map(|m| m.len()).unwrap_or(0);
        n += 1;
        for (i, (c, ti, to)) in scenarios.iter().enumerate() {
            let mut sz = 0usize;
            for e in &parsed.events {
                sz += 120; // per-event JSON overhead estimate
                sz += cap_len(&e.content, *c);
                if let Some(v) = &e.tool_input { sz += cap_len(&v.to_string(), *ti); }
                if let Some(o) = &e.tool_output { sz += cap_len(o, *to); }
            }
            grand[i] += sz;
        }
    }
    println!("{n} transcripts, raw on disk: {:.1} MB", raw_total as f64/1e6);
    let names = ["10K/2K/4K", "20K/2K/4K", "10K/4K/10K", "uncapped"];
    for (i, name) in names.iter().enumerate() {
        println!("{name:>12}: {:.1} MB total, {:.0} KB avg/session", grand[i] as f64/1e6, grand[i] as f64/1024.0/n as f64);
    }
}
fn walkdir(root: &Path) -> Vec<std::path::PathBuf> {
    let mut out = vec![];
    if let Ok(rd) = std::fs::read_dir(root) {
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() { out.extend(walkdir(&p)); }
            else if p.extension().is_some_and(|x| x == "jsonl") { out.push(p); }
        }
    }
    out
}
