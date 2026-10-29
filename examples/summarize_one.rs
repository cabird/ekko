//! Summarize + snapshot one transcript through the configured backend and
//! print the results. Handy for trying a backend without touching the DB:
//!
//!   EKKO_BACKEND=claude-cli cargo run --example summarize_one -- <transcript.jsonl>
use ekko::db::SessionRow;
use ekko::{llm, snapshot, summarize};

#[tokio::main]
async fn main() {
    let transcript = std::env::args().nth(1).expect("transcript path");
    let config = ekko::config::get_config();
    let resolved = llm::resolve_backend(config, llm::Role::Index);
    eprintln!(
        "backend={} model={} ({})",
        resolved.backend.name(),
        resolved.model,
        resolved.reason
    );

    let row = SessionRow {
        session_id: "example".into(),
        cli_source: "claude_code".into(),
        transcript_path: Some(transcript),
        project_path: Some("/Users/christianbird/my_repos/ekko".into()),
        ..Default::default()
    };
    let t = std::time::Instant::now();
    let summary = summarize::summarize_session(&row).await.expect("summarize");
    eprintln!("summarize: {:.1}s", t.elapsed().as_secs_f64());
    println!(
        "TITLE: {}\n\n{}\n",
        summary.title.unwrap_or_default(),
        summary.summary.unwrap_or_default()
    );

    let t = std::time::Instant::now();
    let snap = snapshot::generate_snapshot(&row).await.expect("snapshot");
    eprintln!("snapshot: {:.1}s", t.elapsed().as_secs_f64());
    println!("{}", &snap[..snap.len().min(1500)]);
    eprintln!("usage: {}", llm::get_usage_tracker().to_json());
}
