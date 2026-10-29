use ekko::{parsers, snapshot, summarize, tokens};
use std::path::Path;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let transcript = &args[1];
    let out_dir = &args[2];
    let parser = parsers::parser_for("claude_code").unwrap();
    let parsed = parser.parse_transcript(Path::new(transcript));
    let conversation = parsed.get_conversation_text(true);
    println!("{}", tokens::count_tokens(&conversation));
    let p = summarize::summarize_prompt(&conversation).unwrap();
    std::fs::write(format!("{out_dir}/summarize.txt"), &p).unwrap();
    let meta = snapshot::SnapshotMeta {
        project_path: args.get(3).cloned().unwrap_or("Unknown".into()),
        cli_source: "Claude Code".into(),
        git_branch: "Unknown".into(),
        session_date: "Unknown".into(),
        brief_title: "Session".into(),
    };
    let p = snapshot::snapshot_prompt(&conversation, &meta).unwrap();
    std::fs::write(format!("{out_dir}/snapshot.txt"), &p).unwrap();
}
