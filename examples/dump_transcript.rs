//! Parse one transcript with a named parser and print what ekko would index:
//! metadata, the tool timeline, and the conversation text.
//!
//! Usage: `cargo run --example dump_transcript -- <cli_source> <path> [--events]`
//!
//! Handy for checking a parser against a real file before it lands in the
//! database — nothing here touches the database or the network.

use std::path::Path;

use ekko::indexer::{extract_file_paths, generate_tool_timeline, is_interactive};
use ekko::parsers::parser_for;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (Some(cli_source), Some(path)) = (args.first(), args.get(1)) else {
        eprintln!("usage: dump_transcript <cli_source> <path> [--events]");
        std::process::exit(2);
    };
    let show_events = args.iter().any(|a| a == "--events");
    let Some(parser) = parser_for(cli_source) else {
        eprintln!("unknown cli_source {cli_source}");
        std::process::exit(2);
    };

    let parsed = parser.parse_transcript(Path::new(path));
    let m = &parsed.metadata;
    println!("session_id:    {}", m.session_id);
    println!("cli_source:    {}", m.cli_source);
    println!("title:         {:?}", m.title);
    println!("first_prompt:  {:?}", m.first_prompt);
    println!("project_path:  {:?}", m.project_path);
    println!("git_branch:    {:?}", m.git_branch);
    println!("model:         {:?}", m.model);
    println!(
        "entrypoint:    {:?} (interactive: {})",
        m.entrypoint,
        is_interactive(m)
    );
    println!("created_at:    {:?}", m.created_at);
    println!("modified_at:   {:?}", m.modified_at);
    println!("message_count: {}", m.message_count);
    println!("event_count:   {}", m.event_count);
    println!("files_touched: {:?}", extract_file_paths(&parsed.events));
    println!();
    println!("{}", generate_tool_timeline(&parsed.events));
    println!();
    println!("{}", parsed.get_conversation_text(true));

    if show_events {
        println!("---- events ----");
        for e in &parsed.events {
            println!(
                "{:>4} {:<11} {:<24} {:?} {:?}",
                e.sequence,
                e.event_type.as_str(),
                e.id,
                e.tool_name,
                e.tool_success
            );
        }
    }
}
