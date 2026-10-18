//! `gem`: chat, draw and index on Gemini over gRPC.
//!
//! ```text
//! gem chat [--model M] [--thinking off|<budget>|low|high]
//! gem draw <prompt> [--model M]
//! gem index <dir> [--model M] [--dims N]   (then chats over the notes)
//! ```
//!
//! Needs `GEMINI_API_KEY`. The conversation is kept in `./.gem`.

use std::io::{BufRead, Write};
use std::path::Path;

use corpus_rf_06::{ChatEvent, ChatSettings, Gem, GemError, Notes, Thinking, stem_for};
use rig_core::providers::gemini::GeminiConfig;

/// A command-line failure.
type Failure = Box<dyn std::error::Error>;

/// The value after `--name` in `args`, if given.
fn flag<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    args.iter()
        .position(|arg| arg == name)
        .and_then(|at| args.get(at + 1))
        .map(String::as_str)
}

#[tokio::main]
async fn main() -> Result<(), Failure> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let gem = Gem::connect(&GeminiConfig::from_env()?).await?;
    let model = flag(&args, "--model");
    match args.first().map(String::as_str) {
        Some("chat") => chat(&gem, &args, None).await,
        Some("draw") => {
            let prompt = args.get(1).ok_or("usage: gem draw <prompt>")?;
            let drawing = gem
                .draw(model.unwrap_or("gemini-2.5-flash-image"), prompt, None)
                .await?;
            for path in drawing.save(Path::new("."), &stem_for(prompt))? {
                println!("saved {}", path.display());
            }
            for part in &drawing.parts {
                if let corpus_rf_06::DrawnPart::Text(text) = part {
                    println!("{text}");
                }
            }
            if !drawing.thoughts.is_empty() {
                println!("\x1b[2m{}\x1b[0m", drawing.thoughts);
            }
            println!("usage: {}", drawing.usage);
            Ok(())
        }
        Some("index") => {
            let dir = args.get(1).ok_or("usage: gem index <dir>")?;
            let dims = flag(&args, "--dims").map_or(Ok(768), str::parse)?;
            let notes = gem
                .index(
                    model.unwrap_or("gemini-embedding-001"),
                    dims,
                    Path::new(dir),
                )
                .await?;
            println!("indexed {} notes", notes.len());
            chat(&gem, &args, Some(notes)).await
        }
        _ => Err("usage: gem chat | gem draw <prompt> | gem index <dir>".into()),
    }
}

/// An interactive chat until end of input.
async fn chat(gem: &Gem, args: &[String], notes: Option<Notes>) -> Result<(), Failure> {
    let thinking = match flag(args, "--thinking") {
        Some(text) => Thinking::parse(text).ok_or("--thinking takes off, a budget or a level")?,
        None => Thinking::Budget(1024),
    };
    let settings = ChatSettings {
        model: flag(args, "--model")
            .filter(|_| args.first().map(String::as_str) == Some("chat"))
            .unwrap_or("gemini-2.5-flash")
            .to_owned(),
        thinking,
        conversation: "chat".to_owned(),
    };
    let chat = gem.chat(settings, rig_memory::DirStore::open(".gem")?, notes);
    for line in std::io::stdin().lock().lines() {
        let line = line?;
        let reply = chat
            .send(&line, |event| {
                match event {
                    ChatEvent::Thought(text) => print!("\x1b[2m{text}\x1b[0m"),
                    ChatEvent::Answer(text) => print!("{text}"),
                    ChatEvent::ToolCall(name) => println!("\n[{name}]"),
                    ChatEvent::Usage(usage) => println!("\n(usage: {usage})"),
                }
                let _ = std::io::stdout().flush();
            })
            .await;
        match reply {
            Ok(_) => println!(),
            Err(GemError::RateLimited { retry_after, .. }) => {
                println!("\nrate limited; retry after {retry_after:?}");
            }
            Err(error) => println!("\nerror: {error}"),
        }
    }
    Ok(())
}
