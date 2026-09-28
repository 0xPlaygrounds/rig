use std::io::{BufRead, Write};

use rig::AgentBuilder;
use rig::message::Message;
use rig::providers::gemini::{self, Gemini};

const HISTORY: &str = "chat-history.json";

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let gemini = Gemini::from_env()?;
    let agent = AgentBuilder::new(gemini.completion(gemini::GEMINI_3_8_FLASH))
        .preamble("You are a concise pair programmer. Keep answers under ten lines.")
        .build();

    // Signed text, sealed thoughts and native parts are plain serde data, so a
    // reload re-sends them unchanged.
    let mut history: Vec<Message> = match std::fs::read_to_string(HISTORY) {
        Ok(json) => serde_json::from_str(&json)?,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Vec::new(),
        Err(error) => return Err(error.into()),
    };

    let stdin = std::io::stdin();
    for line in stdin.lock().lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let response = agent.chat(line, &mut history).await?;
        println!("{}", response.output);
        std::io::stdout().flush()?;
        std::fs::write(HISTORY, serde_json::to_string(&history)?)?;
    }
    Ok(())
}
