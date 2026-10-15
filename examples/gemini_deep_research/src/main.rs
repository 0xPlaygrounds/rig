//! Gemini Deep Research as a background job: start the interaction, keep
//! its id, then stream its progress (resuming after the last cursor when the
//! connection drops) or poll it, and print the report with its sources.
//!
//! ```sh
//! cargo run -p gemini_deep_research -- [--max] [--plan] [--poll]
//! ```
//!
//! `--max` runs Deep Research Max, `--plan` asks for a research plan to
//! review instead of the report, and `--poll` polls instead of streaming.

use std::time::Duration;

use anyhow::{Result, bail};
use futures::StreamExt;
use rig::completion::{CompletionRequest, CompletionResponse};
use rig::message::{AssistantContent, SourceLocation};
use rig::providers::gemini::Gemini;
use rig::providers::gemini::extension::{
    DeepResearchConfig, GeminiExt, GeminiOptions, InteractionStatus, InteractionsOptions,
    ThinkingSummaries, Visualization,
};
use rig::providers::gemini::interactions_api::streaming::StreamCursor;
use rig::providers::gemini::interactions_api::{
    DEEP_RESEARCH_MAX_PREVIEW_04_2026, DEEP_RESEARCH_PREVIEW_04_2026,
};
use rig::streaming::{Item, StreamEvent};

const PROMPT: &str = "Research the history of Google TPUs.";
const POLL_INTERVAL: Duration = Duration::from_secs(10);
const RECONNECTS: usize = 5;

#[tokio::main]
async fn main() -> Result<()> {
    let flag = |name: &str| std::env::args().any(|arg| arg == name);
    let agent = match flag("--max") {
        true => DEEP_RESEARCH_MAX_PREVIEW_04_2026,
        false => DEEP_RESEARCH_PREVIEW_04_2026,
    };
    let gemini = Gemini::from_env()?;

    // Start in the background: the reply comes at once and names the
    // interaction, which a job would store to poll or resume it later.
    let config = DeepResearchConfig::new()
        .thinking_summaries(ThinkingSummaries::Auto)
        .visualization(Visualization::Off)
        .collaborative_planning(flag("--plan"));
    let options = InteractionsOptions::new()
        .agent(agent)
        .background(true)
        .agent_config(config.into());
    let start =
        CompletionRequest::new(PROMPT).provider_option(GeminiOptions::new().interactions(options));
    let started = gemini.interactions(agent).call(start).await?;
    let Some(id) = started.response_id().map(str::to_owned) else {
        bail!("the interaction names no id");
    };
    println!("{agent} started {id}");

    if !flag("--poll") {
        stream(&gemini, &id).await?;
    }
    // A resumed stream holds only what followed its cursor, so the report is
    // read whole from the finished interaction.
    let report = poll(&gemini, &id).await?;
    if let Some(error) = &report.error {
        bail!("the research did not complete: {error}");
    }
    print_report(&report);
    Ok(())
}

/// The request the read wires take: they send no body, so its text is
/// never sent.
fn read() -> CompletionRequest {
    CompletionRequest::new("read")
}

/// Print the interaction's progress as it streams, resuming after the last
/// cursor when the stream drops.
async fn stream(gemini: &Gemini, id: &str) -> Result<()> {
    let mut cursor: Option<String> = None;
    for _ in 0..RECONNECTS {
        let mut stream = gemini
            .interaction_resumed(id, cursor.as_deref())
            .stream(read())?;
        while let Some(item) = stream.next().await {
            let item = match item {
                Ok(item) => item,
                Err(error) => {
                    eprintln!("\nThe stream dropped: {error}");
                    break;
                }
            };
            if let Some(event_id) = StreamCursor::of(&item).and_then(|resume| resume.event_id) {
                cursor = Some(event_id);
            }
            match item {
                Item::Event(StreamEvent::Reasoning { text, .. }) => println!("\n[thought] {text}"),
                Item::Event(StreamEvent::Text { text, .. }) => print!("{text}"),
                _ => {}
            }
        }
        match stream.finish().await {
            Ok(_) => return Ok(()),
            Err(error) => eprintln!("Resuming after {cursor:?}: {error}"),
        }
    }
    Ok(())
}

/// Poll the interaction until it stops running.
async fn poll(gemini: &Gemini, id: &str) -> Result<CompletionResponse> {
    loop {
        let interaction = gemini.interaction(id).call(read()).await?;
        let status = interaction
            .extras::<GeminiExt>()
            .transpose()?
            .and_then(|extras| extras.status);
        match status {
            Some(InteractionStatus::InProgress | InteractionStatus::Queued) => {
                println!("Still researching; polling again in {POLL_INTERVAL:?}");
                tokio::time::sleep(POLL_INTERVAL).await;
            }
            _ => return Ok(interaction),
        }
    }
}

/// The report's text and the sources its text cites.
fn print_report(report: &CompletionResponse) {
    let mut sources = Vec::new();
    println!("\n\n== Report ==");
    for content in &report.choice {
        if let AssistantContent::Text(text) = content {
            println!("{}", text.text);
            for source in text
                .citations()
                .iter()
                .flat_map(|citation| &citation.sources)
            {
                if let SourceLocation::Url { url } = &source.location
                    && !sources.contains(url)
                {
                    sources.push(url.clone());
                }
            }
        }
    }
    if !sources.is_empty() {
        println!("\n== Sources ==");
        for url in sources {
            println!("{url}");
        }
    }
}
