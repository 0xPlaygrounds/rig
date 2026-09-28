use anyhow::Result;
use futures::StreamExt;
use rig::completion::CompletionRequest;
use rig::providers::gemini::Gemini;
use rig::providers::gemini::interactions_api::api::{self, Interaction, InteractionStatus};
use rig::streaming::{Item, StreamEvent};
use serde_json::json;
use std::time::Duration;
use tokio::time::sleep;
use tracing_subscriber::EnvFilter;

/// Known-working Deep Research agent for this example.
///
/// Override with `GEMINI_DEEP_RESEARCH_AGENT` to try another documented variant,
/// such as `deep-research-preview-04-2026` or
/// `deep-research-max-preview-04-2026`.
const DEFAULT_DEEP_RESEARCH_AGENT: &str = "deep-research-pro-preview-12-2025";
const DEFAULT_PROMPT: &str = "Research the history of Google TPUs.";
const STREAM_RETRY_DELAY_SECS: u64 = 2;
const POLL_INTERVAL_SECS: u64 = 10;

fn deep_research_agent() -> String {
    std::env::var("GEMINI_DEEP_RESEARCH_AGENT")
        .ok()
        .filter(|agent| !agent.trim().is_empty())
        .unwrap_or_else(|| DEFAULT_DEEP_RESEARCH_AGENT.to_string())
}

/// The Deep Research settings: `agent` replaces the model in the request body,
/// and the run goes on in the background.
fn deep_research_settings(agent: &str, stream: bool) -> api::RequestSettings {
    api::RequestSettings {
        agent: Some(agent.to_owned()),
        background: Some(true),
        // The Gemini docs recommend thinking summaries for Deep Research
        // streams; otherwise a stream may only include final text.
        agent_config: stream
            .then(|| json!({"type": "deep-research", "thinking_summaries": "auto"})),
        ..Default::default()
    }
}

fn print_interaction_result(interaction: &Interaction) {
    match interaction.status.as_ref() {
        Some(InteractionStatus::Completed) => {
            let text = interaction.text();
            if text.is_empty() {
                println!("No text output returned.");
            } else {
                println!("{text}");
            }
        }
        Some(status) => println!("Research ended with status: {status:?}"),
        None => println!("Research ended without a status."),
    }
}

/// Poll a background interaction until it reaches a terminal state.
///
/// The poll wire's reply *is* the interaction document, so it arrives whole on
/// [`CompletionResponse::raw`](rig::completion::CompletionResponse::raw): the
/// normalized halves (`choice`, `usage`) are the folded turn, and the
/// provider's own lifecycle fields — `status`, `steps` — are read back out of
/// `raw` by deserializing Gemini's own type.
async fn poll_until_terminal(
    gemini: &Gemini,
    interaction_id: &str,
    request: &CompletionRequest,
) -> Result<Interaction> {
    let model = gemini.interaction(interaction_id);

    loop {
        let response = model.call(request.clone()).await?;
        let interaction: Interaction = serde_json::from_value(response.raw)?;
        if interaction.is_terminal() {
            return Ok(interaction);
        }

        println!(
            "Status: {:?}. Polling again in {POLL_INTERVAL_SECS}s...",
            interaction.status.unwrap_or(InteractionStatus::InProgress)
        );
        sleep(Duration::from_secs(POLL_INTERVAL_SECS)).await;
    }
}

#[derive(Default)]
struct StreamState {
    interaction_id: Option<String>,
    is_complete: bool,
    saw_text: bool,
    /// The interaction document off the terminal record, when the stream
    /// reached one.
    interaction: Option<Interaction>,
}

fn handle_stream_item(state: &mut StreamState, item: Item<StreamEvent>) {
    match item {
        // Deep Research thinking summaries arrive as reasoning; the answer
        // itself as text. The interesting part of an event is its fragment.
        Item::Event(StreamEvent::Text { text, .. }) => {
            print!("{text}");
            state.saw_text = true;
        }
        Item::Event(StreamEvent::Reasoning { text, .. }) => {
            println!("\nThought: {text}");
        }
        Item::Event(_) | Item::Unknown(_) => {}
    }
}

/// The finished stream's response carries the interaction id rig normalizes
/// and, as its `raw`, Gemini's own document for the finished run.
fn finish_research(state: &mut StreamState, response: rig::completion::CompletionResponse) {
    if let Some(response_id) = response.response_id.as_deref() {
        state.interaction_id = Some(response_id.to_owned());
    }
    state.interaction = serde_json::from_value(response.raw).ok();

    println!("\nResearch complete.");
    if !state.saw_text
        && let Some(interaction) = &state.interaction
    {
        print_interaction_result(interaction);
    }
    state.is_complete = true;
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    let use_streaming = std::env::args().any(|arg| arg == "--stream");
    let agent = deep_research_agent();
    let gemini = Gemini::from_env()?;

    let request = CompletionRequest::new(DEFAULT_PROMPT);
    // The wire that opens an interaction, built once for either surface.
    let interactions = gemini
        .interactions(agent.as_str())
        .settings(deep_research_settings(&agent, use_streaming));

    if use_streaming {
        println!("== Deep Research (streaming) ==");
        println!("Agent: {agent}");
        let mut state = StreamState::default();
        let mut attempt = 0usize;

        loop {
            // The first attempt opens the interaction; a reconnect addresses
            // the one already running by id. They are two different wires, so
            // the branches meet at the opened stream rather than at the model.
            let opened = if attempt == 0 {
                interactions.stream(request.clone())
            } else if let Some(interaction_id) = state.interaction_id.as_deref() {
                gemini
                    .interaction_resumed(interaction_id, None)
                    .stream(request.clone())
            } else {
                eprintln!("Stream closed before an interaction id was received.");
                break;
            };

            let mut stream = match opened {
                Ok(stream) => stream,
                Err(err) => {
                    eprintln!("Failed to open stream: {err}");
                    break;
                }
            };

            let mut failed = false;
            while let Some(item) = stream.next().await {
                match item {
                    Ok(item) => handle_stream_item(&mut state, item),
                    Err(err) => {
                        eprintln!("Stream error: {err}");
                        failed = true;
                        break;
                    }
                }
            }
            if !failed {
                match stream.finish().await {
                    Ok(response) => finish_research(&mut state, response),
                    Err(err) => eprintln!("Stream ended early: {err}"),
                }
            }

            if state.is_complete {
                break;
            }

            let Some(interaction_id) = state.interaction_id.clone() else {
                break;
            };

            // Official Deep Research guidance recommends checking the background
            // interaction status before reconnecting a dropped/expired stream.
            let probe = gemini.interaction(interaction_id.as_str());
            let interaction: Interaction =
                serde_json::from_value(probe.call(request.clone()).await?.raw)?;
            if interaction.is_terminal() {
                println!("Stream ended after interaction reached a terminal state.");
                print_interaction_result(&interaction);
                break;
            }

            attempt += 1;
            println!(
                "\nStream interrupted while status was {:?}. Reconnecting in {STREAM_RETRY_DELAY_SECS}s...",
                interaction.status.unwrap_or(InteractionStatus::InProgress)
            );
            sleep(Duration::from_secs(STREAM_RETRY_DELAY_SECS)).await;
        }

        if !state.is_complete
            && let Some(interaction_id) = state.interaction_id.as_deref()
        {
            println!("Switching to polling for interaction {interaction_id}...");
            let interaction = poll_until_terminal(&gemini, interaction_id, &request).await?;
            print_interaction_result(&interaction);
        }

        if let Some(interaction_id) = state.interaction_id {
            println!("Interaction ID: {interaction_id}");
        }

        return Ok(());
    }

    println!("== Deep Research (background polling) ==");
    println!("Agent: {agent}");
    let opened = interactions.call(request.clone()).await?;
    // rig normalizes the interaction id onto `response_id`, so opening a
    // background run needs no reach into `raw`.
    let Some(interaction_id) = opened.response_id else {
        println!("No interaction id returned; aborting.");
        return Ok(());
    };
    println!("Research started: {interaction_id}");

    let interaction = poll_until_terminal(&gemini, &interaction_id, &request).await?;
    print_interaction_result(&interaction);

    Ok(())
}
