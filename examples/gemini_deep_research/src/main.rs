use anyhow::Result;
use futures::StreamExt;
use rig::completion::{CompletionRequest, CompletionRequestBuilder};
use rig::driver::Bound;
use rig::prelude::*;
use rig::providers::gemini::Gemini;
use rig::providers::gemini::interactions_api::{
    AgentConfig, Content, Interaction, InteractionStatus, Step, ThinkingSummaries,
};
use rig::streaming::{Delta, StreamEvent};
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

/// The Deep Research request.
///
/// The Interactions wire reads the fields rig does not model from
/// `additional_params`, so `agent`, `background` and `agent_config` ride there;
/// `stream` is not among them, because the wire takes that from whether the
/// caller asked for `completion` or `stream`.
fn deep_research_request(
    agent: impl Into<String>,
    prompt: impl Into<String>,
    stream: bool,
) -> Result<CompletionRequest> {
    // Deep Research is selected by `agent`, which suppresses `model` in the
    // outgoing body — matching the official Gemini Deep Research examples.
    let mut params = serde_json::Map::from_iter([
        ("agent".to_owned(), json!(agent.into())),
        ("background".to_owned(), json!(true)),
    ]);

    if stream {
        // The Gemini docs recommend enabling thinking summaries for Deep
        // Research streams; otherwise a stream may only include final text.
        params.insert(
            "agent_config".to_owned(),
            serde_json::to_value(AgentConfig::DeepResearch {
                thinking_summaries: Some(ThinkingSummaries::Auto),
            })?,
        );
    }

    Ok(CompletionRequestBuilder::unbound(prompt.into())
        .additional_params(serde_json::Value::Object(params))
        .build())
}

fn extract_text(contents: &[Content]) -> String {
    contents
        .iter()
        .filter_map(|content| match content {
            Content::Text(text) => Some(text.text.clone()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn last_model_output_text(steps: &[Step]) -> Option<String> {
    steps.iter().rev().find_map(|step| match step {
        Step::ModelOutput { content } => {
            let text = extract_text(content);
            (!text.is_empty()).then_some(text)
        }
        _ => None,
    })
}

fn print_interaction_result(interaction: &Interaction) {
    match interaction.status.as_ref() {
        Some(InteractionStatus::Completed) => match last_model_output_text(&interaction.steps) {
            Some(text) => println!("{text}"),
            None => println!("No text output returned."),
        },
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
    gemini: &Bound<Gemini>,
    interaction_id: &str,
    request: &CompletionRequest,
) -> Result<Interaction> {
    let model = gemini
        .clone()
        .map_wire(|gemini| gemini.interaction(interaction_id));

    loop {
        let response = model.completion(request.clone()).await?;
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

fn handle_stream_event(state: &mut StreamState, event: StreamEvent) {
    match event {
        // Deep Research thinking summaries arrive as reasoning; the answer
        // itself as text. Both are deltas of a block, so the interesting part
        // of an event is its fragment.
        StreamEvent::BlockDelta {
            delta: Delta::Text { text },
            ..
        } => {
            print!("{text}");
            state.saw_text = true;
        }
        StreamEvent::BlockDelta {
            delta: Delta::Reasoning { text },
            ..
        } => {
            println!("\nThought: {text}");
        }
        // The terminal record carries the interaction id rig normalizes and,
        // under `interaction`, Gemini's own document for the finished run.
        StreamEvent::Final(final_record) => {
            if let Some(response_id) = final_record.response_id.as_deref() {
                state.interaction_id = Some(response_id.to_owned());
            }
            state.interaction = final_record
                .raw
                .get("interaction")
                .cloned()
                .and_then(|interaction| serde_json::from_value(interaction).ok());

            println!("\nResearch complete.");
            if !state.saw_text {
                match state
                    .interaction
                    .as_ref()
                    .and_then(|interaction| last_model_output_text(&interaction.steps))
                {
                    Some(text) => println!("{text}"),
                    None => println!("No text output returned."),
                }
            }
            state.is_complete = true;
        }
        StreamEvent::BlockStart { .. }
        | StreamEvent::BlockDelta { .. }
        | StreamEvent::BlockEnd { .. }
        | StreamEvent::Unknown(_) => {}
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    let use_streaming = std::env::args().any(|arg| arg == "--stream");
    let agent = deep_research_agent();
    let gemini = Gemini::from_env()?.bound()?;

    let request = deep_research_request(agent.clone(), DEFAULT_PROMPT, use_streaming)?;

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
                gemini
                    .clone()
                    .map_wire(|gemini| gemini.interactions(agent.as_str()))
                    .stream(request.clone())
                    .await
            } else if let Some(interaction_id) = state.interaction_id.as_deref() {
                gemini
                    .clone()
                    .map_wire(|gemini| gemini.interaction_resumed(interaction_id, None))
                    .stream(request.clone())
                    .await
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

            while let Some(event) = stream.next().await {
                match event {
                    Ok(event) => handle_stream_event(&mut state, event),
                    Err(err) => {
                        eprintln!("Stream error: {err}");
                        break;
                    }
                }

                if state.is_complete {
                    break;
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
            let probe = gemini
                .clone()
                .map_wire(|gemini| gemini.interaction(interaction_id.as_str()));
            let interaction: Interaction =
                serde_json::from_value(probe.completion(request.clone()).await?.raw)?;
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
    let opened = gemini
        .clone()
        .map_wire(|gemini| gemini.interactions(agent.as_str()))
        .completion(request.clone())
        .await?;
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
