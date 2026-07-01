use anyhow::Result;
use futures::StreamExt;
use rig::prelude::*;
use rig::providers::gemini::{
    self,
    interactions_api::{
        AgentConfig, Content, ContentDelta, CreateInteractionRequest, Interaction,
        InteractionInput, InteractionSseEvent, InteractionStatus, Step, TextDelta,
        ThinkingSummaries, ThoughtSummaryContent, ThoughtSummaryDelta,
    },
};
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

fn deep_research_request(
    agent: impl Into<String>,
    prompt: impl Into<String>,
    stream: bool,
) -> CreateInteractionRequest {
    CreateInteractionRequest {
        model: None,
        agent: Some(agent.into()),
        input: InteractionInput::Text(prompt.into()),
        system_instruction: None,
        tools: None,
        response_format: None,
        response_mime_type: None,
        stream: stream.then_some(true),
        store: None,
        background: Some(true),
        generation_config: None,
        agent_config: stream.then_some(AgentConfig::DeepResearch {
            // The Gemini docs recommend enabling thinking summaries for Deep
            // Research streams; otherwise a stream may only include final text.
            thinking_summaries: Some(ThinkingSummaries::Auto),
        }),
        response_modalities: None,
        previous_interaction_id: None,
        additional_params: None,
    }
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

async fn poll_until_terminal(
    client: &gemini::InteractionsClient,
    interaction_id: &str,
) -> Result<Interaction> {
    loop {
        let interaction = client.get_interaction(interaction_id).await?;
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

fn track_event_id(last_event_id: &mut Option<String>, event_id: Option<String>) {
    if let Some(event_id) = event_id {
        *last_event_id = Some(event_id);
    }
}

#[derive(Default)]
struct StreamState {
    interaction_id: Option<String>,
    last_event_id: Option<String>,
    is_complete: bool,
    saw_text: bool,
}

fn handle_stream_event(state: &mut StreamState, event: InteractionSseEvent) {
    match event {
        InteractionSseEvent::InteractionCreated {
            interaction,
            event_id,
        } => {
            track_event_id(&mut state.last_event_id, event_id);
            state.interaction_id = Some(interaction.id.clone());
            println!("Interaction started: {}", interaction.id);
        }
        InteractionSseEvent::StepStart { step, event_id, .. } => {
            track_event_id(&mut state.last_event_id, event_id);
            if let Step::ModelOutput { content } = step {
                let text = extract_text(&content);
                if !text.is_empty() {
                    print!("{text}");
                    state.saw_text = true;
                }
            }
        }
        InteractionSseEvent::StepDelta {
            delta, event_id, ..
        } => {
            track_event_id(&mut state.last_event_id, event_id);
            match delta {
                ContentDelta::Text(TextDelta {
                    text: Some(text), ..
                }) => {
                    print!("{text}");
                    state.saw_text = true;
                }
                ContentDelta::ThoughtSummary(ThoughtSummaryDelta {
                    content: ThoughtSummaryContent::Text(text),
                }) => {
                    println!("\nThought: {}", text.text);
                }
                _ => {}
            }
        }
        InteractionSseEvent::InteractionCompleted {
            interaction,
            event_id,
        } => {
            track_event_id(&mut state.last_event_id, event_id);
            println!("\nResearch complete.");
            if !state.saw_text {
                match last_model_output_text(&interaction.steps) {
                    Some(text) => println!("{text}"),
                    None => println!("No text output returned."),
                }
            }
            state.is_complete = true;
        }
        InteractionSseEvent::InteractionStatusUpdate {
            status, event_id, ..
        } => {
            track_event_id(&mut state.last_event_id, event_id);
            println!("Status update: {status:?}");
        }
        InteractionSseEvent::Error { error, event_id } => {
            track_event_id(&mut state.last_event_id, event_id);
            eprintln!("Stream error: {} ({})", error.message, error.code);
            state.is_complete = true;
        }
        InteractionSseEvent::StepStop { event_id, .. } => {
            track_event_id(&mut state.last_event_id, event_id);
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    let use_streaming = std::env::args().any(|arg| arg == "--stream");
    let agent = deep_research_agent();
    let client = gemini::Client::from_env()?.interactions_api();

    // Deep Research is selected by `request.agent`; the request intentionally
    // omits `model`, matching the official Gemini Deep Research examples.
    let request = deep_research_request(agent.clone(), DEFAULT_PROMPT, use_streaming);

    if use_streaming {
        println!("== Deep Research (streaming) ==");
        println!("Agent: {agent}");
        let mut state = StreamState::default();
        let mut attempt = 0usize;

        loop {
            let stream = if attempt == 0 {
                client.stream_interaction_events(request.clone()).await
            } else if let Some(interaction_id) = state.interaction_id.as_deref() {
                client
                    .stream_interaction_events_by_id(interaction_id, state.last_event_id.as_deref())
                    .await
            } else {
                eprintln!("Stream closed before interaction_id was received.");
                break;
            };

            let mut stream = match stream {
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

            let Some(interaction_id) = state.interaction_id.as_deref() else {
                break;
            };

            // Official Deep Research guidance recommends checking the background
            // interaction status before reconnecting a dropped/expired stream.
            let interaction = client.get_interaction(interaction_id).await?;
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
            let interaction = poll_until_terminal(&client, interaction_id).await?;
            print_interaction_result(&interaction);
        }

        if let Some(interaction_id) = state.interaction_id {
            println!("Interaction ID: {interaction_id}");
        }
        if let Some(last_event_id) = state.last_event_id {
            println!("Last event ID: {last_event_id}");
        }

        return Ok(());
    }

    println!("== Deep Research (background polling) ==");
    println!("Agent: {agent}");
    let interaction = client.create_interaction(request).await?;
    if interaction.id.is_empty() {
        println!("No interaction id returned; aborting.");
        return Ok(());
    }
    println!("Research started: {}", interaction.id);

    let interaction = poll_until_terminal(&client, &interaction.id).await?;
    print_interaction_result(&interaction);

    Ok(())
}
