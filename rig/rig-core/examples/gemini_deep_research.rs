use anyhow::Result;
use futures::StreamExt;
use rig::completion::CompletionModel;
use rig::prelude::*;
use rig::providers::gemini::{
    self,
    interactions_api::{
        AdditionalParameters, AgentConfig, Content, ContentDelta, InteractionSseEvent,
        InteractionStatus, TextDelta, ThinkingSummaries, ThoughtSummaryContent,
        ThoughtSummaryDelta,
    },
};
use std::time::Duration;
use tokio::time::sleep;
use tracing_subscriber::EnvFilter;

const DEEP_RESEARCH_AGENT: &str = "deep-research-pro-preview-12-2025";
const STREAM_RETRY_DELAY_SECS: u64 = 2;

fn deep_research_params() -> AdditionalParameters {
    AdditionalParameters {
        agent: Some(DEEP_RESEARCH_AGENT.to_string()),
        background: Some(true),
        store: Some(true),
        agent_config: Some(AgentConfig::DeepResearch {
            thinking_summaries: Some(ThinkingSummaries::Auto),
        }),
        ..Default::default()
    }
}

fn extract_text(outputs: &[Content]) -> String {
    outputs
        .iter()
        .filter_map(|content| match content {
            Content::Text(text) => Some(text.text.clone()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
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
        InteractionSseEvent::InteractionStart {
            interaction,
            event_id,
        } => {
            track_event_id(&mut state.last_event_id, event_id);
            state.interaction_id = Some(interaction.id.clone());
            println!("Interaction started: {}", interaction.id);
        }
        InteractionSseEvent::ContentStart {
            content, event_id, ..
        } => {
            track_event_id(&mut state.last_event_id, event_id);
            if let Content::Text(text) = content {
                print!("{}", text.text);
                state.saw_text = true;
            }
        }
        InteractionSseEvent::ContentDelta {
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
        InteractionSseEvent::InteractionComplete {
            interaction,
            event_id,
        } => {
            track_event_id(&mut state.last_event_id, event_id);
            println!("\nResearch complete.");
            if !state.saw_text {
                let text = extract_text(&interaction.outputs);
                if text.is_empty() {
                    println!("No text output returned.");
                } else {
                    println!("{text}");
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
        InteractionSseEvent::ContentStop { event_id, .. } => {
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
    let client = gemini::Client::from_env().interactions_api();
    let model = client.completion_model("gemini-3-pro-preview");
    let prompt = "Research the history of Google TPUs.";

    let params = deep_research_params();
    let request = model
        .completion_request(prompt)
        .additional_params(serde_json::to_value(&params)?)
        .build();

    if use_streaming {
        println!("== Deep Research (streaming) ==");
        let mut state = StreamState::default();
        let mut attempt = 0usize;

        loop {
            let stream = if attempt == 0 {
                model.stream_interaction_events(request.clone()).await
            } else if let Some(interaction_id) = state.interaction_id.as_deref() {
                model
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

            if state.interaction_id.is_none() {
                break;
            }

            attempt += 1;
            println!("\nStream interrupted. Reconnecting in {STREAM_RETRY_DELAY_SECS}s...");
            sleep(Duration::from_secs(STREAM_RETRY_DELAY_SECS)).await;
        }

        if !state.is_complete
            && let Some(interaction_id) = state.interaction_id.as_deref()
        {
            println!("Switching to polling for interaction {interaction_id}...");
            loop {
                let interaction = model.get_interaction(interaction_id).await?;
                if interaction.is_terminal() {
                    match interaction.status {
                        Some(InteractionStatus::Completed) => {
                            let text = extract_text(&interaction.outputs);
                            if text.is_empty() {
                                println!("No text output returned.");
                            } else {
                                println!("{text}");
                            }
                        }
                        Some(status) => {
                            println!("Research ended with status: {status:?}");
                        }
                        None => {
                            println!("Research ended without a status.");
                        }
                    }
                    break;
                }
                println!(
                    "Status: {:?}. Polling again in 10s...",
                    interaction.status.unwrap_or(InteractionStatus::InProgress)
                );
                sleep(Duration::from_secs(10)).await;
            }
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
    let interaction = model.create_interaction(request).await?;
    if interaction.id.is_empty() {
        println!("No interaction id returned; aborting.");
        return Ok(());
    }
    println!("Research started: {}", interaction.id);

    loop {
        let interaction = model.get_interaction(&interaction.id).await?;
        if interaction.is_terminal() {
            match interaction.status {
                Some(InteractionStatus::Completed) => {
                    let text = extract_text(&interaction.outputs);
                    if text.is_empty() {
                        println!("No text output returned.");
                    } else {
                        println!("{text}");
                    }
                }
                Some(status) => {
                    println!("Research ended with status: {status:?}");
                }
                None => {
                    println!("Research ended without a status.");
                }
            }
            break;
        }

        println!(
            "Status: {:?}. Polling again in 10s...",
            interaction.status.unwrap_or(InteractionStatus::InProgress)
        );
        sleep(Duration::from_secs(10)).await;
    }

    Ok(())
}
