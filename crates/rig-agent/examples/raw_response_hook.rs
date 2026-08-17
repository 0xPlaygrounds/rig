//! Read a provider-specific field off the provider's own response, from a hook.
//!
//! An agent erases its model behind `ModelHandle`, so the typed escape hatch
//! every provider offers — `raw_completion` / `raw_stream`, which hand back the
//! provider's own response type — is unreachable from an agent run. The
//! normalized `CompletionResponse` deliberately carries only what every
//! provider has in common, and some of what a provider says is only there:
//! OpenAI's `system_fingerprint` and `service_tier`, Anthropic's
//! `stop_sequence`, Ollama's timings, and so on.
//!
//! `AgentBuilder::capture_raw_response(true)` is the opt-in path. With it on,
//! every attempt's provider response — the value `raw_completion` /
//! `raw_stream` would have returned, serialized — travels as `raw` on the
//! `CompletionResponse` (blocking) and `StreamResponseFinish` (streamed) hook
//! events, on the medium-neutral `ModelTurnFinished` event, and on each
//! `CompletionCall` in the run's record. It is off by default because it costs
//! a serialization of the provider's parsed response per call, and it is local
//! policy: the provider never sees the flag.
//!
//! The hook below runs unchanged on both surfaces. It recovers OpenAI's own
//! response types from `raw` — the provider's types are `Deserialize`, so
//! typed access is one `serde_json::from_value` away — and prints the fields
//! Rig does not normalize.
//!
//! ```not_rust
//! OPENAI_API_KEY=... cargo run -p rig-agent --example raw_response_hook
//! ```

use anyhow::Result;
use futures::StreamExt;
use rig_agent::{
    agent::{CompletionResponseEvent, ObservationAction, StreamResponseFinish},
    prelude::*,
};
use rig_core::providers::openai;
use serde::Deserialize;

/// Prints the OpenAI-only fields of every completed call. Provider-specific by
/// design: that is the whole point of reaching for `raw`.
struct PrintOpenAiFields;

impl AgentHook for PrintOpenAiFields {
    /// Blocking surface: `raw` is the Chat Completions response as OpenAI's
    /// wire type parsed it, so it deserializes straight back into that type.
    async fn on_completion_response(
        &self,
        _ctx: &HookContext,
        event: CompletionResponseEvent<'_>,
    ) -> ObservationAction {
        match event.raw.map(openai::CompletionResponse::deserialize) {
            Some(Ok(response)) => println!(
                "  id {} · system_fingerprint {:?} · service_tier {:?}",
                response.id, response.system_fingerprint, response.service_tier
            ),
            Some(Err(err)) => println!("  raw is not an OpenAI response: {err}"),
            // `None` means capture was off, not that the provider sent nothing.
            None => println!("  no raw captured"),
        }
        ObservationAction::continue_run()
    }

    /// Streamed surface: `raw` is the stream's *terminal record* as OpenAI's
    /// wire type accumulated it — the top-level chunk fields Rig does not
    /// normalize land in its `additional_params`.
    async fn on_stream_response_finish(
        &self,
        _ctx: &HookContext,
        event: StreamResponseFinish<'_>,
    ) -> ObservationAction {
        match event
            .raw
            .map(openai::StreamingCompletionResponse::<openai::Usage>::deserialize)
        {
            Some(Ok(terminal)) => {
                let extra = |key: &str| {
                    terminal
                        .additional_params
                        .as_ref()
                        .and_then(|params| params.get(key))
                        .cloned()
                };
                println!(
                    "  id {:?} · system_fingerprint {:?} · service_tier {:?}",
                    terminal.response_id,
                    extra("system_fingerprint"),
                    extra("service_tier"),
                );
            }
            Some(Err(err)) => println!("  raw is not an OpenAI terminal: {err}"),
            None => println!("  no raw captured"),
        }
        ObservationAction::continue_run()
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // The Chat Completions route, whose response carries `system_fingerprint`.
    let client = openai::Client::from_env()?.completions_api();
    let agent = client
        .agent(openai::GPT_5_2)
        .preamble("Answer in one short sentence.")
        // The opt-in. Everything below would print "no raw captured" without it.
        .capture_raw_response(true)
        .add_hook(PrintOpenAiFields)
        .build();

    println!("blocking:");
    let response = agent
        .prompt("What does a system fingerprint identify?")
        .extended_details()
        .await?;
    println!("  => {}", response.output);
    // The same payload the hook saw is on the run's record, per call.
    for call in &response.completion_calls {
        println!(
            "  call {} recorded raw: {}",
            call.call_index,
            call.raw
                .as_ref()
                .map_or("none".to_owned(), |raw| raw.to_string())
        );
    }

    println!("\nstreaming:");
    let mut stream = agent
        .stream_prompt("What does a system fingerprint identify?")
        .await;
    while let Some(item) = stream.next().await {
        if let MultiTurnStreamItem::FinalResponse(final_response) = item? {
            println!("  => {}", final_response.output);
        }
    }

    Ok(())
}
