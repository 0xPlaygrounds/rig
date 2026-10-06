//! Read a provider-specific field off the provider's own response, from a hook.
//!
//! An agent erases its model behind `ModelHandle`, and the normalized
//! `CompletionResponse` deliberately carries only what every provider has in
//! common — so some of what a provider says would have nowhere to land:
//! OpenAI's `system_fingerprint` and `service_tier`, Anthropic's
//! `stop_sequence`, Ollama's timings, and so on.
//!
//! It lands anyway: the provider's own response for every attempt, serialized,
//! arrives as `CompletionResponse::raw` — on the response an `on_outcome` hook
//! sees for a completion effect (fired on both the blocking and the streamed
//! surface), on the medium-neutral `ModelTurnFinished` event, and on each
//! `CompletionCall` in the run's record. Nothing to switch on: it is the
//! same parity the pre-normalization `raw_response` had.
//!
//! The hook below runs unchanged on both surfaces. It reads `raw` as JSON and
//! prints the fields Rig does not normalize. `raw` is the provider's reply
//! document on both: the body itself on the blocking surface, and the same
//! document rebuilt from the stream's chunks on the streamed one.
//!
//! ```not_rust
//! OPENAI_API_KEY=... cargo run -p rig-agent --example raw_response_hook
//! ```
//!

use anyhow::Result;
use futures::StreamExt;
use rig_agent::{
    agent::{OutcomeAction, OutcomeEvent},
    prelude::*,
};
use rig_core::providers::openai;
use rig_core::providers::openai::{OpenAIConfig, Route};

/// Prints the OpenAI-only fields of every completed call. Provider-specific by
/// design: that is the whole point of reaching for `raw`.
struct PrintOpenAiFields;

impl AgentHook for PrintOpenAiFields {
    /// `raw` is the Chat Completions reply document on both surfaces.
    async fn on_outcome(&self, ctx: &HookContext, event: OutcomeEvent<'_>) -> OutcomeAction {
        let Some(response) = event.completion() else {
            return OutcomeAction::proceed();
        };
        let raw = &response.raw;
        println!(
            "  streamed {} · id {:?} · system_fingerprint {:?} · service_tier {:?}",
            ctx.is_streaming(),
            raw.get("id"),
            raw.get("system_fingerprint"),
            raw.get("service_tier"),
        );
        OutcomeAction::proceed()
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // The Chat Completions route, whose response carries `system_fingerprint`;
    // OpenAI's default route is the Responses API, so the configuration is
    // routed once and the agent follows.
    let model = OpenAIConfig::from_env()?
        .with_route(Route::Chat)
        .client()
        .completion(openai::GPT_5_2);
    let agent = AgentBuilder::new(model)
        .preamble("Answer in one short sentence.")
        .add_hook(PrintOpenAiFields)
        .build();

    println!("blocking:");
    let response = agent
        .prompt("What does a system fingerprint identify?")
        .await?;
    println!("  => {}", response.output());
    // The same payload the hook saw is on the run's record, per call.
    for call in &response.completion_calls {
        println!("  call {} recorded raw: {}", call.call_index, call.raw);
    }

    println!("\nstreaming:");
    let mut stream = agent
        .prompt("What does a system fingerprint identify?")
        .stream();
    while let Some(item) = stream.next().await {
        if let MultiTurnStreamItem::FinalResponse(final_response) = item? {
            println!("  => {}", final_response.output());
        }
    }

    Ok(())
}
