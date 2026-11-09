//! Reads provider-specific reply fields from a hook.
//!
//! `extras::<OpenAiExt>()` reads typed extras such as OpenAI's
//! `system_fingerprint` and `service_tier` from the provider's reply, and returns
//! `None` for another provider's. The whole reply is also on
//! `CompletionResponse::raw`, which an `on_outcome` hook sees for a completion on
//! both the blocking and the streamed surface, rebuilt from the chunks when
//! streaming. The hook below runs unchanged on both.
//!
//! ```not_rust
//! OPENAI_API_KEY=... cargo run -p rig-agent --example raw_response_hook
//! ```

use anyhow::Result;
use futures::StreamExt;
use rig_agent::{
    agent::{OutcomeAction, OutcomeEvent},
    prelude::*,
};
use rig_core::providers::openai;
use rig_core::providers::openai::extension::OpenAiExt;
use rig_core::providers::openai::{OpenAIConfig, Route};

/// Prints the OpenAI-only fields of every completed call.
struct PrintOpenAiFields;

impl AgentHook for PrintOpenAiFields {
    /// The extras read the Chat Completions reply document on both surfaces.
    async fn on_outcome(&self, ctx: &HookContext, event: OutcomeEvent<'_>) -> OutcomeAction {
        let Some(response) = event.completion() else {
            return OutcomeAction::proceed();
        };
        let (system_fingerprint, service_tier) = match response.extras::<OpenAiExt>() {
            Some(Ok(extras)) => (extras.system_fingerprint, extras.service_tier),
            Some(Err(error)) => {
                println!("  the OpenAI reply did not decode: {error}");
                (None, None)
            }
            None => (None, None),
        };
        println!(
            "  streamed {} · id {:?} · system_fingerprint {:?} · service_tier {:?}",
            ctx.is_streaming(),
            response.response_id(),
            system_fingerprint,
            service_tier,
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
