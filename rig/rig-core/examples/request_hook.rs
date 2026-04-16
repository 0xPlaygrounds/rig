//! Demonstrates observing prompt/response lifecycle events with `PromptHook`.
//! Requires `OPENAI_API_KEY`.
//! Run it to see the hook log the outgoing prompt and the incoming model response.

use anyhow::Result;
use rig::agent::{HookAction, PromptHook};
use rig::client::{CompletionClient, ProviderClient};
use rig::completion::{CompletionModel, CompletionResponse, Message, Prompt};
use rig::message::UserContent;
use rig::providers::openai;

#[derive(Clone)]
struct LoggingHook<'a> {
    session_id: &'a str,
}

impl<'a, M> PromptHook<M> for LoggingHook<'a>
where
    M: CompletionModel,
{
    async fn on_completion_call(&self, prompt: &Message, _history: &[Message]) -> HookAction {
        if let Message::User { content } = prompt {
            let prompt_text = content
                .iter()
                .filter_map(|content| match content {
                    UserContent::Text(text) => Some(text.text.clone()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("\n");
            if !prompt_text.is_empty() {
                println!("[{}] sending prompt: {}", self.session_id, prompt_text);
            }
        }

        HookAction::cont()
    }

    async fn on_completion_response(
        &self,
        _prompt: &Message,
        response: &CompletionResponse<M::Response>,
    ) -> HookAction {
        println!(
            "[{}] received response: {:?}",
            self.session_id, response.choice
        );
        HookAction::cont()
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let agent = openai::Client::from_env()
        .agent(openai::GPT_4O)
        .preamble("You are a comedian here to entertain the user using humour and jokes.")
        .build();

    let response = agent
        .prompt("Entertain me!")
        .with_hook(LoggingHook {
            session_id: "demo-session",
        })
        .await?;

    println!("\nFinal response:\n{response}");

    Ok(())
}
