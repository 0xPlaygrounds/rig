use std::env;

use rig::agent::{CancelSignal, PromptHook};
use rig::client::CompletionClient;
use rig::completion::{CompletionModel, CompletionResponse, Message, Prompt};
use rig::message::{AssistantContent, UserContent};
use rig::providers;

#[derive(Clone)]
struct SessionIdHook<'a> {
    session_id: &'a str,
}

impl<'a, M: CompletionModel> PromptHook<M> for SessionIdHook<'a> {
    async fn on_tool_call(&self, tool_name: &str, args: &str, _cancel_sig: CancelSignal) {
        println!(
            "[Session {}] Calling tool: {} with args: {}",
            self.session_id, tool_name, args
        );
    }
    async fn on_tool_result(
        &self,
        tool_name: &str,
        args: &str,
        result: &str,
        _cancel_sig: CancelSignal,
    ) {
        println!(
            "[Session {}] Tool result for {} (args: {}): {}",
            self.session_id, tool_name, args, result
        );
    }

    async fn on_completion_call(
        &self,
        prompt: &Message,
        _history: &[Message],
        _cancel_sig: CancelSignal,
    ) {
        println!(
            "[Session {}] Sending prompt: {}",
            self.session_id,
            match prompt {
                Message::User { content } => content
                    .iter()
                    .filter_map(|c| {
                        if let UserContent::Text(text_content) = c {
                            Some(text_content.text.clone())
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<_>>()
                    .join("\n"),
                Message::Assistant { content, .. } => content
                    .iter()
                    .filter_map(|c| if let AssistantContent::Text(text_content) = c {
                        Some(text_content.text.clone())
                    } else {
                        None
                    })
                    .collect::<Vec<_>>()
                    .join("\n"),
            }
        );
    }

    async fn on_completion_response(
        &self,
        _prompt: &Message,
        response: &CompletionResponse<M::Response>,
        _cancel_sig: CancelSignal,
    ) {
        if let Ok(resp) = serde_json::to_string(&response.raw_response) {
            println!("[Session {}] Received response: {}", self.session_id, resp);
        } else {
            println!(
                "[Session {}] Received response: <non-serializable>",
                self.session_id
            );
        }
    }
}

// Example main function (pseudo-code, as actual Agent/CompletionModel setup is project-specific)
#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    let client = providers::openai::Client::new(
        &env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY not set"),
    );

    // Create agent with a single context prompt
    let comedian_agent = client
        .agent("gpt-4o")
        .preamble("You are a comedian here to entertain the user using humour and jokes.")
        .build();

    let session_id = "abc123";
    let hook = SessionIdHook { session_id };

    // Prompt the agent and print the response
    comedian_agent
        .prompt("Entertain me!")
        .with_hook(hook)
        .await?;

    Ok(())
}
