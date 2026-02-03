use rig::agent::{HookAction, PromptHook, ToolCallHookAction};
use rig::client::{CompletionClient, ProviderClient};
use rig::completion::{CompletionModel, CompletionResponse, Message, Prompt};
use rig::message::{AssistantContent, UserContent};
use rig::providers::{self, openai};

#[derive(Clone)]
struct SessionIdHook<'a> {
    session_id: &'a str,
}

impl<'a, M: CompletionModel> PromptHook<M> for SessionIdHook<'a> {
    async fn on_tool_call(
        &self,
        tool_name: &str,
        tool_call_id: Option<String>,
        internal_call_id: &str,
        args: &str,
    ) -> ToolCallHookAction {
        println!(
            "[Session {}] Calling tool: {} with call ID: {tool_call_id} (internal: {internal_call_id}) with args: {}",
            self.session_id,
            tool_name,
            args,
            tool_call_id = tool_call_id.unwrap_or("<no call ID provided>".to_string()),
        );
        ToolCallHookAction::Continue
    }

    async fn on_tool_result(
        &self,
        tool_name: &str,
        _tool_call_id: Option<String>,
        _internal_call_id: &str,
        args: &str,
        result: &str,
    ) -> HookAction {
        println!(
            "[Session {}] Tool result for {} (args: {}): {}",
            self.session_id, tool_name, args, result
        );

        HookAction::cont()
    }

    async fn on_completion_call(&self, prompt: &Message, _history: &[Message]) -> HookAction {
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

        HookAction::cont()
    }

    async fn on_completion_response(
        &self,
        _prompt: &Message,
        response: &CompletionResponse<M::Response>,
    ) -> HookAction {
        if let Ok(resp) = serde_json::to_string(&response.raw_response) {
            println!("[Session {}] Received response: {}", self.session_id, resp);
        } else {
            println!(
                "[Session {}] Received response: <non-serializable>",
                self.session_id
            );
        }

        HookAction::cont()
    }
}

// Example main function (pseudo-code, as actual Agent/CompletionModel setup is project-specific)
#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    let client = providers::openai::Client::from_env();

    // Create agent with a single context prompt
    let comedian_agent = client
        .agent(openai::GPT_4O)
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
