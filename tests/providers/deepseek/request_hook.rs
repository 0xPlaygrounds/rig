//! DeepSeek request-hook regression coverage.

use anyhow::{Result, anyhow};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use rig::agent::{AgentHook, Flow, StepEvent};
use rig::client::CompletionClient;
use rig::completion::{CompletionModel, Message, Prompt};
use rig::message::UserContent;
use rig::providers::deepseek;

use super::support::with_deepseek_cassette_result;
use crate::support::assert_nonempty_response;

#[derive(Clone)]
struct SessionIdHook<'a> {
    session_id: &'a str,
    prompt_calls: Arc<AtomicUsize>,
    response_calls: Arc<AtomicUsize>,
    seen_prompt: Arc<Mutex<Option<String>>>,
    seen_response: Arc<Mutex<Option<String>>>,
}

impl<'a, M> AgentHook<M> for SessionIdHook<'a>
where
    M: CompletionModel,
{
    async fn on_event(&self, _ctx: &rig::agent::HookContext, event: StepEvent<'_, M>) -> Flow {
        match event {
            StepEvent::CompletionCall { prompt, .. } => {
                let Message::User { content } = prompt else {
                    return Flow::terminate("expected a user message");
                };

                let prompt_text = content
                    .iter()
                    .filter_map(|content| match content {
                        UserContent::Text(text) => Some(text.text.clone()),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join("\n");

                self.prompt_calls.fetch_add(1, Ordering::SeqCst);
                match self.seen_prompt.lock() {
                    Ok(mut seen_prompt) => {
                        *seen_prompt = Some(format!("{}:{prompt_text}", self.session_id));
                        Flow::cont()
                    }
                    Err(_) => Flow::terminate("prompt hook state unavailable"),
                }
            }
            StepEvent::CompletionResponse { response, .. } => {
                self.response_calls.fetch_add(1, Ordering::SeqCst);
                match self.seen_response.lock() {
                    Ok(mut seen_response) => {
                        *seen_response = Some(format!("{:?}", response.choice));
                        Flow::cont()
                    }
                    Err(_) => Flow::terminate("response hook state unavailable"),
                }
            }
            _ => Flow::cont(),
        }
    }
}

#[tokio::test]
async fn request_hook_records_prompt_and_response() -> Result<()> {
    with_deepseek_cassette_result(
        "request_hook/request_hook_records_prompt_and_response",
        |client| async move {
            let agent = client
                .agent(deepseek::DEEPSEEK_V4_FLASH)
                .preamble("You are a comedian here to entertain the user using humour and jokes.")
                .build();

            let hook = SessionIdHook {
                session_id: "abc123",
                prompt_calls: Arc::new(AtomicUsize::new(0)),
                response_calls: Arc::new(AtomicUsize::new(0)),
                seen_prompt: Arc::new(Mutex::new(None)),
                seen_response: Arc::new(Mutex::new(None)),
            };

            let response = agent.prompt("Entertain me!").add_hook(hook.clone()).await?;

            assert_nonempty_response(&response);
            anyhow::ensure!(
                hook.prompt_calls.load(Ordering::SeqCst) == 1,
                "expected one prompt hook call"
            );
            anyhow::ensure!(
                hook.response_calls.load(Ordering::SeqCst) == 1,
                "expected one response hook call"
            );

            let seen_prompt = hook
                .seen_prompt
                .lock()
                .map_err(|_| anyhow!("prompt hook state unavailable"))?
                .clone();
            let seen_response = hook
                .seen_response
                .lock()
                .map_err(|_| anyhow!("response hook state unavailable"))?
                .clone();

            anyhow::ensure!(
                seen_prompt
                    .as_deref()
                    .is_some_and(|prompt| prompt.contains("Entertain me!")),
                "expected hook to capture prompt text"
            );
            anyhow::ensure!(
                seen_response
                    .as_deref()
                    .is_some_and(|captured| !captured.is_empty()),
                "expected hook to capture response text"
            );

            Ok(())
        },
    )
    .await
}
