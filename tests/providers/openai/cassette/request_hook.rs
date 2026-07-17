//! Preserves the live request-hook example as provider-local regression coverage.

use anyhow::{Result, anyhow};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use rig::agent::{
    AgentHook, CompletionCallAction, CompletionCallEvent, ModelTurnPrepared, ObservationAction,
};
use rig::client::CompletionClient;
use rig::completion::{Message, Prompt};
use rig::message::UserContent;
use rig::providers::openai;

use super::super::support::with_openai_cassette_result;
use crate::support::assert_nonempty_response;

#[derive(Clone)]
struct SessionIdHook<'a> {
    session_id: &'a str,
    prompt_calls: Arc<AtomicUsize>,
    response_calls: Arc<AtomicUsize>,
    seen_prompt: Arc<Mutex<Option<String>>>,
    seen_response: Arc<Mutex<Option<String>>>,
}

impl AgentHook for SessionIdHook<'_> {
    async fn on_completion_call(
        &self,
        _ctx: &rig::agent::HookContext,
        event: CompletionCallEvent<'_>,
    ) -> CompletionCallAction {
        let Message::User { content } = event.prompt else {
            return CompletionCallAction::stop("expected a user message");
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
                CompletionCallAction::continue_run()
            }
            Err(_) => CompletionCallAction::stop("prompt hook state unavailable"),
        }
    }

    async fn on_model_turn_prepared(
        &self,
        _ctx: &rig::agent::HookContext,
        event: ModelTurnPrepared<'_>,
    ) -> ObservationAction {
        self.response_calls.fetch_add(1, Ordering::SeqCst);
        match self.seen_response.lock() {
            Ok(mut seen_response) => {
                *seen_response = Some(format!("{:?}", event.content));
                ObservationAction::continue_run()
            }
            Err(_) => ObservationAction::stop("prepared-turn hook state unavailable"),
        }
    }
}

#[tokio::test]
async fn request_hook_records_prompt_and_response() -> Result<()> {
    with_openai_cassette_result(
        "request_hook/request_hook_records_prompt_and_response",
        |client| async move {
            let agent = client
                .agent(openai::GPT_4O)
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
            anyhow::ensure!(hook.prompt_calls.load(Ordering::SeqCst) == 1);
            anyhow::ensure!(hook.response_calls.load(Ordering::SeqCst) == 1);

            let seen_prompt = hook
                .seen_prompt
                .lock()
                .map_err(|_| anyhow!("prompt hook state unavailable"))?
                .clone();
            let seen_response = hook
                .seen_response
                .lock()
                .map_err(|_| anyhow!("prepared-turn hook state unavailable"))?
                .clone();

            anyhow::ensure!(
                seen_prompt
                    .as_deref()
                    .is_some_and(|prompt| prompt.contains("Entertain me!"))
            );
            anyhow::ensure!(
                seen_response
                    .as_deref()
                    .is_some_and(|captured| !captured.is_empty())
            );

            Ok(())
        },
    )
    .await
}
