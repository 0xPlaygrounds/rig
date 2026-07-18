//! Cassette-backed Doubleword request-hook regression coverage.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use rig::agent::{
    AgentHook, CompletionCallAction, CompletionCallEvent, CompletionResponseEvent,
    ObservationAction,
};
use rig::completion::{Message, Prompt};
use rig::message::UserContent;
use rig::prelude::AgentClientExt;

use super::super::{DEFAULT_MODEL, support::with_doubleword_cassette};
use crate::support::assert_nonempty_response;

#[derive(Clone, Default)]
struct ObservingHook {
    prompt_calls: Arc<AtomicUsize>,
    response_calls: Arc<AtomicUsize>,
    seen_prompt: Arc<Mutex<Option<String>>>,
}

impl AgentHook for ObservingHook {
    async fn on_completion_call(
        &self,
        _ctx: &rig::agent::HookContext,
        event: CompletionCallEvent<'_>,
    ) -> CompletionCallAction {
        let Message::User { content } = event.prompt else {
            return CompletionCallAction::stop("expected a user message");
        };
        let prompt = content
            .iter()
            .filter_map(|item| match item {
                UserContent::Text(text) => Some(text.text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n");
        self.prompt_calls.fetch_add(1, Ordering::SeqCst);
        *self.seen_prompt.lock().expect("prompt hook lock") = Some(prompt);
        CompletionCallAction::continue_run()
    }

    async fn on_completion_response(
        &self,
        _ctx: &rig::agent::HookContext,
        _event: CompletionResponseEvent<'_>,
    ) -> ObservationAction {
        self.response_calls.fetch_add(1, Ordering::SeqCst);
        ObservationAction::continue_run()
    }
}

#[tokio::test]
async fn request_hook_records_prompt_and_response() {
    with_doubleword_cassette(
        "request_hook/request_hook_records_prompt_and_response",
        |client| async move {
            let hook = ObservingHook::default();
            let response = client
                .agent(DEFAULT_MODEL)
                .build()
                .prompt("Entertain me with one short joke.")
                .add_hook(hook.clone())
                .await
                .expect("hooked prompt should succeed");
            assert_nonempty_response(&response);
            assert_eq!(hook.prompt_calls.load(Ordering::SeqCst), 1);
            assert_eq!(hook.response_calls.load(Ordering::SeqCst), 1);
            assert!(
                hook.seen_prompt
                    .lock()
                    .expect("prompt hook lock")
                    .as_deref()
                    .is_some_and(|prompt| prompt.contains("Entertain me"))
            );
        },
    )
    .await;
}
