//! Cassette-backed Doubleword request-hook regression coverage.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use rig::agent::{CompletionCallAction, ObservationAction};
use rig::completion::Message;

use rig::hooks::{HookDecision, HookEntry, HookEvent};
use rig::message::UserContent;
use rig::prelude::*;

use super::super::{DEFAULT_MODEL, support::with_doubleword_cassette};
use crate::support::assert_nonempty_response;

#[derive(Clone, Default)]
struct ObservingHook {
    prompt_calls: Arc<AtomicUsize>,
    response_calls: Arc<AtomicUsize>,
    seen_prompt: Arc<Mutex<Option<String>>>,
}

impl ObservingHook {
    /// The hook record: records the outbound prompt text and counts responses.
    fn entry(&self) -> HookEntry {
        let hook = self.clone();
        HookEntry::new("observing", move |event| {
            let decision = hook.decide(event);
            Box::pin(async move { decision })
        })
    }

    fn decide(&self, event: HookEvent) -> HookDecision {
        match event {
            HookEvent::BeforeModelCall { prompt, .. } => {
                let Message::User { content } = prompt else {
                    return HookDecision::CompletionCall(CompletionCallAction::stop(
                        "expected a user message",
                    ));
                };
                let prompt = content
                    .iter()
                    .filter_map(|item| match item {
                        UserContent::Text(text) => Some(text.text.clone()),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join("\n");
                self.prompt_calls.fetch_add(1, Ordering::SeqCst);
                *self.seen_prompt.lock().expect("prompt hook lock") = Some(prompt);
                HookDecision::CompletionCall(CompletionCallAction::continue_run())
            }
            HookEvent::CompletionResponse { .. } => {
                self.response_calls.fetch_add(1, Ordering::SeqCst);
                HookDecision::Observation(ObservationAction::continue_run())
            }
            _ => HookDecision::Continue,
        }
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
                .runner("Entertain me with one short joke.")
                .add_hook(hook.entry())
                .run()
                .await
                .expect("hooked prompt should succeed")
                .output;
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
