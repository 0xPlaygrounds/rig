//! Mistral request-hook regression coverage.

use anyhow::{Result, anyhow};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use rig::agent::{CompletionCallAction, ObservationAction};
use rig::completion::Message;

use rig::hooks::{HookDecision, HookEntry, HookEvent};
use rig::message::UserContent;
use rig::prelude::*;

use crate::support::assert_nonempty_response;

use super::DEFAULT_MODEL;

#[derive(Clone)]
struct SessionIdHook {
    session_id: String,
    prompt_calls: Arc<AtomicUsize>,
    response_calls: Arc<AtomicUsize>,
    seen_prompt: Arc<Mutex<Option<String>>>,
    seen_response: Arc<Mutex<Option<String>>>,
}

impl SessionIdHook {
    /// The hook record: records the outbound prompt and the normalized response.
    fn entry(&self) -> HookEntry {
        let hook = self.clone();
        HookEntry::sync("session-id", move |event| hook.decide(event))
    }

    fn decide(&self, event: HookEvent) -> HookDecision {
        match event {
            HookEvent::BeforeModelCall { prompt, .. } => {
                let Message::User { content } = prompt.as_ref() else {
                    return HookDecision::CompletionCall(CompletionCallAction::stop(
                        "expected a user message",
                    ));
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
                        HookDecision::CompletionCall(CompletionCallAction::continue_run())
                    }
                    Err(_) => HookDecision::CompletionCall(CompletionCallAction::stop(
                        "prompt hook state unavailable",
                    )),
                }
            }
            HookEvent::CompletionResponse { response, .. } => {
                self.response_calls.fetch_add(1, Ordering::SeqCst);
                match self.seen_response.lock() {
                    Ok(mut seen_response) => {
                        *seen_response = Some(format!("{:?}", response.choice));
                        HookDecision::Observation(ObservationAction::continue_run())
                    }
                    Err(_) => HookDecision::Observation(ObservationAction::stop(
                        "response hook state unavailable",
                    )),
                }
            }
            _ => HookDecision::Continue,
        }
    }
}

#[tokio::test]
#[ignore = "requires MISTRAL_API_KEY"]
async fn request_hook_records_prompt_and_response() -> Result<()> {
    let agent = rig::providers::mistral::Client::from_env()
        .expect("client should build")
        .agent(DEFAULT_MODEL)
        .preamble("You are a comedian here to entertain the user using humour and jokes.")
        .build();

    let hook = SessionIdHook {
        session_id: "abc123".to_owned(),
        prompt_calls: Arc::new(AtomicUsize::new(0)),
        response_calls: Arc::new(AtomicUsize::new(0)),
        seen_prompt: Arc::new(Mutex::new(None)),
        seen_response: Arc::new(Mutex::new(None)),
    };

    let response = agent
        .runner("Entertain me!")
        .add_hook(hook.entry())
        .run()
        .await?
        .output;

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
        .map_err(|_| anyhow!("response hook state unavailable"))?
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
}
