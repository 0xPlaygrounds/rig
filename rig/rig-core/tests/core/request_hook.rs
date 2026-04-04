//! Migrated from `examples/request_hook.rs`.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use rig::agent::HookAction;
use rig::agent::PromptHook;
use rig::client::{CompletionClient, ProviderClient};
use rig::completion::{CompletionModel, CompletionResponse, Message, Prompt};
use rig::message::UserContent;
use rig::providers::{self, openai};

use crate::support::assert_nonempty_response;

#[derive(Clone)]
struct SessionIdHook<'a> {
    session_id: &'a str,
    prompt_calls: Arc<AtomicUsize>,
    response_calls: Arc<AtomicUsize>,
    seen_prompt: Arc<Mutex<Option<String>>>,
    seen_response: Arc<Mutex<Option<String>>>,
}

impl<'a, M: CompletionModel> PromptHook<M> for SessionIdHook<'a> {
    async fn on_completion_call(&self, prompt: &Message, _history: &[Message]) -> HookAction {
        let Message::User { content } = prompt else {
            return HookAction::terminate("expected a user message");
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
        let mut seen_prompt = self.seen_prompt.lock().expect("prompt mutex");
        *seen_prompt = Some(format!("{}:{prompt_text}", self.session_id));
        HookAction::cont()
    }

    async fn on_completion_response(
        &self,
        _prompt: &Message,
        response: &CompletionResponse<M::Response>,
    ) -> HookAction {
        self.response_calls.fetch_add(1, Ordering::SeqCst);
        let mut seen_response = self.seen_response.lock().expect("response mutex");
        *seen_response = Some(format!("{:?}", response.choice));
        HookAction::cont()
    }
}

#[tokio::test]
#[ignore = "requires OPENAI_API_KEY"]
async fn request_hook_records_prompt_and_response() {
    let client = providers::openai::Client::from_env();
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

    let response = agent
        .prompt("Entertain me!")
        .with_hook(hook.clone())
        .await
        .expect("prompt should succeed");

    assert_nonempty_response(&response);
    assert_eq!(hook.prompt_calls.load(Ordering::SeqCst), 1);
    assert_eq!(hook.response_calls.load(Ordering::SeqCst), 1);
    assert!(
        hook.seen_prompt
            .lock()
            .expect("prompt mutex")
            .as_deref()
            .is_some_and(|prompt| prompt.contains("Entertain me!"))
    );
    assert!(
        hook.seen_response
            .lock()
            .expect("response mutex")
            .as_deref()
            .is_some_and(|captured| !captured.is_empty())
    );
}
