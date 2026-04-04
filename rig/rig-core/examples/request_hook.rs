use anyhow::{Result, anyhow};
use rig::agent::{HookAction, PromptHook};
use rig::client::{CompletionClient, ProviderClient};
use rig::completion::{CompletionModel, CompletionResponse, Message, Prompt};
use rig::message::UserContent;
use rig::providers::openai;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

#[derive(Clone)]
struct SessionIdHook<'a> {
    session_id: &'a str,
    prompt_calls: Arc<AtomicUsize>,
    response_calls: Arc<AtomicUsize>,
    seen_prompt: Arc<Mutex<Option<String>>>,
    seen_response: Arc<Mutex<Option<String>>>,
}

impl<'a, M> PromptHook<M> for SessionIdHook<'a>
where
    M: CompletionModel,
{
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
        match self.seen_prompt.lock() {
            Ok(mut seen_prompt) => {
                *seen_prompt = Some(format!("{}:{prompt_text}", self.session_id));
                HookAction::cont()
            }
            Err(_) => HookAction::terminate("prompt hook state unavailable"),
        }
    }

    async fn on_completion_response(
        &self,
        _prompt: &Message,
        response: &CompletionResponse<M::Response>,
    ) -> HookAction {
        self.response_calls.fetch_add(1, Ordering::SeqCst);
        match self.seen_response.lock() {
            Ok(mut seen_response) => {
                *seen_response = Some(format!("{:?}", response.choice));
                HookAction::cont()
            }
            Err(_) => HookAction::terminate("response hook state unavailable"),
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let agent = openai::Client::from_env()
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
        .await?;

    let seen_prompt = hook
        .seen_prompt
        .lock()
        .map_err(|_| anyhow!("prompt hook state unavailable"))?
        .clone()
        .unwrap_or_default();
    let seen_response = hook
        .seen_response
        .lock()
        .map_err(|_| anyhow!("response hook state unavailable"))?
        .clone()
        .unwrap_or_default();

    println!("response:\n{response}\n");
    println!(
        "prompt hook calls: {}",
        hook.prompt_calls.load(Ordering::SeqCst)
    );
    println!(
        "response hook calls: {}",
        hook.response_calls.load(Ordering::SeqCst)
    );
    println!("captured prompt: {seen_prompt}");
    println!("captured response: {seen_response}");

    Ok(())
}
