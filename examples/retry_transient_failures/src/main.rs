//! Retries a model call that failed with a transient error, in user code.
//!
//! A rate limit (429), a server error (5xx) or a dropped connection fails the
//! run with a retryable error. This example waits as long as the provider
//! asked ([`ErrorReport::retry_after`]), else a doubling wait, and sends the
//! prompt again, a few times at most. An [`AgentHook`] cannot do this:
//! [`ModelTurnAction::Retry`] only re-asks after a model call that succeeded,
//! and no hook can re-send a call that failed. Sending the prompt again also
//! repeats any tool calls the failed run made.
//!
//! [`ErrorReport::retry_after`]: rig::error::ErrorReport::retry_after
//! [`AgentHook`]: rig::agent::AgentHook
//! [`ModelTurnAction::Retry`]: rig::agent::ModelTurnAction::Retry
//!
//! Requires `OPENAI_API_KEY`. `OPENAI_BASE_URL` points it at another
//! OpenAI-compatible endpoint, such as a local stub. Run with
//! `cargo run -p retry_transient_failures`.

use std::time::{Duration, SystemTime};

use anyhow::Result;
use rig::completion::PromptError;
use rig::prelude::*;
use rig::providers::openai::{self, OpenAI};

const MAX_ATTEMPTS: u32 = 4;
const FIRST_WAIT: Duration = Duration::from_secs(1);
/// The longest wait accepted; a provider asking for more fails the run.
const MAX_WAIT: Duration = Duration::from_secs(60);

/// How long to wait before attempt `attempt + 1`, or `None` when the error
/// is not worth retrying.
fn retry_wait(error: &PromptError, attempt: u32) -> Option<Duration> {
    let report = match error {
        PromptError::Provider(error) => error.report(),
        PromptError::Report(report) => report.clone(),
        _ => return None,
    };
    if !report.is_retryable() {
        return None;
    }
    let wait = report.retry_after(SystemTime::now()).unwrap_or_else(|| {
        FIRST_WAIT.saturating_mul(2u32.saturating_pow(attempt.saturating_sub(1)))
    });
    (wait <= MAX_WAIT).then_some(wait)
}

#[tokio::main]
async fn main() -> Result<()> {
    let agent = AgentBuilder::new(OpenAI::from_env()?.completion(openai::GPT_4O_MINI))
        .preamble("Answer in one short sentence.")
        .build();

    let mut attempt = 1;
    let response = loop {
        match agent.prompt("What is a rate limit?").await {
            Ok(response) => break response,
            Err(error) => match retry_wait(&error, attempt) {
                Some(wait) if attempt < MAX_ATTEMPTS => {
                    println!("attempt {attempt} failed: {error}; retrying in {wait:?}");
                    tokio::time::sleep(wait).await;
                    attempt += 1;
                }
                _ => return Err(error.into()),
            },
        }
    };

    println!("{}", response.output());
    Ok(())
}
