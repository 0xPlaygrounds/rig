//! Structured extraction over the session runtime.
//!
//! [`extract`] is the data-oriented successor of `Extractor<M, T>`: a
//! generic *function* over serde/schemars capability bounds (nothing stores
//! `T`), driving an [`AgentSession`](crate::session::AgentSession) in Tool
//! output mode and deserializing the structured result.

use std::sync::Arc;

use rig_core::schemars;

use crate::agent::AgentConfig;
use crate::agent::run::OutputMode;
use crate::completion::{Message, PromptError, Usage};

use crate::provider::{ProviderConfig, Runtime};
use crate::session::AgentSession;

/// Structured extraction failure.
#[derive(Debug, thiserror::Error)]
pub enum ExtractError {
    /// The underlying run failed.
    #[error(transparent)]
    Prompt(#[from] PromptError),
    /// The model's final output did not deserialize into the target type,
    /// after all retries. Carries the last raw output.
    #[error("structured output did not match the target type: {source}; raw output: {raw}")]
    Deserialization {
        /// The serde failure for the last attempt.
        source: serde_json::Error,
        /// The raw output of the last attempt.
        raw: String,
    },
}

/// Extract a `T` from the model's answer to `prompt`.
///
/// Sets the output schema from `T`'s [`schemars::JsonSchema`], forces Tool
/// output mode (the run finalizes via the synthetic output tool), and
/// deserializes the structured result. On deserialization failure the
/// extraction is retried up to `retries` times, each attempt carrying the
/// previous raw output plus corrective feedback in its history.
pub async fn extract<T>(
    config: AgentConfig,
    provider: ProviderConfig,
    rt: Arc<Runtime>,
    prompt: impl Into<Message>,
    retries: usize,
) -> Result<T, ExtractError>
where
    T: schemars::JsonSchema + serde::de::DeserializeOwned,
{
    extract_with_usage(config, provider, rt, prompt, retries)
        .await
        .map(|outcome| outcome.value)
}

/// A successful extraction plus the token usage it cost.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct ExtractionOutcome<T> {
    /// The deserialized structured value.
    pub value: T,
    /// Token usage accumulated across **all** attempts, including billed
    /// attempts whose output failed deserialization.
    pub usage: Usage,
}

/// [`extract`], additionally reporting the token usage accumulated across
/// every attempt (mirroring the classic `Extractor::extract_with_usage`).
///
/// Usage accumulates across all retry attempts, including attempts that
/// received a billed response but failed deserialization. Attempts whose run
/// itself returned an error contribute no usage, and when every attempt fails
/// the returned error carries no usage information at all (matching the
/// classic retry semantics: accumulated usage is only observable on success).
pub async fn extract_with_usage<T>(
    mut config: AgentConfig,
    provider: ProviderConfig,
    rt: Arc<Runtime>,
    prompt: impl Into<Message>,
    retries: usize,
) -> Result<ExtractionOutcome<T>, ExtractError>
where
    T: schemars::JsonSchema + serde::de::DeserializeOwned,
{
    config.output_schema = Some(schemars::schema_for!(T));
    config.output_mode = OutputMode::Tool;
    if config.max_turns.is_none() {
        // Tool output mode needs at least a call budget of two: one for the
        // answer and one for the output-tool retry the run may issue.
        config.max_turns = Some(2);
    }
    extract_attempt_loop(config, provider, rt, prompt.into(), retries, |raw| {
        serde_json::from_str::<T>(raw.trim())
    })
    .await
}

/// [`extract`] in **Native** output mode, mirroring the classic
/// `TypedPromptRequest`: the schema is passed as the provider's native
/// structured-output constraint (no synthetic output tool), and the final text
/// is parsed with a balanced-JSON fallback so prose or markdown fences around
/// the JSON don't break the parse.
///
/// Like [`extract_with_usage`], the outcome reports token usage accumulated
/// across every attempt, including billed attempts that failed to parse.
pub async fn extract_native<T>(
    mut config: AgentConfig,
    provider: ProviderConfig,
    rt: Arc<Runtime>,
    prompt: impl Into<Message>,
    retries: usize,
) -> Result<ExtractionOutcome<T>, ExtractError>
where
    T: schemars::JsonSchema + serde::de::DeserializeOwned,
{
    config.output_schema = Some(schemars::schema_for!(T));
    config.output_mode = OutputMode::Native;
    if config.max_turns.is_none() {
        config.max_turns = Some(1);
    }
    extract_attempt_loop(
        config,
        provider,
        rt,
        prompt.into(),
        retries,
        deserialize_structured_output::<T>,
    )
    .await
}

/// Deserialize a typed structured response from the model's final text.
///
/// Tries a direct parse first (the common path — native output is already
/// clean JSON), then falls back to the first balanced JSON value in the text
/// so prose or markdown code fences around the JSON don't break weaker
/// best-effort output (#1928). Ported from the classic typed prompt path.
fn deserialize_structured_output<T: serde::de::DeserializeOwned>(
    text: &str,
) -> Result<T, serde_json::Error> {
    let trimmed = text.trim();
    match serde_json::from_str::<T>(trimmed) {
        Ok(value) => Ok(value),
        Err(direct_err) => {
            let Some(start) = trimmed.find(['{', '[']) else {
                return Err(direct_err);
            };
            serde_json::Deserializer::from_str(&trimmed[start..])
                .into_iter::<T>()
                .next()
                .unwrap_or(Err(direct_err))
        }
    }
}

/// The shared attempt/retry loop behind every `extract*` flavor: run a
/// session, parse the final output with `parse`, and on parse failure retry
/// with corrective feedback, accumulating usage across attempts.
async fn extract_attempt_loop<T>(
    config: AgentConfig,
    provider: ProviderConfig,
    rt: Arc<Runtime>,
    prompt: Message,
    retries: usize,
    parse: impl Fn(&str) -> Result<T, serde_json::Error>,
) -> Result<ExtractionOutcome<T>, ExtractError> {
    let mut next_prompt = prompt;
    let mut history: Vec<Message> = Vec::new();
    let mut last_error: Option<(serde_json::Error, String)> = None;
    let mut usage = Usage::new();

    for _attempt in 0..=retries {
        let mut session = AgentSession::new(
            config.clone(),
            provider.clone(),
            rt.clone(),
            next_prompt.clone(),
        );
        if !history.is_empty() {
            session = session.with_history(history.clone());
        }
        let done = session.run().await?;
        usage += done.usage;
        match parse(&done.output) {
            Ok(value) => return Ok(ExtractionOutcome { value, usage }),
            Err(error) => {
                // Commit this attempt's exchange to the retry history —
                // prompt first, so the next request's history never opens
                // with an assistant message (strict providers reject that) —
                // and make the corrective feedback the next attempt's prompt.
                history.push(next_prompt);
                history.push(Message::assistant(done.output.clone()));
                next_prompt = Message::user(format!(
                    "The previous output did not match the required schema ({error}). \
                     Answer again with ONLY a JSON object satisfying the schema."
                ));
                last_error = Some((error, done.output));
            }
        }
    }

    // The 0..=retries loop always runs at least once, so last_error is Some;
    // fall back to an empty-input parse error if it somehow is not.
    let (source, raw) = match last_error {
        Some(pair) => pair,
        None => match parse("") {
            Err(error) => (error, String::new()),
            Ok(_) => {
                return Err(ExtractError::Deserialization {
                    source: serde::de::Error::custom("extraction made no attempts"),
                    raw: String::new(),
                });
            }
        },
    };
    Err(ExtractError::Deserialization { source, raw })
}
