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
use crate::completion::{Message, PromptError};

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
    mut config: AgentConfig,
    provider: ProviderConfig,
    rt: Arc<Runtime>,
    prompt: impl Into<Message>,
    retries: usize,
) -> Result<T, ExtractError>
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

    let prompt = prompt.into();
    let mut history: Vec<Message> = Vec::new();
    let mut last_error: Option<(serde_json::Error, String)> = None;

    for _attempt in 0..=retries {
        let mut session = AgentSession::new(
            config.clone(),
            provider.clone(),
            rt.clone(),
            prompt.clone(),
        );
        if !history.is_empty() {
            session = session.with_history(history.clone());
        }
        let done = session.run().await?;
        match serde_json::from_str::<T>(done.output.trim()) {
            Ok(value) => return Ok(value),
            Err(error) => {
                history.push(Message::assistant(done.output.clone()));
                history.push(Message::user(format!(
                    "The previous output did not match the required schema ({error}). \
                     Answer again with ONLY a JSON object satisfying the schema."
                )));
                last_error = Some((error, done.output));
            }
        }
    }

    // The 0..=retries loop always runs at least once, so last_error is Some;
    // fall back to an empty-input parse error if it somehow is not.
    let (source, raw) = match last_error {
        Some(pair) => pair,
        None => match serde_json::from_str::<T>("") {
            Err(error) => (error, String::new()),
            Ok(_) => return Err(ExtractError::Deserialization {
                source: serde::de::Error::custom("extraction made no attempts"),
                raw: String::new(),
            }),
        },
    };
    Err(ExtractError::Deserialization { source, raw })
}
