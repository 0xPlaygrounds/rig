//! Structured extraction over the session runtime.
//!
//! [`extract`] is the data-oriented successor of `Extractor<M, T>`: a
//! generic *function* over serde/schemars capability bounds (nothing stores
//! `T`), driving an [`AgentSession`] in Tool
//! output mode and deserializing the structured result.

use std::sync::Arc;

use rig_core::schemars;

use rig_core::message::ToolChoice;

use crate::agent::AgentConfig;
use crate::agent::hook::InvalidToolCallAction;
use crate::agent::run::OutputMode;
use crate::completion::{Message, PromptError, Usage};

use crate::provider::{ProviderConfig, Runtime};
use crate::session::{AgentSession, SessionEvent};

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

/// The exact output-tool name the deleted `ExtractorBuilder` pinned.
const CLASSIC_SUBMIT_TOOL_NAME: &str = "submit";

/// The exact output-tool description the deleted `ExtractorBuilder` pinned.
const CLASSIC_SUBMIT_TOOL_DESCRIPTION: &str =
    "Submit the structured data you extracted from the provided text.";

/// The exact preamble the deleted `ExtractorBuilder` set on its inner agent.
/// Byte-for-byte identical (including the trailing whitespace/newline the
/// original's raw string carried), so recorded classic-extractor exchanges stay
/// expressible through [`extract_with_options`].
const CLASSIC_EXTRACTOR_PREAMBLE: &str = "\
                    You are an AI assistant whose purpose is to extract structured data from the provided text.\n\
                    You will have access to a `submit` function that defines the structure of the data to extract from the provided text.\n\
                    Use the `submit` function to submit the structured data.\n\
                    Be sure to fill out every field and ALWAYS CALL THE `submit` function, even with default values!!!.
                ";

/// Per-extraction knobs that are *not* part of [`AgentConfig`], plus overrides
/// for the config fields an extraction protocol has to pin.
///
/// This is the replacement for the deleted `ExtractorBuilder<T>`'s bespoke
/// setup. [`Self::classic_extractor`] reproduces that setup exactly — output
/// tool named `submit`, the extraction preamble, [`ToolChoice::Required`], no
/// preamble augmentation, invalid non-`submit` calls ignored, a one-call budget,
/// and prompt-repeating retries — so recorded classic-extractor exchanges
/// remain expressible.
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct ExtractOptions {
    /// Chat history preceding the prompt (the classic
    /// `extract_with_chat_history`).
    pub history: Vec<Message>,
    /// Deserialization-failure retry budget (attempts = `retries + 1`).
    pub retries: usize,
    /// Overrides [`AgentConfig::output_tool_name`].
    pub output_tool_name: Option<String>,
    /// Overrides [`AgentConfig::output_tool_description`].
    pub output_tool_description: Option<String>,
    /// Overrides [`AgentConfig::augment_output_preamble`].
    pub augment_output_preamble: Option<bool>,
    /// Overrides [`AgentConfig::preamble`].
    pub preamble: Option<String>,
    /// Overrides [`AgentConfig::tool_choice`].
    pub tool_choice: Option<ToolChoice>,
    /// Overrides [`AgentConfig::max_turns`] (otherwise the flavor's default).
    pub max_turns: Option<usize>,
    /// Treat a call to anything but the output tool as irrelevant response
    /// content instead of failing the run
    /// ([`InvalidToolCallAction::Ignore`]).
    pub ignore_unhandled_invalid_tool_calls: bool,
    /// Retry by re-sending the *identical* request instead of appending the
    /// previous output plus corrective feedback to the history. The classic
    /// extractor retried this way, so multi-attempt recordings replay only
    /// with this set.
    pub repeat_prompt_on_retry: bool,
}

impl ExtractOptions {
    /// Default options: no history, no retries, no overrides.
    pub fn new() -> Self {
        Self::default()
    }

    /// The deleted `ExtractorBuilder<T>`'s exact configuration.
    pub fn classic_extractor() -> Self {
        Self {
            output_tool_name: Some(CLASSIC_SUBMIT_TOOL_NAME.to_string()),
            output_tool_description: Some(CLASSIC_SUBMIT_TOOL_DESCRIPTION.to_string()),
            augment_output_preamble: Some(false),
            preamble: Some(CLASSIC_EXTRACTOR_PREAMBLE.to_string()),
            tool_choice: Some(ToolChoice::Required),
            max_turns: Some(1),
            ignore_unhandled_invalid_tool_calls: true,
            repeat_prompt_on_retry: true,
            ..Self::default()
        }
    }

    /// Set the chat history preceding the prompt.
    pub fn with_history(mut self, history: impl IntoIterator<Item = Message>) -> Self {
        self.history = history.into_iter().collect();
        self
    }

    /// Set the deserialization-failure retry budget.
    pub fn with_retries(mut self, retries: usize) -> Self {
        self.retries = retries;
        self
    }

    /// Pin the synthetic output tool's name, description, and whether Tool
    /// output mode augments the preamble with calling instructions.
    pub fn with_output_tool(
        mut self,
        name: impl Into<String>,
        description: impl Into<String>,
        augment_preamble: bool,
    ) -> Self {
        self.output_tool_name = Some(name.into());
        self.output_tool_description = Some(description.into());
        self.augment_output_preamble = Some(augment_preamble);
        self
    }

    /// Override the extraction preamble.
    pub fn with_preamble(mut self, preamble: impl Into<String>) -> Self {
        self.preamble = Some(preamble.into());
        self
    }

    /// Override the tool-choice policy.
    pub fn with_tool_choice(mut self, tool_choice: ToolChoice) -> Self {
        self.tool_choice = Some(tool_choice);
        self
    }

    /// Override the total model-call budget.
    pub fn with_max_turns(mut self, max_turns: usize) -> Self {
        self.max_turns = Some(max_turns);
        self
    }

    /// Apply the config overrides onto `config`.
    fn apply(&self, config: &mut AgentConfig) {
        if let Some(name) = &self.output_tool_name {
            config.output_tool_name = Some(name.clone());
        }
        if let Some(description) = &self.output_tool_description {
            config.output_tool_description = Some(description.clone());
        }
        if let Some(augment) = self.augment_output_preamble {
            config.augment_output_preamble = augment;
        }
        if let Some(preamble) = &self.preamble {
            config.preamble = Some(preamble.clone());
        }
        if let Some(tool_choice) = &self.tool_choice {
            config.tool_choice = Some(tool_choice.clone());
        }
        if let Some(max_turns) = self.max_turns {
            config.max_turns = Some(max_turns);
        }
    }
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
    extract_attempt_loop(
        config,
        provider,
        rt,
        prompt.into(),
        ExtractOptions::new().with_retries(retries),
        |raw| serde_json::from_str::<T>(raw.trim()),
    )
    .await
}

/// [`extract_with_usage`] with full control over the extraction protocol:
/// history, retry budget and style, and the output-tool/preamble/tool-choice
/// pinning an interoperating protocol may require (see [`ExtractOptions`]).
///
/// `ExtractOptions::classic_extractor()` reproduces the deleted
/// `Extractor<T>`'s exchange exactly.
pub async fn extract_with_options<T>(
    mut config: AgentConfig,
    provider: ProviderConfig,
    rt: Arc<Runtime>,
    prompt: impl Into<Message>,
    options: ExtractOptions,
) -> Result<ExtractionOutcome<T>, ExtractError>
where
    T: schemars::JsonSchema + serde::de::DeserializeOwned,
{
    config.output_schema = Some(schemars::schema_for!(T));
    config.output_mode = OutputMode::Tool;
    if config.max_turns.is_none() {
        config.max_turns = Some(2);
    }
    options.apply(&mut config);
    extract_attempt_loop(config, provider, rt, prompt.into(), options, |raw| {
        serde_json::from_str::<T>(raw.trim())
    })
    .await
}

/// A fluent, non-generic front end for [`extract_with_options`].
///
/// The runner stores only concrete owned data — an [`AgentConfig`], a
/// [`ProviderConfig`], the shared [`Runtime`] handle, the prompt, and
/// [`ExtractOptions`]. It is **not** generic over the extracted type: the
/// payload is chosen at the terminal ([`Self::run`] / [`Self::run_with_usage`]),
/// so there is no `ExtractionRunner<T>` and no `PhantomData<T>`, and one
/// configured runner can produce different shapes.
///
/// ```no_run
/// # async fn run(agent: &rig_agent::Agent) -> Result<(), Box<dyn std::error::Error>> {
/// #[derive(serde::Deserialize, schemars::JsonSchema)]
/// struct Person {
///     name: String,
/// }
///
/// let person: Person = agent
///     .extractor("Alice is 30.")
///     .classic()
///     .retries(2)
///     .run()
///     .await?;
/// # let _ = person;
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct ExtractionRunner {
    config: AgentConfig,
    provider: ProviderConfig,
    rt: Arc<Runtime>,
    prompt: Message,
    options: ExtractOptions,
}

impl ExtractionRunner {
    /// Builds a runner from explicit parts.
    pub fn new(
        config: AgentConfig,
        provider: ProviderConfig,
        rt: Arc<Runtime>,
        prompt: impl Into<Message>,
    ) -> Self {
        Self {
            config,
            provider,
            rt,
            prompt: prompt.into(),
            options: ExtractOptions::new(),
        }
    }

    /// Adopts the deleted `ExtractorBuilder<T>`'s exact configuration.
    ///
    /// History and the retry budget are preserved, so this composes in any
    /// order with [`Self::history`] and [`Self::retries`].
    pub fn classic(mut self) -> Self {
        let history = std::mem::take(&mut self.options.history);
        let retries = self.options.retries;
        self.options = ExtractOptions::classic_extractor();
        self.options.history = history;
        self.options.retries = retries;
        self
    }

    /// Sets the deserialization-failure retry budget (attempts = `retries + 1`).
    pub fn retries(mut self, retries: usize) -> Self {
        self.options.retries = retries;
        self
    }

    /// Replaces the chat history preceding the prompt.
    pub fn history(mut self, history: impl IntoIterator<Item = Message>) -> Self {
        self.options.history = history.into_iter().collect();
        self
    }

    /// Overrides the system prompt for this extraction.
    pub fn preamble(mut self, preamble: impl Into<String>) -> Self {
        self.options.preamble = Some(preamble.into());
        self
    }

    /// Overrides the tool-choice policy.
    pub fn tool_choice(mut self, tool_choice: ToolChoice) -> Self {
        self.options.tool_choice = Some(tool_choice);
        self
    }

    /// Overrides the model-call budget.
    pub fn max_turns(mut self, max_turns: usize) -> Self {
        self.options.max_turns = Some(max_turns);
        self
    }

    /// Names the synthetic output tool.
    pub fn output_tool_name(mut self, name: impl Into<String>) -> Self {
        self.options.output_tool_name = Some(name.into());
        self
    }

    /// Describes the synthetic output tool.
    pub fn output_tool_description(mut self, description: impl Into<String>) -> Self {
        self.options.output_tool_description = Some(description.into());
        self
    }

    /// Whether Tool output mode augments the preamble with calling
    /// instructions.
    pub fn augment_output_preamble(mut self, augment: bool) -> Self {
        self.options.augment_output_preamble = Some(augment);
        self
    }

    /// Treat a call to anything but the output tool as irrelevant content
    /// rather than failing the run.
    pub fn ignore_unhandled_invalid_tool_calls(mut self, ignore: bool) -> Self {
        self.options.ignore_unhandled_invalid_tool_calls = ignore;
        self
    }

    /// Retry by re-sending the identical request instead of appending
    /// corrective feedback.
    pub fn repeat_prompt_on_retry(mut self, repeat: bool) -> Self {
        self.options.repeat_prompt_on_retry = repeat;
        self
    }

    /// Appends a context document always provided to the model.
    pub fn context(mut self, document: rig_core::completion::Document) -> Self {
        self.config.static_context.push(document);
        self
    }

    /// Appends context documents, in order.
    pub fn contexts(
        mut self,
        documents: impl IntoIterator<Item = rig_core::completion::Document>,
    ) -> Self {
        self.config.static_context.extend(documents);
        self
    }

    /// **Replaces** provider-specific request parameters, matching
    /// [`AgentBuilder::additional_params`](crate::AgentBuilder::additional_params).
    pub fn additional_params(mut self, params: serde_json::Value) -> Self {
        self.config.additional_params = Some(params);
        self
    }

    /// Sets the maximum output-token count.
    pub fn max_tokens(mut self, max_tokens: u64) -> Self {
        self.config.max_tokens = Some(max_tokens);
        self
    }

    /// Sets the sampling temperature.
    pub fn temperature(mut self, temperature: f64) -> Self {
        self.config.temperature = Some(temperature);
        self
    }

    /// The options this runner would extract with.
    pub fn options(&self) -> &ExtractOptions {
        &self.options
    }

    /// Runs the extraction, returning the deserialized value.
    ///
    /// The payload type is chosen here, not on the runner.
    pub async fn run<T>(self) -> Result<T, ExtractError>
    where
        T: schemars::JsonSchema + serde::de::DeserializeOwned,
    {
        self.run_with_usage::<T>()
            .await
            .map(|outcome| outcome.value)
    }

    /// Runs the extraction, returning the value plus accumulated token usage.
    pub async fn run_with_usage<T>(self) -> Result<ExtractionOutcome<T>, ExtractError>
    where
        T: schemars::JsonSchema + serde::de::DeserializeOwned,
    {
        extract_with_options(
            self.config,
            self.provider,
            self.rt,
            self.prompt,
            self.options,
        )
        .await
    }
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
        ExtractOptions::new().with_retries(retries),
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
pub(crate) fn deserialize_structured_output<T: serde::de::DeserializeOwned>(
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
/// session, parse the final output with `parse`, and on parse failure retry,
/// accumulating usage across attempts.
///
/// Retry style follows [`ExtractOptions::repeat_prompt_on_retry`]: by default
/// the failed attempt's exchange is committed to the retry history and the
/// corrective feedback becomes the next prompt; with the flag set the identical
/// request is re-sent (the classic extractor's semantics).
async fn extract_attempt_loop<T>(
    config: AgentConfig,
    provider: ProviderConfig,
    rt: Arc<Runtime>,
    prompt: Message,
    options: ExtractOptions,
    parse: impl Fn(&str) -> Result<T, serde_json::Error>,
) -> Result<ExtractionOutcome<T>, ExtractError> {
    let mut next_prompt = prompt;
    let mut history: Vec<Message> = options.history.clone();
    let mut last_error: Option<(serde_json::Error, String)> = None;
    let mut usage = Usage::new();

    for _attempt in 0..=options.retries {
        let mut session = AgentSession::new(
            config.clone(),
            provider.clone(),
            rt.clone(),
            next_prompt.clone(),
        );
        if !history.is_empty() {
            session = session.with_history(history.clone());
        }
        let done = run_extraction_session(session, &options).await?;
        usage += done.usage;
        match parse(&done.output) {
            Ok(value) => return Ok(ExtractionOutcome { value, usage }),
            Err(error) => {
                if !options.repeat_prompt_on_retry {
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
                }
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

/// Drive one extraction attempt to completion.
///
/// Identical to [`AgentSession::run`] except for the invalid-tool-call policy:
/// with [`ExtractOptions::ignore_unhandled_invalid_tool_calls`] a call to
/// anything but the output tool is dropped as irrelevant response content
/// instead of failing the run. Extraction sessions advertise no executable
/// tools, so no other event can surface.
async fn run_extraction_session(
    mut session: AgentSession,
    options: &ExtractOptions,
) -> Result<crate::agent::PromptResponse, PromptError> {
    if !options.ignore_unhandled_invalid_tool_calls {
        return session.run().await;
    }
    loop {
        match session.advance().await? {
            SessionEvent::Done(response) => return Ok(response),
            SessionEvent::InvalidToolCall(_) => {
                session.resolve_invalid(InvalidToolCallAction::ignore())?;
            }
            other => {
                return Err(PromptError::CompletionError(
                    crate::completion::CompletionError::ResponseError(format!(
                        "extraction session surfaced an unexpected event: {other:?}"
                    )),
                ));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::MockScript;
    use rig_core::OneOrMany;
    use rig_core::completion::CompletionResponse;
    use rig_core::message::{AssistantContent, ToolCall, ToolFunction};

    #[derive(Debug, PartialEq, serde::Deserialize, schemars::JsonSchema)]
    struct Person {
        name: String,
    }

    fn submit_response(name: &str) -> CompletionResponse {
        CompletionResponse::new(
            OneOrMany::one(AssistantContent::ToolCall(ToolCall::new(
                "id1".to_string(),
                ToolFunction::new(
                    CLASSIC_SUBMIT_TOOL_NAME.to_string(),
                    serde_json::json!({ "name": name }),
                ),
            ))),
            Usage::new(),
            "mock",
        )
    }

    /// `.classic()` on the runner must produce byte-identical requests to
    /// passing `ExtractOptions::classic_extractor()` to the free function —
    /// otherwise the fluent surface would silently diverge from recorded
    /// cassettes.
    #[tokio::test]
    async fn runner_classic_matches_the_free_function_request() {
        let via_runner_script = MockScript::from_responses(vec![submit_response("Ada")]);
        let runner_probe = via_runner_script.clone();
        let runner_value = ExtractionRunner::new(
            AgentConfig::new().with_max_tokens(384),
            ProviderConfig::Mock(via_runner_script),
            Arc::new(Runtime::new()),
            "Hello, my name is Ada.",
        )
        .classic()
        .run::<Person>()
        .await
        .expect("runner extraction succeeds");

        let via_free_script = MockScript::from_responses(vec![submit_response("Ada")]);
        let free_probe = via_free_script.clone();
        let free_value = extract_with_options::<Person>(
            AgentConfig::new().with_max_tokens(384),
            ProviderConfig::Mock(via_free_script),
            Arc::new(Runtime::new()),
            "Hello, my name is Ada.",
            ExtractOptions::classic_extractor(),
        )
        .await
        .expect("free extraction succeeds")
        .value;

        assert_eq!(runner_value, free_value);

        let runner_requests = runner_probe.requests();
        let free_requests = free_probe.requests();
        assert_eq!(runner_requests.len(), free_requests.len());
        let runner_request = runner_requests.first().expect("one request");
        let free_request = free_requests.first().expect("one request");
        assert_eq!(runner_request.chat_history, free_request.chat_history);
        assert_eq!(runner_request.tools, free_request.tools);
        assert_eq!(runner_request.tool_choice, free_request.tool_choice);
        assert_eq!(runner_request.max_tokens, free_request.max_tokens);
        assert_eq!(runner_request.temperature, free_request.temperature);
        assert_eq!(
            runner_request.additional_params,
            free_request.additional_params
        );
    }

    /// `.classic()` must not discard history or retries set around it, in
    /// either order.
    #[test]
    fn classic_preserves_history_and_retries_in_any_order() {
        let runner = || {
            ExtractionRunner::new(
                AgentConfig::new(),
                ProviderConfig::Mock(MockScript::from_responses(vec![])),
                Arc::new(Runtime::new()),
                "prompt",
            )
        };

        let before = runner()
            .history([Message::from("earlier")])
            .retries(3)
            .classic();
        let after = runner()
            .classic()
            .history([Message::from("earlier")])
            .retries(3);

        for configured in [before, after] {
            let options = configured.options();
            assert_eq!(options.retries, 3);
            assert_eq!(options.history.len(), 1);
            // …while still carrying the classic preset.
            assert_eq!(
                options.output_tool_name.as_deref(),
                Some(CLASSIC_SUBMIT_TOOL_NAME)
            );
            assert_eq!(options.tool_choice, Some(ToolChoice::Required));
            assert!(options.repeat_prompt_on_retry);
        }
    }

    /// One runner value, cloned, extracting two *different* payload types —
    /// only possible because the type is chosen at the terminal rather than
    /// baked into the runner.
    #[tokio::test]
    async fn one_runner_serves_two_payload_types() {
        #[derive(Debug, PartialEq, serde::Deserialize, schemars::JsonSchema)]
        struct Name {
            name: String,
        }

        let runner = ExtractionRunner::new(
            AgentConfig::new(),
            ProviderConfig::Mock(MockScript::from_responses(vec![
                submit_response("Ada"),
                submit_response("Ada"),
            ])),
            Arc::new(Runtime::new()),
            "Hello, my name is Ada.",
        )
        .classic();

        let as_person = runner.clone().run::<Person>().await.expect("Person");
        let as_name = runner.run::<Name>().await.expect("Name");

        assert_eq!(as_person.name, "Ada");
        assert_eq!(as_name.name, "Ada");
    }

    /// Config-backed setters must reach the built request.
    #[tokio::test]
    async fn config_setters_reach_the_request_and_payload_is_chosen_at_the_terminal() {
        let script = MockScript::from_responses(vec![submit_response("Ada")]);
        let probe = script.clone();

        let _person = ExtractionRunner::new(
            AgentConfig::new(),
            ProviderConfig::Mock(script),
            Arc::new(Runtime::new()),
            "Hello, my name is Ada.",
        )
        .classic()
        .max_tokens(256)
        .temperature(0.1)
        .additional_params(serde_json::json!({ "custom": true }))
        .run::<Person>()
        .await
        .expect("extraction succeeds");

        let requests = probe.requests();
        let request = requests.first().expect("one request");
        assert_eq!(request.max_tokens, Some(256));
        assert_eq!(request.temperature, Some(0.1));
        assert_eq!(
            request
                .additional_params
                .as_ref()
                .and_then(|p| p.get("custom")),
            Some(&serde_json::json!(true))
        );
    }

    /// The deleted `Extractor<T>`'s exchange must stay expressible: one model
    /// call carrying the extraction preamble verbatim, a single advertised
    /// tool named `submit` with the classic description and `T`'s schema, and
    /// `ToolChoice::Required`. Recorded classic-extractor cassettes replay
    /// only if every one of these matches.
    #[tokio::test]
    async fn classic_extractor_preset_reproduces_the_recorded_request() {
        let script = MockScript::from_responses(vec![submit_response("Ada")]);
        let probe = script.clone();

        let outcome = extract_with_options::<Person>(
            AgentConfig::new().with_max_tokens(384),
            ProviderConfig::Mock(script),
            Arc::new(Runtime::new()),
            "Hello, my name is Ada.",
            ExtractOptions::classic_extractor(),
        )
        .await
        .expect("extraction should succeed");
        assert_eq!(
            outcome.value,
            Person {
                name: "Ada".to_string()
            }
        );

        let requests = probe.requests();
        assert_eq!(requests.len(), 1, "one model call, exactly as max_turns(1)");
        let request = requests.first().expect("one recorded request");
        // `prepare_request` carries the preamble as the leading system
        // message (system instructions are canonical messages).
        let system = request.chat_history.first();
        let Message::System { content } = system else {
            panic!("the preamble must lead the history as a system message: {system:?}");
        };
        assert_eq!(
            content, CLASSIC_EXTRACTOR_PREAMBLE,
            "the extraction preamble must be verbatim (no Tool-mode augmentation)"
        );
        assert_eq!(request.max_tokens, Some(384));
        assert_eq!(request.tool_choice, Some(ToolChoice::Required));
        let tools = &request.tools;
        assert_eq!(tools.len(), 1, "only the synthetic output tool: {tools:?}");
        let tool = tools.first().expect("the output tool");
        assert_eq!(tool.name, CLASSIC_SUBMIT_TOOL_NAME);
        assert_eq!(tool.description, CLASSIC_SUBMIT_TOOL_DESCRIPTION);
        assert_eq!(
            tool.parameters,
            serde_json::to_value(schemars::schema_for!(Person)).expect("schema as json"),
        );
    }

    /// A non-output tool call is dropped as irrelevant response content when
    /// `ignore_unhandled_invalid_tool_calls` is set (the classic extractor's
    /// `IgnoreForExtractor` policy), letting the sibling `submit` call
    /// finalize the turn instead of failing the run.
    #[tokio::test]
    async fn ignored_invalid_call_does_not_fail_the_extraction() {
        let response = CompletionResponse::new(
            OneOrMany::many([
                AssistantContent::ToolCall(ToolCall::new(
                    "id0".to_string(),
                    ToolFunction::new("unrelated".to_string(), serde_json::json!({})),
                )),
                AssistantContent::ToolCall(ToolCall::new(
                    "id1".to_string(),
                    ToolFunction::new(
                        CLASSIC_SUBMIT_TOOL_NAME.to_string(),
                        serde_json::json!({ "name": "Ada" }),
                    ),
                )),
            ])
            .expect("two content items"),
            Usage::new(),
            "mock",
        );

        let outcome = extract_with_options::<Person>(
            AgentConfig::new(),
            ProviderConfig::Mock(MockScript::from_responses(vec![response])),
            Arc::new(Runtime::new()),
            "Hello, my name is Ada.",
            ExtractOptions::classic_extractor(),
        )
        .await
        .expect("the unrelated call should be ignored");
        assert_eq!(outcome.value.name, "Ada");
    }

    /// `repeat_prompt_on_retry` re-sends the identical request instead of
    /// appending corrective feedback — the classic extractor's retry style, so
    /// multi-attempt recordings stay replayable.
    #[tokio::test]
    async fn classic_retry_repeats_the_identical_request() {
        let script = MockScript::from_responses(vec![
            CompletionResponse::new(
                OneOrMany::one(AssistantContent::text("not json")),
                Usage::new(),
                "mock",
            ),
            submit_response("Ada"),
        ]);
        let probe = script.clone();

        let outcome = extract_with_options::<Person>(
            AgentConfig::new(),
            ProviderConfig::Mock(script),
            Arc::new(Runtime::new()),
            "Hello, my name is Ada.",
            ExtractOptions::classic_extractor().with_retries(1),
        )
        .await
        .expect("the retry should succeed");
        assert_eq!(outcome.value.name, "Ada");

        let requests = probe.requests();
        assert_eq!(requests.len(), 2);
        let first = requests.first().expect("first attempt");
        let second = requests.get(1).expect("second attempt");
        assert_eq!(
            first.chat_history.len(),
            second.chat_history.len(),
            "the retry must not append the failed attempt's exchange"
        );
        // System instructions are canonical messages now, so comparing the
        // histories above already covers them; assert the leading system
        // message explicitly to keep the intent visible.
        assert_eq!(first.chat_history.first(), second.chat_history.first());
        assert_eq!(first.tools.len(), second.tools.len());
    }
}
