//! This module primarily concerns being able to orchestrate telemetry across a given pipeline or workflow.
//! This includes tracing, being able to send traces to an OpenTelemetry collector, setting up your
//! agents with the correct tracing style so you can emit the right traces for platforms like Langfuse,
//! and more.

use crate::completion::GetTokenUsage;
use serde::Serialize;

pub trait ProviderRequestExt {
    type InputMessage: Serialize;

    fn get_input_messages(&self) -> Vec<Self::InputMessage>;
    fn get_system_prompt(&self) -> Option<String>;
    fn get_model_name(&self) -> String;
    fn get_prompt(&self) -> Option<String>;
}

pub trait ProviderResponseExt {
    type OutputMessage: Serialize;
    type Usage: Serialize;

    fn get_response_id(&self) -> Option<String>;

    fn get_response_model_name(&self) -> Option<String>;

    fn get_output_messages(&self) -> Vec<Self::OutputMessage>;

    fn get_text_response(&self) -> Option<String>;

    fn get_usage(&self) -> Option<Self::Usage>;
}

/// A trait designed specifically to be used with Spans for the purpose of recording telemetry.
/// Nearly all methods
pub trait SpanCombinator {
    fn record_preamble(&self, preamble: &Option<String>);

    fn record_token_usage<U>(&self, usage: &U)
    where
        U: GetTokenUsage;

    fn record_response_metadata<R>(&self, response: &R)
    where
        R: ProviderResponseExt;

    fn record_model_input<T>(&self, messages: &T)
    where
        T: Serialize;

    fn record_model_output<T>(&self, messages: &T)
    where
        T: Serialize;
}

impl SpanCombinator for tracing::Span {
    fn record_preamble(&self, preamble: &Option<String>) {
        if self.is_disabled() {
            return;
        }

        if let Some(preamble) = preamble {
            self.record("gen_ai.system_instructions", preamble);
        }
    }

    fn record_token_usage<U>(&self, usage: &U)
    where
        U: GetTokenUsage,
    {
        if self.is_disabled() {
            return;
        }

        if let Some(usage) = usage.token_usage() {
            self.record("gen_ai.usage.input_tokens", usage.input_tokens);
            self.record("gen_ai.usage.output_tokens", usage.output_tokens);
        }
    }

    fn record_response_metadata<R>(&self, response: &R)
    where
        R: ProviderResponseExt,
    {
        if self.is_disabled() {
            return;
        }

        if let Some(id) = response.get_response_id() {
            self.record("gen_ai.response.id", id);
        }

        if let Some(model_name) = response.get_response_model_name() {
            self.record("gen_ai.response.model_name", model_name);
        }
    }

    fn record_model_input<T>(&self, input: &T)
    where
        T: Serialize,
    {
        if self.is_disabled() {
            return;
        }

        let input_as_json_string =
            serde_json::to_string(input).expect("Serializing a Rust type to JSON should not break");

        self.record("gen_ai.input.messages", input_as_json_string);
    }

    fn record_model_output<T>(&self, output: &T)
    where
        T: Serialize,
    {
        if self.is_disabled() {
            return;
        }

        let output_as_json_string = serde_json::to_string(output)
            .expect("Serializing a Rust type to JSON should not break");

        self.record("gen_ai.output.messages", output_as_json_string);
    }
}

/// Telemetry configuration for all LLM model related ops (completions, agent workflows).
/// By default, all options are set to false.
#[derive(Clone, Debug, Default)]
pub struct TelemetryConfiguration {
    /// Whether or not debug-print logging should be shown (ie, the raw contents of a given request to a provider).
    /// These are provided as `trace` level logs due to potentially containing large JSON blobs.
    pub debug_logging: bool,
    /// Whether or not the preamble ("system prompt") should be included in traces.
    pub include_preamble: bool,
    /// Whether or not message contents should be included in traces.
    /// Ensure this is disabled if you would like to exclude message contents for PII purposes.
    pub include_message_contents: bool,
}

impl TelemetryConfiguration {
    /// Creates a new instance of `TelemetryConfiguration`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Whether or not to include debug logging.
    pub fn debug_logging(mut self, debug_logging: bool) -> Self {
        self.debug_logging = debug_logging;

        self
    }

    /// Whether or not to include the preamble ("system prompt")
    pub fn include_preamble(mut self, include_preamble: bool) -> Self {
        self.include_preamble = include_preamble;

        self
    }

    /// Whether or not to include message contents in traces.
    pub fn include_message_contents(mut self, include_message_contents: bool) -> Self {
        self.include_message_contents = include_message_contents;

        self
    }

    /// Creates a telemetry configuration with all telemetry enabled.
    /// This will log ALL messages, prompts as well as debug logging in Rig's traces.
    pub fn all_telemetry_enabled() -> Self {
        Self {
            debug_logging: true,
            include_preamble: true,
            include_message_contents: true,
        }
    }
}
