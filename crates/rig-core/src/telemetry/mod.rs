//! This module primarily concerns being able to orchestrate telemetry across a given pipeline or workflow.
//! This includes tracing, being able to send traces to an OpenTelemetry collector, setting up your
//! agents with the correct tracing style so you can emit the right traces for platforms like Langfuse,
//! and more.

use crate::completion::GetTokenUsage;
use serde::Serialize;

/// Provider request metadata used to populate GenAI telemetry spans.
pub trait ProviderRequestExt {
    /// Provider-native message type used for serialized input messages.
    type InputMessage: Serialize;

    /// Returns serialized input messages sent to the provider.
    fn get_input_messages(&self) -> Vec<Self::InputMessage>;
    /// Returns the system prompt, if represented separately by the provider.
    fn get_system_prompt(&self) -> Option<String>;
    /// Returns the model name requested from the provider.
    fn get_model_name(&self) -> String;
    /// Returns the primary prompt text, when available.
    fn get_prompt(&self) -> Option<String>;
}

/// Provider response metadata used to populate GenAI telemetry spans.
pub trait ProviderResponseExt {
    /// Provider-native output message type.
    type OutputMessage: Serialize;
    /// Provider-native usage type.
    type Usage: Serialize;

    /// Returns the provider response ID, if supplied.
    fn get_response_id(&self) -> Option<String>;

    /// Returns the provider response model name, if supplied.
    fn get_response_model_name(&self) -> Option<String>;

    /// Returns serialized output messages produced by the provider.
    fn get_output_messages(&self) -> Vec<Self::OutputMessage>;

    /// Returns the primary text response, when available.
    fn get_text_response(&self) -> Option<String>;

    /// Returns provider-native usage metrics, if supplied.
    fn get_usage(&self) -> Option<Self::Usage>;
}

/// A trait designed specifically to be used with Spans for the purpose of recording telemetry.
/// Implemented for [`tracing::Span`] to record GenAI semantic convention fields.
pub trait SpanCombinator {
    /// Record Rig-normalized token usage fields on the span.
    fn record_token_usage<U>(&self, usage: &U)
    where
        U: GetTokenUsage;

    /// Record provider response metadata such as response ID and model name.
    fn record_response_metadata<R>(&self, response: &R)
    where
        R: ProviderResponseExt;

    /// Record serialized model input messages.
    fn record_model_input<T>(&self, messages: &T)
    where
        T: Serialize;

    /// Record serialized model output messages.
    fn record_model_output<T>(&self, messages: &T)
    where
        T: Serialize;
}

impl SpanCombinator for tracing::Span {
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
            self.record(
                "gen_ai.usage.cache_read.input_tokens",
                usage.cached_input_tokens,
            );
            self.record(
                "gen_ai.usage.cache_creation.input_tokens",
                usage.cache_creation_input_tokens,
            );
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

        if let Ok(input_as_json_string) = serde_json::to_string(input) {
            self.record("gen_ai.input.messages", input_as_json_string);
        }
    }

    fn record_model_output<T>(&self, output: &T)
    where
        T: Serialize,
    {
        if self.is_disabled() {
            return;
        }

        if let Ok(output_as_json_string) = serde_json::to_string(output) {
            self.record("gen_ai.output.messages", output_as_json_string);
        }
    }
}
