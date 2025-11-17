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
