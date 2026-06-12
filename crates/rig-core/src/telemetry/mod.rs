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

        let usage = usage.token_usage();
        // Zero-valued usage is the documented sentinel for missing provider
        // usage metrics; leave the span fields unset.
        if usage.has_values() {
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
            self.record(
                "gen_ai.usage.tool_use_prompt_tokens",
                usage.tool_use_prompt_tokens,
            );
            self.record("gen_ai.usage.reasoning_tokens", usage.reasoning_tokens);
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
            self.record("gen_ai.response.model", model_name);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::completion::{GetTokenUsage, Usage};
    use std::sync::{Arc, Mutex};
    use tracing::field::{Field, Visit};
    use tracing::{Id, Subscriber};
    use tracing_subscriber::layer::{Context, SubscriberExt};
    use tracing_subscriber::{Layer, Registry, registry::LookupSpan};

    #[derive(Clone)]
    struct TestUsage(Usage);

    impl GetTokenUsage for TestUsage {
        fn token_usage(&self) -> Usage {
            self.0
        }
    }

    #[derive(Clone, Default)]
    struct CapturedFields(Arc<Mutex<Vec<(String, u64)>>>);

    impl CapturedFields {
        fn push(&self, name: &str, value: u64) {
            if let Ok(mut fields) = self.0.lock() {
                fields.push((name.to_string(), value));
            }
        }

        fn contains(&self, name: &str, value: u64) -> bool {
            self.0.lock().is_ok_and(|fields| {
                fields
                    .iter()
                    .any(|field| field == &(name.to_string(), value))
            })
        }
    }

    struct FieldCaptureLayer {
        fields: CapturedFields,
    }

    impl<S> Layer<S> for FieldCaptureLayer
    where
        S: Subscriber,
        S: for<'lookup> LookupSpan<'lookup>,
    {
        fn on_record(&self, _span: &Id, values: &tracing::span::Record<'_>, _ctx: Context<'_, S>) {
            values.record(&mut FieldCaptureVisitor {
                fields: self.fields.clone(),
            });
        }
    }

    struct FieldCaptureVisitor {
        fields: CapturedFields,
    }

    impl Visit for FieldCaptureVisitor {
        fn record_u64(&mut self, field: &Field, value: u64) {
            self.fields.push(field.name(), value);
        }

        fn record_debug(&mut self, _field: &Field, _value: &dyn std::fmt::Debug) {}
    }

    #[test]
    fn record_token_usage_records_tool_use_prompt_tokens() {
        let fields = CapturedFields::default();
        let subscriber = Registry::default().with(FieldCaptureLayer {
            fields: fields.clone(),
        });
        let usage = TestUsage(Usage {
            input_tokens: 1,
            output_tokens: 2,
            total_tokens: 15,
            cached_input_tokens: 3,
            cache_creation_input_tokens: 4,
            tool_use_prompt_tokens: 12,
            reasoning_tokens: 5,
        });

        // Scoped-subscriber tests must not run concurrently; see
        // `test_utils::scoped_tracing_subscriber_guard`.
        let _isolation = crate::test_utils::scoped_tracing_subscriber_guard_blocking();
        tracing::subscriber::with_default(subscriber, || {
            let span = tracing::info_span!(
                "usage_recording",
                gen_ai.usage.input_tokens = tracing::field::Empty,
                gen_ai.usage.output_tokens = tracing::field::Empty,
                gen_ai.usage.cache_read.input_tokens = tracing::field::Empty,
                gen_ai.usage.cache_creation.input_tokens = tracing::field::Empty,
                gen_ai.usage.tool_use_prompt_tokens = tracing::field::Empty,
                gen_ai.usage.reasoning_tokens = tracing::field::Empty,
            );

            span.record_token_usage(&usage);
        });

        assert!(fields.contains("gen_ai.usage.tool_use_prompt_tokens", 12));
    }
}
