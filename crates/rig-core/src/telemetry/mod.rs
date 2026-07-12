//! This module primarily concerns being able to orchestrate telemetry across a given pipeline or workflow.
//! This includes tracing, being able to send traces to an OpenTelemetry collector, setting up your
//! agents with the correct tracing style so you can emit the right traces for platforms like Langfuse,
//! and more.

use crate::completion::GetTokenUsage;
use serde::Serialize;

macro_rules! new_completion_span {
    ($name:literal, $provider:expr, $request_model:expr, $operation:expr, $system:expr) => {
        tracing::info_span!(
            target: "rig::completions",
            $name,
            gen_ai.operation.name = $operation,
            gen_ai.provider.name = $provider,
            gen_ai.request.model = $request_model,
            gen_ai.system_instructions = $system,
            gen_ai.response.id = tracing::field::Empty,
            gen_ai.response.model = tracing::field::Empty,
            gen_ai.usage.input_tokens = tracing::field::Empty,
            gen_ai.usage.output_tokens = tracing::field::Empty,
            gen_ai.usage.cache_read.input_tokens = tracing::field::Empty,
            gen_ai.usage.cache_creation.input_tokens = tracing::field::Empty,
            gen_ai.usage.tool_use_prompt_tokens = tracing::field::Empty,
            gen_ai.usage.reasoning_tokens = tracing::field::Empty,
            gen_ai.input.messages = tracing::field::Empty,
            gen_ai.output.messages = tracing::field::Empty,
        )
    };
}

/// A supported GenAI completion operation and its canonical span name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum CompletionOperation {
    /// A non-streaming chat completion.
    Chat,
    /// A streaming chat completion.
    ChatStreaming,
    /// A Gemini generate-content request.
    GenerateContent,
    /// A Gemini Interactions API request.
    Interactions,
    /// A streaming Gemini Interactions API request.
    InteractionsStreaming,
}

impl CompletionOperation {
    fn as_str(self) -> &'static str {
        match self {
            Self::Chat => "chat",
            Self::ChatStreaming => "chat_streaming",
            Self::GenerateContent => "generate_content",
            Self::Interactions => "interactions",
            Self::InteractionsStreaming => "interactions_streaming",
        }
    }
}

/// Builder for a canonical GenAI completion span.
///
/// Rig-owned `rig::agent_chat` spans are enriched and reused so an agent turn
/// has exactly one model span. Other ambient spans remain parents of a newly
/// created `rig::completions` span.
pub struct CompletionSpanBuilder<'a> {
    provider: &'a str,
    request_model: &'a str,
    operation: CompletionOperation,
    system_instructions: Option<&'a str>,
}

impl<'a> CompletionSpanBuilder<'a> {
    /// Create a completion-span builder for a provider request.
    pub fn new(provider: &'a str, request_model: &'a str, operation: CompletionOperation) -> Self {
        Self {
            provider,
            request_model,
            operation,
            system_instructions: None,
        }
    }

    /// Set the system instructions sent with the request.
    pub fn system_instructions(mut self, system_instructions: Option<&'a str>) -> Self {
        self.system_instructions = system_instructions;
        self
    }

    /// Build a canonical completion span or enrich Rig's current agent-chat span.
    pub fn build(self) -> tracing::Span {
        let current = tracing::Span::current();
        if current
            .metadata()
            .is_some_and(|metadata| metadata.target() == "rig::agent_chat")
        {
            current.record("gen_ai.operation.name", self.operation.as_str());
            current.record("gen_ai.provider.name", self.provider);
            current.record("gen_ai.request.model", self.request_model);
            if let Some(system_instructions) = self.system_instructions {
                current.record("gen_ai.system_instructions", system_instructions);
            }
            return current;
        }

        let operation = self.operation.as_str();
        match self.operation {
            CompletionOperation::Chat => new_completion_span!(
                "chat",
                self.provider,
                self.request_model,
                operation,
                self.system_instructions
            ),
            CompletionOperation::ChatStreaming => new_completion_span!(
                "chat_streaming",
                self.provider,
                self.request_model,
                operation,
                self.system_instructions
            ),
            CompletionOperation::GenerateContent => new_completion_span!(
                "generate_content",
                self.provider,
                self.request_model,
                operation,
                self.system_instructions
            ),
            CompletionOperation::Interactions => new_completion_span!(
                "interactions",
                self.provider,
                self.request_model,
                operation,
                self.system_instructions
            ),
            CompletionOperation::InteractionsStreaming => new_completion_span!(
                "interactions_streaming",
                self.provider,
                self.request_model,
                operation,
                self.system_instructions
            ),
        }
    }
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

    #[derive(Clone, Default)]
    struct CapturedSpan(Arc<Mutex<Option<CapturedSpanData>>>);

    struct CapturedSpanData {
        name: String,
        target: String,
        parent_name: Option<String>,
        fields: Vec<String>,
        initial_values: Vec<(String, String)>,
        recorded_values: Vec<(String, String)>,
    }

    struct SpanCaptureLayer {
        span: CapturedSpan,
    }

    #[derive(Default)]
    struct StringFieldVisitor {
        values: Vec<(String, String)>,
    }

    impl Visit for StringFieldVisitor {
        fn record_str(&mut self, field: &Field, value: &str) {
            self.values
                .push((field.name().to_owned(), value.to_owned()));
        }

        fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
            self.values
                .push((field.name().to_owned(), format!("{value:?}")));
        }
    }

    impl<S> Layer<S> for SpanCaptureLayer
    where
        S: Subscriber,
        S: for<'lookup> LookupSpan<'lookup>,
    {
        fn on_new_span(
            &self,
            attrs: &tracing::span::Attributes<'_>,
            _id: &Id,
            ctx: Context<'_, S>,
        ) {
            if let Ok(mut captured) = self.span.0.lock() {
                let mut visitor = StringFieldVisitor::default();
                attrs.record(&mut visitor);
                let parent_name = if let Some(parent) = attrs.parent() {
                    ctx.span(parent)
                        .map(|span| span.metadata().name().to_owned())
                } else if attrs.is_contextual() {
                    ctx.lookup_current()
                        .map(|span| span.metadata().name().to_owned())
                } else {
                    None
                };
                *captured = Some(CapturedSpanData {
                    name: attrs.metadata().name().to_owned(),
                    target: attrs.metadata().target().to_owned(),
                    parent_name,
                    fields: attrs
                        .metadata()
                        .fields()
                        .iter()
                        .map(|field| field.name().to_owned())
                        .collect(),
                    initial_values: visitor.values,
                    recorded_values: Vec::new(),
                });
            }
        }

        fn on_record(&self, _span: &Id, values: &tracing::span::Record<'_>, _ctx: Context<'_, S>) {
            if let Ok(mut captured) = self.span.0.lock()
                && let Some(captured) = captured.as_mut()
            {
                let mut visitor = StringFieldVisitor::default();
                values.record(&mut visitor);
                captured.recorded_values.extend(visitor.values);
            }
        }
    }

    fn contains_string(values: &[(String, String)], field: &str, value: &str) -> bool {
        values
            .iter()
            .any(|candidate| candidate == &(field.to_owned(), value.to_owned()))
    }

    #[test]
    fn completion_span_uses_canonical_names_fields_and_initial_attributes() {
        let _isolation = crate::test_utils::scoped_tracing_subscriber_guard_blocking();

        for (operation, expected_name) in [
            (CompletionOperation::Chat, "chat"),
            (CompletionOperation::ChatStreaming, "chat_streaming"),
            (CompletionOperation::GenerateContent, "generate_content"),
            (CompletionOperation::Interactions, "interactions"),
            (
                CompletionOperation::InteractionsStreaming,
                "interactions_streaming",
            ),
        ] {
            let captured = CapturedSpan::default();
            let subscriber = Registry::default().with(SpanCaptureLayer {
                span: captured.clone(),
            });
            tracing::subscriber::with_default(subscriber, || {
                let span = CompletionSpanBuilder::new("openai", "gpt-5", operation)
                    .system_instructions(Some("system prompt"))
                    .build();
                assert!(!span.is_disabled());
            });

            let Ok(captured) = captured.0.lock() else {
                panic!("captured span lock poisoned");
            };
            let Some(span) = captured.as_ref() else {
                panic!("completion span was not created");
            };
            assert_eq!(span.name, expected_name);
            assert_eq!(span.target, "rig::completions");
            assert_eq!(span.parent_name, None);
            for (field, value) in [
                ("gen_ai.operation.name", expected_name),
                ("gen_ai.provider.name", "openai"),
                ("gen_ai.request.model", "gpt-5"),
                ("gen_ai.system_instructions", "system prompt"),
            ] {
                assert!(
                    contains_string(&span.initial_values, field, value),
                    "missing initial {field}={value}"
                );
            }
            assert!(span.recorded_values.is_empty());
            assert!(
                !span
                    .initial_values
                    .iter()
                    .any(|(field, _)| field == "gen_ai.response.model")
            );
            for field in [
                "gen_ai.operation.name",
                "gen_ai.provider.name",
                "gen_ai.request.model",
                "gen_ai.system_instructions",
                "gen_ai.response.id",
                "gen_ai.response.model",
                "gen_ai.usage.input_tokens",
                "gen_ai.usage.output_tokens",
                "gen_ai.usage.cache_read.input_tokens",
                "gen_ai.usage.cache_creation.input_tokens",
                "gen_ai.usage.tool_use_prompt_tokens",
                "gen_ai.usage.reasoning_tokens",
                "gen_ai.input.messages",
                "gen_ai.output.messages",
            ] {
                assert!(
                    span.fields.iter().any(|candidate| candidate == field),
                    "missing {field}"
                );
            }
        }
    }

    #[test]
    fn unrelated_ambient_span_is_parent_not_adopted() {
        let captured = CapturedSpan::default();
        let subscriber = Registry::default().with(SpanCaptureLayer {
            span: captured.clone(),
        });
        let _isolation = crate::test_utils::scoped_tracing_subscriber_guard_blocking();
        tracing::subscriber::with_default(subscriber, || {
            let ambient = tracing::info_span!(target: "application", "ambient");
            let _guard = ambient.enter();
            let span =
                CompletionSpanBuilder::new("openai", "gpt-5", CompletionOperation::Chat).build();
            assert_ne!(span.id(), ambient.id());
        });

        let Ok(captured) = captured.0.lock() else {
            panic!("captured span lock poisoned");
        };
        let Some(span) = captured.as_ref() else {
            panic!("completion span was not captured");
        };
        assert_eq!(span.target, "rig::completions");
        assert_eq!(span.parent_name.as_deref(), Some("ambient"));
    }

    #[test]
    fn agent_chat_span_is_adopted_and_enriched() {
        let captured = CapturedSpan::default();
        let subscriber = Registry::default().with(SpanCaptureLayer {
            span: captured.clone(),
        });
        let _isolation = crate::test_utils::scoped_tracing_subscriber_guard_blocking();
        tracing::subscriber::with_default(subscriber, || {
            let agent_chat = tracing::info_span!(
                target: "rig::agent_chat",
                "chat_streaming",
                gen_ai.operation.name = tracing::field::Empty,
                gen_ai.provider.name = tracing::field::Empty,
                gen_ai.request.model = tracing::field::Empty,
                gen_ai.system_instructions = tracing::field::Empty,
            );
            let _guard = agent_chat.enter();
            let span = CompletionSpanBuilder::new(
                "anthropic",
                "claude-sonnet",
                CompletionOperation::ChatStreaming,
            )
            .system_instructions(Some("provider system"))
            .build();
            assert_eq!(span.id(), agent_chat.id());
        });

        let Ok(captured) = captured.0.lock() else {
            panic!("captured span lock poisoned");
        };
        let Some(span) = captured.as_ref() else {
            panic!("agent chat span was not captured");
        };
        for (field, value) in [
            ("gen_ai.operation.name", "chat_streaming"),
            ("gen_ai.provider.name", "anthropic"),
            ("gen_ai.request.model", "claude-sonnet"),
            ("gen_ai.system_instructions", "provider system"),
        ] {
            assert!(contains_string(&span.recorded_values, field, value));
        }
    }

    #[test]
    fn absent_provider_system_does_not_overwrite_agent_instructions() {
        let captured = CapturedSpan::default();
        let subscriber = Registry::default().with(SpanCaptureLayer {
            span: captured.clone(),
        });
        let _isolation = crate::test_utils::scoped_tracing_subscriber_guard_blocking();
        tracing::subscriber::with_default(subscriber, || {
            let agent_chat = tracing::info_span!(
                target: "rig::agent_chat",
                "chat",
                gen_ai.provider.name = tracing::field::Empty,
                gen_ai.request.model = tracing::field::Empty,
                gen_ai.system_instructions = "effective agent instructions",
            );
            let _guard = agent_chat.enter();
            CompletionSpanBuilder::new("openai", "gpt-5", CompletionOperation::Chat).build();
        });

        let Ok(captured) = captured.0.lock() else {
            panic!("captured span lock poisoned");
        };
        let Some(span) = captured.as_ref() else {
            panic!("agent chat span was not captured");
        };
        assert!(contains_string(
            &span.initial_values,
            "gen_ai.system_instructions",
            "effective agent instructions"
        ));
        assert!(
            !span
                .recorded_values
                .iter()
                .any(|(field, _)| field == "gen_ai.system_instructions")
        );
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
