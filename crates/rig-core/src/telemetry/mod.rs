//! This module primarily concerns being able to orchestrate telemetry across a given pipeline or workflow.
//! This includes tracing, being able to send traces to an OpenTelemetry collector, setting up your
//! agents with the correct tracing style so you can emit the right traces for platforms like Langfuse,
//! and more.

use crate::OneOrMany;
use crate::completion::{AssistantContent, GetTokenUsage, Message};
use crate::message::{
    DocumentSourceKind, Image, MimeType, Reasoning, ReasoningContent, ToolResult,
    ToolResultContent, UserContent,
};
use base64::Engine;
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
    /// A chat completion.
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

/// Core-owned marker target for a runtime span that may absorb provider completion fields.
pub const COMPLETION_PARENT_SPAN_TARGET: &str = "rig::completion_parent";

/// Builder for a canonical GenAI completion span.
///
/// Runtime spans using [`COMPLETION_PARENT_SPAN_TARGET`] are enriched and reused
/// so one model turn has exactly one model span. Other ambient spans remain
/// parents of a newly created `rig::completions` span.
pub struct CompletionSpanBuilder<'a> {
    provider: &'a str,
    request_model: &'a str,
    operation: CompletionOperation,
    system_instructions: Option<String>,
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

    /// Set the system instructions sent with the request when sensitive content
    /// telemetry has been explicitly enabled.
    pub fn system_instructions(
        mut self,
        system_instructions: Option<&'a str>,
        record_content: bool,
    ) -> Self {
        self.system_instructions = system_instructions_json(system_instructions, record_content);
        self
    }

    /// Build a canonical completion span or enrich Rig's current completion-parent span.
    pub fn build(self) -> tracing::Span {
        let current = tracing::Span::current();
        if current
            .metadata()
            .is_some_and(|metadata| metadata.target() == COMPLETION_PARENT_SPAN_TARGET)
        {
            current.record("gen_ai.operation.name", self.operation.as_str());
            current.record("gen_ai.provider.name", self.provider);
            current.record("gen_ai.request.model", self.request_model);
            if let Some(system_instructions) = self.system_instructions.as_deref() {
                current.record("gen_ai.system_instructions", system_instructions);
            }
            return current;
        }

        let operation = self.operation.as_str();
        let system_instructions = self.system_instructions.as_deref();
        match self.operation {
            CompletionOperation::Chat => new_completion_span!(
                "chat",
                self.provider,
                self.request_model,
                operation,
                system_instructions
            ),
            CompletionOperation::ChatStreaming => new_completion_span!(
                "chat_streaming",
                self.provider,
                self.request_model,
                operation,
                system_instructions
            ),
            CompletionOperation::GenerateContent => new_completion_span!(
                "generate_content",
                self.provider,
                self.request_model,
                operation,
                system_instructions
            ),
            CompletionOperation::Interactions => new_completion_span!(
                "interactions",
                self.provider,
                self.request_model,
                operation,
                system_instructions
            ),
            CompletionOperation::InteractionsStreaming => new_completion_span!(
                "interactions_streaming",
                self.provider,
                self.request_model,
                operation,
                system_instructions
            ),
        }
    }
}

#[derive(Serialize)]
struct TelemetryChatMessage {
    role: &'static str,
    parts: Vec<TelemetryPart>,
}

#[derive(Serialize)]
struct TelemetryOutputMessage {
    role: &'static str,
    parts: Vec<TelemetryPart>,
    finish_reason: &'static str,
}

#[derive(Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum TelemetryPart {
    Text {
        content: String,
    },
    ToolCall {
        #[serde(skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        name: String,
        arguments: serde_json::Value,
    },
    ToolCallResponse {
        #[serde(skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        response: serde_json::Value,
    },
    Reasoning {
        content: String,
    },
    Uri {
        #[serde(skip_serializing_if = "Option::is_none")]
        mime_type: Option<String>,
        modality: &'static str,
        uri: String,
    },
    File {
        #[serde(skip_serializing_if = "Option::is_none")]
        mime_type: Option<String>,
        modality: &'static str,
        file_id: String,
    },
    Blob {
        #[serde(skip_serializing_if = "Option::is_none")]
        mime_type: Option<String>,
        modality: &'static str,
        content: String,
    },
}

fn media_part<T>(
    data: &DocumentSourceKind,
    media_type: Option<&T>,
    modality: &'static str,
) -> Option<TelemetryPart>
where
    T: MimeType,
{
    let mime_type = media_type.map(|media_type| media_type.to_mime_type().to_string());
    match data {
        DocumentSourceKind::Url(uri) => Some(TelemetryPart::Uri {
            mime_type,
            modality,
            uri: uri.clone(),
        }),
        DocumentSourceKind::FileId(file_id) => Some(TelemetryPart::File {
            mime_type,
            modality,
            file_id: file_id.clone(),
        }),
        DocumentSourceKind::Base64(content) => Some(TelemetryPart::Blob {
            mime_type,
            modality,
            content: content.clone(),
        }),
        DocumentSourceKind::Raw(content) => Some(TelemetryPart::Blob {
            mime_type,
            modality,
            content: base64::engine::general_purpose::STANDARD.encode(content),
        }),
        DocumentSourceKind::String(content) => Some(TelemetryPart::Text {
            content: content.clone(),
        }),
        DocumentSourceKind::Unknown => None,
    }
}

fn image_part(image: &Image) -> Option<TelemetryPart> {
    media_part(&image.data, image.media_type.as_ref(), "image")
}

fn reasoning_parts(reasoning: &Reasoning) -> Vec<TelemetryPart> {
    reasoning
        .content
        .iter()
        .map(|content| {
            let content = match content {
                ReasoningContent::Text { text, .. } | ReasoningContent::Summary(text) => text,
                ReasoningContent::Encrypted(content) => content,
                ReasoningContent::Redacted { data } => data,
            };
            TelemetryPart::Reasoning {
                content: content.clone(),
            }
        })
        .collect()
}

fn tool_result_response(result: &ToolResult) -> serde_json::Value {
    let mut content = result
        .content
        .iter()
        .filter_map(|content| match content {
            ToolResultContent::Text(text) => Some(serde_json::Value::String(text.text.clone())),
            ToolResultContent::Json { value } => Some(value.clone()),
            ToolResultContent::Image(image) => {
                image_part(image).and_then(|part| serde_json::to_value(part).ok())
            }
        })
        .collect::<Vec<_>>();

    if content.len() == 1 {
        content.pop().unwrap_or(serde_json::Value::Null)
    } else {
        serde_json::Value::Array(content)
    }
}

fn user_parts(content: &OneOrMany<UserContent>) -> Vec<TelemetryPart> {
    content
        .iter()
        .filter_map(|content| match content {
            UserContent::Text(text) => Some(TelemetryPart::Text {
                content: text.text.clone(),
            }),
            UserContent::ToolResult(result) => Some(TelemetryPart::ToolCallResponse {
                id: Some(result.id.clone()),
                response: tool_result_response(result),
            }),
            UserContent::Image(image) => image_part(image),
            UserContent::Audio(audio) => {
                media_part(&audio.data, audio.media_type.as_ref(), "audio")
            }
            UserContent::Video(video) => {
                media_part(&video.data, video.media_type.as_ref(), "video")
            }
            UserContent::Document(document) => {
                media_part(&document.data, document.media_type.as_ref(), "document")
            }
        })
        .collect()
}

fn assistant_parts(content: &OneOrMany<AssistantContent>) -> Vec<TelemetryPart> {
    content
        .iter()
        .flat_map(|content| match content {
            AssistantContent::Text(text) => vec![TelemetryPart::Text {
                content: text.text.clone(),
            }],
            AssistantContent::ToolCall(tool_call) => vec![TelemetryPart::ToolCall {
                id: Some(tool_call.id.clone()),
                name: tool_call.function.name.clone(),
                arguments: tool_call.function.arguments.clone(),
            }],
            AssistantContent::Reasoning(reasoning) => reasoning_parts(reasoning),
            AssistantContent::Image(image) => image_part(image).into_iter().collect(),
        })
        .collect()
}

fn input_messages(messages: &[Message]) -> Vec<TelemetryChatMessage> {
    messages
        .iter()
        .map(|message| match message {
            Message::System { content } => TelemetryChatMessage {
                role: "system",
                parts: vec![TelemetryPart::Text {
                    content: content.clone(),
                }],
            },
            Message::User { content } => TelemetryChatMessage {
                role: "user",
                parts: user_parts(content),
            },
            Message::Assistant { content, .. } => TelemetryChatMessage {
                role: "assistant",
                parts: assistant_parts(content),
            },
        })
        .collect()
}

fn output_messages(content: &OneOrMany<AssistantContent>) -> Vec<TelemetryOutputMessage> {
    let finish_reason = if content
        .iter()
        .any(|content| matches!(content, AssistantContent::ToolCall(_)))
    {
        "tool_call"
    } else {
        // Rig's normalized assistant content does not retain provider finish
        // reasons such as length or content filtering. Avoid claiming a clean
        // stop when the actual reason is unavailable.
        "unknown"
    };
    vec![TelemetryOutputMessage {
        role: "assistant",
        parts: assistant_parts(content),
        finish_reason,
    }]
}

/// Serializes system instructions using the normalized GenAI telemetry shape.
pub fn system_instructions_json(instructions: Option<&str>, enabled: bool) -> Option<String> {
    if !enabled {
        return None;
    }

    instructions.and_then(|instructions| {
        serde_json::to_string(&vec![TelemetryPart::Text {
            content: instructions.to_string(),
        }])
        .ok()
    })
}

/// Records serialized model input messages on `gen_ai.input.messages` when
/// content telemetry is explicitly enabled.
///
/// Message content can contain prompts, retrieved context, tool results, and
/// other sensitive or high-cardinality data. Keep this disabled unless the
/// caller has explicitly opted in for debugging/observability.
pub fn record_model_input(span: &tracing::Span, messages: &[Message], enabled: bool) {
    if !enabled || span.is_disabled() {
        return;
    }

    if let Ok(messages) = serde_json::to_string(&input_messages(messages)) {
        span.record("gen_ai.input.messages", messages);
    }
}

/// Records serialized model output messages on `gen_ai.output.messages` when
/// content telemetry is explicitly enabled.
///
/// Message content can contain model responses, tool calls, and other sensitive
/// or high-cardinality data. Keep this disabled unless the caller has explicitly
/// opted in for debugging/observability.
pub fn record_model_output(
    span: &tracing::Span,
    content: &OneOrMany<AssistantContent>,
    enabled: bool,
) {
    if !enabled || span.is_disabled() {
        return;
    }

    let messages = output_messages(content);
    if let Ok(messages) = serde_json::to_string(&messages) {
        span.record("gen_ai.output.messages", messages);
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
    use crate::completion::{AssistantContent, GetTokenUsage, Message, Usage};
    use serde_json::json;
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

    #[test]
    fn content_attributes_follow_gen_ai_semantic_convention_json_shapes() {
        assert_eq!(
            system_instructions_json(Some("follow policy"), true).as_deref(),
            Some(r#"[{"type":"text","content":"follow policy"}]"#)
        );
        assert_eq!(system_instructions_json(Some("secret"), false), None);

        let input = input_messages(&[
            Message::system("follow policy"),
            Message::user("hello"),
            Message::tool_result("call_1", "sunny"),
        ]);
        assert_eq!(
            serde_json::to_value(input).expect("semantic-convention input DTOs serialize"),
            json!([
                {
                    "role": "system",
                    "parts": [{"type": "text", "content": "follow policy"}]
                },
                {
                    "role": "user",
                    "parts": [{"type": "text", "content": "hello"}]
                },
                {
                    "role": "user",
                    "parts": [{
                        "type": "tool_call_response",
                        "id": "call_1",
                        "response": "sunny"
                    }]
                }
            ])
        );

        let output = OneOrMany::one(AssistantContent::tool_call(
            "call_1",
            "weather",
            json!({"city": "Paris"}),
        ));
        assert_eq!(
            serde_json::to_value(output_messages(&output))
                .expect("semantic-convention output DTOs serialize"),
            json!([{
                "role": "assistant",
                "parts": [{
                    "type": "tool_call",
                    "id": "call_1",
                    "name": "weather",
                    "arguments": {"city": "Paris"}
                }],
                "finish_reason": "tool_call"
            }])
        );

        let text_output = OneOrMany::one(AssistantContent::text("done"));
        assert_eq!(
            serde_json::to_value(output_messages(&text_output))
                .expect("semantic-convention text output DTOs serialize"),
            json!([{
                "role": "assistant",
                "parts": [{"type": "text", "content": "done"}],
                "finish_reason": "unknown"
            }])
        );
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
                    .system_instructions(Some("system prompt"), true)
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
                (
                    "gen_ai.system_instructions",
                    r#"[{"type":"text","content":"system prompt"}]"#,
                ),
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
    fn completion_parent_span_is_adopted_and_enriched() {
        let captured = CapturedSpan::default();
        let subscriber = Registry::default().with(SpanCaptureLayer {
            span: captured.clone(),
        });
        let _isolation = crate::test_utils::scoped_tracing_subscriber_guard_blocking();
        tracing::subscriber::with_default(subscriber, || {
            let completion_parent = tracing::info_span!(
                target: "rig::completion_parent",
                "chat_streaming",
                gen_ai.operation.name = tracing::field::Empty,
                gen_ai.provider.name = tracing::field::Empty,
                gen_ai.request.model = tracing::field::Empty,
                gen_ai.system_instructions = tracing::field::Empty,
            );
            let _guard = completion_parent.enter();
            let span = CompletionSpanBuilder::new(
                "anthropic",
                "claude-sonnet",
                CompletionOperation::ChatStreaming,
            )
            .system_instructions(Some("provider system"), true)
            .build();
            assert_eq!(span.id(), completion_parent.id());
        });

        let Ok(captured) = captured.0.lock() else {
            panic!("captured span lock poisoned");
        };
        let Some(span) = captured.as_ref() else {
            panic!("completion-parent span was not captured");
        };
        for (field, value) in [
            ("gen_ai.operation.name", "chat_streaming"),
            ("gen_ai.provider.name", "anthropic"),
            ("gen_ai.request.model", "claude-sonnet"),
            (
                "gen_ai.system_instructions",
                r#"[{"type":"text","content":"provider system"}]"#,
            ),
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
            let completion_parent = tracing::info_span!(
                target: "rig::completion_parent",
                "chat",
                gen_ai.provider.name = tracing::field::Empty,
                gen_ai.request.model = tracing::field::Empty,
                gen_ai.system_instructions = "effective agent instructions",
            );
            let _guard = completion_parent.enter();
            CompletionSpanBuilder::new("openai", "gpt-5", CompletionOperation::Chat).build();
        });

        let Ok(captured) = captured.0.lock() else {
            panic!("captured span lock poisoned");
        };
        let Some(span) = captured.as_ref() else {
            panic!("completion-parent span was not captured");
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
