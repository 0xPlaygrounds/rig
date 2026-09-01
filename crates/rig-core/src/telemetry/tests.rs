use super::*;
use crate::completion::{AssistantContent, Message, Usage};
use serde_json::json;
use std::sync::{Arc, Mutex};
use tracing::field::{Field, Visit};
use tracing::{Id, Subscriber};
use tracing_subscriber::layer::{Context, SubscriberExt};
use tracing_subscriber::{Layer, Registry, registry::LookupSpan};

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
        Message::tool_result("call_1", "weather", "sunny"),
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

    let output = vec![AssistantContent::tool_call(
        "call_1",
        "weather",
        json!({"city": "Paris"}),
    )];
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

    let text_output = vec![AssistantContent::text("done")];
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

/// Field capture for modality spans: names paired with stringified
/// values, taken from both span creation and later `record` calls.
#[derive(Clone, Default)]
struct ModalityCapture(Arc<Mutex<Vec<(String, String)>>>);

impl ModalityCapture {
    fn get(&self, name: &str) -> Option<String> {
        self.0.lock().ok().and_then(|fields| {
            fields
                .iter()
                .rev()
                .find(|(field, _)| field == name)
                .map(|(_, value)| value.clone())
        })
    }
}

struct ModalityCaptureVisitor(ModalityCapture);

impl Visit for ModalityCaptureVisitor {
    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
        if let Ok(mut fields) = self.0.0.lock() {
            fields.push((field.name().to_string(), format!("{value:?}")));
        }
    }
}

struct ModalityCaptureLayer {
    fields: ModalityCapture,
}

impl<S> Layer<S> for ModalityCaptureLayer
where
    S: Subscriber + for<'a> LookupSpan<'a>,
{
    fn on_new_span(&self, attrs: &tracing::span::Attributes<'_>, _id: &Id, _ctx: Context<'_, S>) {
        attrs.record(&mut ModalityCaptureVisitor(self.fields.clone()));
    }

    fn on_record(&self, _span: &Id, values: &tracing::span::Record<'_>, _ctx: Context<'_, S>) {
        values.record(&mut ModalityCaptureVisitor(self.fields.clone()));
    }
}

/// `instrument_modality` opens the canonical span with the request fields
/// and records the normalized response's usage and identity on success.
#[test]
fn instrument_modality_records_usage_and_identity() {
    let fields = ModalityCapture::default();
    let subscriber = Registry::default().with(ModalityCaptureLayer {
        fields: fields.clone(),
    });
    let _isolation = crate::test_utils::scoped_tracing_subscriber_guard_blocking();
    tracing::subscriber::with_default(subscriber, || {
        let response = crate::embeddings::EmbeddingResponse::new(vec![], "probe")
            .with_model("probe-embed-v2")
            .with_response_id("emb_123")
            .with_usage(Usage {
                input_tokens: 7,
                total_tokens: 7,
                ..Usage::new()
            });
        let runtime = tokio::runtime::Builder::new_current_thread()
            .build()
            .expect("runtime");
        runtime
            .block_on(instrument_modality(
                "probe",
                "probe-embed",
                ModalityOperation::Embeddings,
                async { Ok::<_, crate::embeddings::EmbeddingError>(response) },
            ))
            .expect("call succeeds");
    });

    assert_eq!(
        fields.get("gen_ai.operation.name").as_deref(),
        Some("\"embeddings\"")
    );
    assert_eq!(
        fields.get("gen_ai.provider.name").as_deref(),
        Some("\"probe\"")
    );
    assert_eq!(
        fields.get("gen_ai.request.model").as_deref(),
        Some("\"probe-embed\"")
    );
    assert_eq!(
        fields.get("gen_ai.response.model").as_deref(),
        Some("\"probe-embed-v2\"")
    );
    assert_eq!(
        fields.get("gen_ai.response.id").as_deref(),
        Some("\"emb_123\"")
    );
    assert_eq!(
        fields.get("gen_ai.usage.input_tokens").as_deref(),
        Some("7")
    );
}

/// The provider seams are wired: an `embed_texts_response` call through
/// the shared OpenAI-compatible driver opens the embeddings span and
/// records usage — and because the vector stores' `embed_text` defaults
/// route through the same method, a `top_n` query over an
/// `InMemoryVectorIndex` records the same telemetry with no store-side
/// instrumentation.
#[test]
fn embedding_seam_and_vector_search_record_on_the_span() {
    use crate::client::EmbeddingsClient;
    use crate::embeddings::{Embedding, EmbeddingModel as _};
    use crate::vector_store::VectorStoreIndex as _;
    use crate::vector_store::in_memory_store::InMemoryVectorStore;
    use crate::vector_store::request::VectorSearchRequest;

    const BODY: &str = r#"{
            "object": "list",
            "model": "text-embedding-3-small",
            "usage": { "prompt_tokens": 4, "total_tokens": 4 },
            "data": [{ "object": "embedding", "index": 0, "embedding": [0.1, 0.2] }]
        }"#;

    let fields = ModalityCapture::default();
    let subscriber = Registry::default().with(ModalityCaptureLayer {
        fields: fields.clone(),
    });
    let _isolation = crate::test_utils::scoped_tracing_subscriber_guard_blocking();
    tracing::subscriber::with_default(subscriber, || {
        let client = crate::providers::openai::CompletionsClient::builder()
            .api_key("test-key")
            .http_client(crate::test_utils::RecordingHttpClient::new(BODY))
            .build()
            .expect("client");
        let model = client.embedding_model_with_ndims("text-embedding-3-small", 2);

        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("runtime");
        runtime.block_on(async {
            let response = model
                .embed_texts_response(["hello".to_owned()])
                .await
                .expect("embedding succeeds");
            assert_eq!(response.usage.input_tokens, 4);

            let store = InMemoryVectorStore::from_documents([(
                "doc".to_owned(),
                vec![Embedding {
                    document: "doc".to_owned(),
                    vec: vec![0.1, 0.2],
                }],
            )]);
            let index = store.index(model);
            let request = VectorSearchRequest::builder()
                .query("hello")
                .samples(1)
                .build();
            let hits: Vec<(f64, String, String)> =
                index.top_n(request).await.expect("search succeeds");
            assert_eq!(hits.len(), 1);
        });
    });

    assert_eq!(
        fields.get("gen_ai.operation.name").as_deref(),
        Some("\"embeddings\"")
    );
    assert_eq!(
        fields.get("gen_ai.provider.name").as_deref(),
        Some("\"openai\"")
    );
    assert_eq!(
        fields.get("gen_ai.usage.input_tokens").as_deref(),
        Some("4")
    );
    assert_eq!(
        fields.get("gen_ai.response.model").as_deref(),
        Some("\"text-embedding-3-small\"")
    );
    // Two embeds ran (direct + the top_n query); both hit the same seam.
    let usage_records = fields
        .0
        .lock()
        .expect("fields")
        .iter()
        .filter(|(name, _)| name == "gen_ai.usage.input_tokens")
        .count();
    assert_eq!(
        usage_records, 2,
        "the vector-search query embeds through the instrumented seam"
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

/// WARN-level events, rendered as `field=value` pairs joined with the
/// event's message, in emission order.
#[derive(Clone, Default)]
struct CapturedWarnings(Arc<Mutex<Vec<String>>>);

impl CapturedWarnings {
    fn push(&self, rendered: String) {
        if let Ok(mut events) = self.0.lock() {
            events.push(rendered);
        }
    }

    /// Drains, so a test can assert on one phase and then assert that a
    /// later phase added nothing. A cloning read would make the second
    /// assertion see the first phase's events and quietly fail.
    fn take(&self) -> Vec<String> {
        self.0
            .lock()
            .map(|mut events| std::mem::take(&mut *events))
            .unwrap_or_default()
    }
}

struct WarningCaptureLayer {
    warnings: CapturedWarnings,
}

impl<S> Layer<S> for WarningCaptureLayer
where
    S: Subscriber,
    S: for<'lookup> LookupSpan<'lookup>,
{
    fn on_event(&self, event: &tracing::Event<'_>, _ctx: Context<'_, S>) {
        if *event.metadata().level() != tracing::Level::WARN {
            return;
        }
        let mut visitor = WarningCaptureVisitor::default();
        event.record(&mut visitor);
        self.warnings.push(visitor.rendered);
    }
}

#[derive(Default)]
struct WarningCaptureVisitor {
    rendered: String,
}

impl Visit for WarningCaptureVisitor {
    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
        use std::fmt::Write;

        // `missing_fields = ?vec` arrives here; the message itself arrives
        // as the reserved `message` field. Both matter to the assertions,
        // so render every field rather than special-casing.
        let _ = write!(&mut self.rendered, " {}={value:?}", field.name());
    }

    fn record_str(&mut self, field: &Field, value: &str) {
        use std::fmt::Write;

        let _ = write!(&mut self.rendered, " {}={value}", field.name());
    }
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
    fn on_new_span(&self, attrs: &tracing::span::Attributes<'_>, _id: &Id, ctx: Context<'_, S>) {
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
        for field in COMPLETION_PARENT_REQUIRED_FIELDS {
            assert!(
                span.fields.iter().any(|candidate| candidate == field),
                "missing {field}"
            );
        }
    }
}

/// The default arm parents on the ambient span, and an explicit `parent:`
/// overrides it.
///
/// A regression to `parent: None` in the default arm would root every
/// completion-parent span, detaching it from the surrounding trace. No
/// field-set assertion in this module can see that — the fields are
/// identical either way — while an operator sees completion spans floating
/// as roots instead of nesting under the agent span.
#[test]
fn completion_parent_span_macro_honours_its_parent_argument() {
    /// `SpanCaptureLayer` has no target filter and keeps only the most
    /// recent span, so read it immediately after the span under test is
    /// created, and confirm the target before trusting the parent.
    fn captured_parent(captured: &CapturedSpan) -> Option<String> {
        let Ok(captured) = captured.0.lock() else {
            panic!("captured span lock poisoned");
        };
        let Some(span) = captured.as_ref() else {
            panic!("completion-parent span was not captured");
        };
        assert_eq!(span.target, "third_party_runtime");
        span.parent_name.clone()
    }

    let captured = CapturedSpan::default();
    let subscriber = Registry::default().with(SpanCaptureLayer {
        span: captured.clone(),
    });
    let _isolation = crate::test_utils::scoped_tracing_subscriber_guard_blocking();
    tracing::subscriber::with_default(subscriber, || {
        let ambient = tracing::info_span!(target: "application", "ambient");

        // Default arm: nests under whatever span is current.
        ambient.in_scope(|| {
            let _default_arm = completion_parent_span!(
                target: "third_party_runtime",
                name: "chat",
                operation: Empty,
                system_instructions: Option::<&str>::None,
            );
        });
        assert_eq!(captured_parent(&captured).as_deref(), Some("ambient"));

        // Explicit arm: the caller's parent wins over the ambient span, so
        // this one is a root despite `ambient` being current.
        ambient.in_scope(|| {
            let _explicit_arm = completion_parent_span!(
                target: "third_party_runtime",
                parent: None,
                name: "chat",
                operation: Empty,
                system_instructions: Option::<&str>::None,
            );
        });
        assert_eq!(captured_parent(&captured), None);
    });
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
        let span = CompletionSpanBuilder::new("openai", "gpt-5", CompletionOperation::Chat).build();
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
fn marker_span_missing_required_fields_is_not_adopted() {
    // A span that carries the marker but omits required canonical fields
    // (here: everything past `gen_ai.request.model`) must NOT be adopted.
    // Adopting it would silently drop the response/usage/content telemetry
    // that `Span::record` no-ops on for undeclared fields. Instead the
    // builder creates a fresh `rig::completions` child so nothing is lost.
    let captured = CapturedSpan::default();
    let subscriber = Registry::default().with(SpanCaptureLayer {
        span: captured.clone(),
    });
    let _isolation = crate::test_utils::scoped_tracing_subscriber_guard_blocking();
    tracing::subscriber::with_default(subscriber, || {
        // Deliberately hand-written (not `completion_parent_span!`): the
        // point is a marker span that fails to declare required fields.
        let partial_marker = tracing::info_span!(
            target: "third_party_runtime",
            "chat",
            rig.completion_parent = true,
            gen_ai.operation.name = tracing::field::Empty,
            gen_ai.provider.name = tracing::field::Empty,
            gen_ai.request.model = tracing::field::Empty,
        );
        // Premise check: the hand-written marker literal above must still
        // match the constant — if the marker is ever renamed this fails
        // first, pointing at the stale literal, so the test cannot keep
        // passing for the wrong reason (no marker at all, rather than the
        // marker with fields missing).
        let Some(metadata) = partial_marker.metadata() else {
            panic!("partial marker span was disabled");
        };
        assert!(
            metadata
                .fields()
                .field(COMPLETION_PARENT_MARKER_FIELD)
                .is_some(),
            "hand-written marker literal is stale; update it to {COMPLETION_PARENT_MARKER_FIELD}"
        );
        let _guard = partial_marker.enter();
        let span = CompletionSpanBuilder::new("openai", "gpt-5", CompletionOperation::Chat).build();
        assert_ne!(span.id(), partial_marker.id());
    });

    let Ok(captured) = captured.0.lock() else {
        panic!("captured span lock poisoned");
    };
    let Some(span) = captured.as_ref() else {
        panic!("completion span was not captured");
    };
    // A canonical child span is created and parented under the marker span,
    // and it carries the completion fields the marker span could not absorb.
    assert_eq!(span.target, "rig::completions");
    assert_eq!(span.parent_name.as_deref(), Some("chat"));
    for (field, value) in [
        ("gen_ai.operation.name", "chat"),
        ("gen_ai.provider.name", "openai"),
        ("gen_ai.request.model", "gpt-5"),
    ] {
        assert!(contains_string(&span.initial_values, field, value));
    }
}

#[test]
fn completion_parent_span_macro_matches_the_contract_exactly() {
    use std::collections::HashSet;

    let _isolation = crate::test_utils::scoped_tracing_subscriber_guard_blocking();
    tracing::subscriber::with_default(Registry::default(), || {
        let span = completion_parent_span!(
            target: "contract_test",
            name: "chat",
            operation: "chat",
            system_instructions: Option::<&str>::None,
        );
        let Some(metadata) = span.metadata() else {
            panic!("contract span was disabled");
        };
        let declared: HashSet<&str> = metadata.fields().iter().map(|field| field.name()).collect();
        let expected: HashSet<&str> = COMPLETION_PARENT_REQUIRED_FIELDS
            .iter()
            .copied()
            .chain([COMPLETION_PARENT_MARKER_FIELD])
            .collect();
        // Exact equality in both directions: a field added to the macro
        // but not the constant (or vice versa) is drift, not a superset.
        assert_eq!(declared, expected);
        // Duplicate field names collapse in a `HashSet`, so also pin the
        // count: set equality alone cannot catch a field declared twice.
        assert_eq!(metadata.fields().len(), expected.len());
        assert_eq!(
            classify_completion_parent(metadata),
            CompletionParentVerdict::Adopt
        );

        // The explicit-`parent:` arm declares the identical field set.
        let span = completion_parent_span!(
            target: "contract_test",
            parent: None,
            name: "chat",
            operation: "chat",
            system_instructions: Option::<&str>::None,
        );
        let Some(metadata) = span.metadata() else {
            panic!("contract span with explicit parent was disabled");
        };
        let declared: HashSet<&str> = metadata.fields().iter().map(|field| field.name()).collect();
        assert_eq!(declared, expected);
        assert_eq!(metadata.fields().len(), expected.len());
        assert_eq!(
            classify_completion_parent(metadata),
            CompletionParentVerdict::Adopt
        );

        // Runtime-specific extra fields are additive on top of the contract.
        let span = completion_parent_span!(
            target: "contract_test",
            name: "chat",
            operation: "chat",
            system_instructions: Option::<&str>::None,
            gen_ai.agent.name = "assistant",
        );
        let Some(metadata) = span.metadata() else {
            panic!("contract span with extras was disabled");
        };
        let declared: HashSet<&str> = metadata.fields().iter().map(|field| field.name()).collect();
        let expected: HashSet<&str> = expected.into_iter().chain(["gen_ai.agent.name"]).collect();
        assert_eq!(declared, expected);
        assert_eq!(metadata.fields().len(), expected.len());
        assert_eq!(
            classify_completion_parent(metadata),
            CompletionParentVerdict::Adopt
        );
    });
}

#[test]
fn canonical_completion_span_declares_exactly_the_required_fields() {
    use std::collections::HashSet;

    let _isolation = crate::test_utils::scoped_tracing_subscriber_guard_blocking();
    tracing::subscriber::with_default(Registry::default(), || {
        let span = CompletionSpanBuilder::new("openai", "gpt-5", CompletionOperation::Chat).build();
        let Some(metadata) = span.metadata() else {
            panic!("completion span was disabled");
        };
        let declared: HashSet<&str> = metadata.fields().iter().map(|field| field.name()).collect();
        let expected: HashSet<&str> = COMPLETION_PARENT_REQUIRED_FIELDS.iter().copied().collect();
        // The adoption checklist and the span the builder itself creates
        // must be the same set, or an adopted parent could not absorb
        // every field the builder records.
        assert_eq!(declared, expected);
        // Duplicate field names collapse in a `HashSet`, so also pin the
        // count: set equality alone cannot catch a field declared twice.
        assert_eq!(metadata.fields().len(), expected.len());
    });
}

#[test]
fn completion_parent_required_fields_are_pinned() {
    // Changing this list is a contract change. An adopted parent declares
    // its fields statically, so a runtime whose span was hand-written
    // against the old list stops being adopted once the list moves — it
    // degrades gracefully (fresh child span, one-time warning naming what
    // is missing), but it does degrade. Confirm that is intended, note it
    // in the CHANGELOG, then update this snapshot.
    //
    // This is the only test that notices. Every other contract test
    // compares the three forms of the contract to each other, so a
    // *coherent* change — a field added to both this constant and the
    // macro — leaves them all agreeing, and green.
    assert_eq!(COMPLETION_PARENT_MARKER_FIELD, "rig.completion_parent");
    assert_eq!(
        COMPLETION_PARENT_REQUIRED_FIELDS,
        &[
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
        ]
    );
}

/// Every row of the adoption decision table, asserted against the pure
/// classifier so no global warn-once state is involved.
#[test]
fn classify_completion_parent_covers_the_decision_table() {
    let _isolation = crate::test_utils::scoped_tracing_subscriber_guard_blocking();
    tracing::subscriber::with_default(Registry::default(), || {
        let verdict = |span: &tracing::Span| {
            let Some(metadata) = span.metadata() else {
                panic!("classifier fixture span was disabled");
            };
            classify_completion_parent(metadata)
        };

        // Marker + full contract.
        let conforming = completion_parent_span!(
            target: "classifier_test",
            name: "chat",
            operation: tracing::field::Empty,
            system_instructions: tracing::field::Empty,
        );
        assert_eq!(verdict(&conforming), CompletionParentVerdict::Adopt);

        // Marker, incomplete contract.
        let partial = tracing::info_span!(
            target: "classifier_test",
            "chat",
            rig.completion_parent = true,
            gen_ai.operation.name = tracing::field::Empty,
        );
        assert_eq!(
            verdict(&partial),
            CompletionParentVerdict::RejectMissingFields
        );
        let Some(partial_metadata) = partial.metadata() else {
            panic!("classifier fixture span was disabled");
        };
        // The names only get computed on the warning path, so pin them here.
        assert_eq!(
            missing_required_fields(partial_metadata),
            COMPLETION_PARENT_REQUIRED_FIELDS
                .iter()
                .copied()
                .filter(|name| *name != "gen_ai.operation.name")
                .collect::<Vec<_>>()
        );

        // An ordinary ambient span.
        let ambient = tracing::info_span!(target: "application", "ambient");
        assert_eq!(verdict(&ambient), CompletionParentVerdict::NotAParent);

        // Marker detection is an exact field-name match, never a prefix: a
        // runtime field that merely starts with the marker name must not
        // make its span a rejected parent and warn at a runtime that never
        // opted in.
        let lookalike = tracing::info_span!(
            target: "application",
            "ambient",
            rig.completion_parent.id = "abc",
            rig.completion_parent_id = "abc",
        );
        assert_eq!(verdict(&lookalike), CompletionParentVerdict::NotAParent);
    });
}

/// The near-miss diagnostic is the only thing that makes a rejected parent
/// visible to an operator — otherwise the sole symptom is a duplicated span
/// layer in dashboards — so its message, its `missing_fields` payload, and
/// its once-per-callsite budget all need pinning.
#[test]
fn near_miss_completion_parent_warns_once_per_callsite() {
    let warnings = CapturedWarnings::default();
    let subscriber = Registry::default().with(WarningCaptureLayer {
        warnings: warnings.clone(),
    });
    let _isolation = crate::test_utils::scoped_tracing_subscriber_guard_blocking();
    // The warn budget is process-global; claim a clean one rather than
    // relying on this test's fixture span owning a callsite no other test
    // touches.
    reset_near_miss_warnings();
    tracing::subscriber::with_default(subscriber, || {
        // The `warn!` callsite lives in `warn_once_on_completion_parent_verdict`
        // and is shared with every other near-miss test, so its interest may
        // already be cached as `never` from a run under a different
        // subscriber. Same hazard `test_utils::scoped_tracing_subscriber_guard`
        // documents; same fix used in
        // `agent::prompt_request::streaming`'s scoped-subscriber tests.
        tracing::callsite::rebuild_interest_cache();

        let near_miss = tracing::info_span!(
            target: "third_party_runtime",
            "chat",
            rig.completion_parent = true,
            gen_ai.operation.name = tracing::field::Empty,
        );
        let _guard = near_miss.enter();
        CompletionSpanBuilder::new("openai", "gpt-5", CompletionOperation::Chat).build();
        // Second completion under the *same* span callsite: the budget is
        // per callsite, so this one must stay silent.
        CompletionSpanBuilder::new("openai", "gpt-5", CompletionOperation::Chat).build();
    });

    let captured = warnings.take();
    assert_eq!(
        captured.len(),
        1,
        "a near-miss callsite warns exactly once, got: {captured:?}"
    );
    let Some(message) = captured.first() else {
        panic!("near miss did not warn");
    };
    assert!(
        message.contains("gen_ai.provider.name"),
        "warning must name the missing fields, got: {message}"
    );
    assert!(
        message.contains("completion_parent_span!"),
        "warning must point at the supported fix, got: {message}"
    );
}

/// The property that justifies keying the budget on the callsite rather
/// than a single process-wide flag: two runtimes each declaring a broken
/// parent are both reported. A global flag would report whichever ran first
/// and stay silent about the other — and every other test in this module
/// passes under that behaviour, so this is the only one that pins it.
#[test]
fn distinct_near_miss_callsites_each_warn() {
    let warnings = CapturedWarnings::default();
    let subscriber = Registry::default().with(WarningCaptureLayer {
        warnings: warnings.clone(),
    });
    let _isolation = crate::test_utils::scoped_tracing_subscriber_guard_blocking();
    reset_near_miss_warnings();
    tracing::subscriber::with_default(subscriber, || {
        tracing::callsite::rebuild_interest_cache();

        // Two separate `info_span!` invocations, and they must stay
        // separate: the callsite *is* the dedup key, so extracting these
        // into a shared helper collapses them into one and this test would
        // assert 1, not 2. `reset_near_miss_warnings` cannot protect this
        // test the way it protects the others — distinct callsites are the
        // thing under test.
        let first = tracing::info_span!(
            target: "runtime_a",
            "chat",
            rig.completion_parent = true,
            gen_ai.operation.name = tracing::field::Empty,
        );
        first.in_scope(|| {
            CompletionSpanBuilder::new("openai", "gpt-5", CompletionOperation::Chat).build();
        });

        let second = tracing::info_span!(
            target: "runtime_b",
            "chat",
            rig.completion_parent = true,
            gen_ai.operation.name = tracing::field::Empty,
        );
        second.in_scope(|| {
            CompletionSpanBuilder::new("openai", "gpt-5", CompletionOperation::Chat).build();
        });
    });

    let captured = warnings.take();
    assert_eq!(
        captured.len(),
        2,
        "each offending callsite warns once, got: {captured:?}"
    );
}

/// The happy path must stay quiet: a conforming parent is adopted silently,
/// so the diagnostic above cannot become background noise on every
/// completion.
#[test]
fn conforming_completion_parent_never_warns() {
    let warnings = CapturedWarnings::default();
    let subscriber = Registry::default().with(WarningCaptureLayer {
        warnings: warnings.clone(),
    });
    let _isolation = crate::test_utils::scoped_tracing_subscriber_guard_blocking();
    // Claim a clean budget: the control below must be able to warn even if
    // another test already reported this fixture's callsite.
    reset_near_miss_warnings();
    tracing::subscriber::with_default(subscriber, || {
        tracing::callsite::rebuild_interest_cache();

        // Control. Asserting an absence proves nothing unless the warning
        // pipe is known live in *this* subscriber: without this, the
        // assertions below pass just as happily when the diagnostic has
        // been deleted, or when callsite interest is stuck at `never`.
        let near_miss = tracing::info_span!(
            target: "third_party_runtime",
            "chat",
            rig.completion_parent = true,
            gen_ai.operation.name = tracing::field::Empty,
        );
        near_miss.in_scope(|| {
            CompletionSpanBuilder::new("openai", "gpt-5", CompletionOperation::Chat).build();
        });
        assert_eq!(
            warnings.take().len(),
            1,
            "control: a near miss must warn, or this test cannot detect silence"
        );

        let conforming = completion_parent_span!(
            target: "third_party_runtime",
            name: "chat",
            operation: Empty,
            system_instructions: Option::<&str>::None,
        );
        let _guard = conforming.enter();
        CompletionSpanBuilder::new("openai", "gpt-5", CompletionOperation::Chat).build();

        // An ordinary ambient span is not a parent at all, and must not warn
        // either — a runtime that never opted in should never hear about
        // this contract.
        drop(_guard);
        let ambient = tracing::info_span!(target: "application", "ambient");
        let _ambient_guard = ambient.enter();
        CompletionSpanBuilder::new("openai", "gpt-5", CompletionOperation::Chat).build();
    });

    // The control drained the buffer, so anything here was emitted by the
    // conforming or ambient span.
    let captured = warnings.take();
    assert!(
        captured.is_empty(),
        "adoption and non-participation are both silent, got: {captured:?}"
    );
}

#[test]
fn agent_chat_span_is_adopted_and_enriched() {
    let captured = CapturedSpan::default();
    let subscriber = Registry::default().with(SpanCaptureLayer {
        span: captured.clone(),
    });
    let _isolation = crate::test_utils::scoped_tracing_subscriber_guard_blocking();
    tracing::subscriber::with_default(subscriber, || {
        let completion_parent = completion_parent_span!(
            target: "rig::agent_chat",
            name: "chat_streaming",
            operation: tracing::field::Empty,
            system_instructions: tracing::field::Empty,
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
    assert_eq!(span.target, "rig::agent_chat");
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
fn neutral_completion_parent_span_is_adopted_and_enriched() {
    let captured = CapturedSpan::default();
    let subscriber = Registry::default().with(SpanCaptureLayer {
        span: captured.clone(),
    });
    let _isolation = crate::test_utils::scoped_tracing_subscriber_guard_blocking();
    tracing::subscriber::with_default(subscriber, || {
        let completion_parent = completion_parent_span!(
            target: "test_runtime",
            name: "chat",
            operation: tracing::field::Empty,
            system_instructions: tracing::field::Empty,
        );
        let _guard = completion_parent.enter();
        let span = CompletionSpanBuilder::new(
            "neutral-provider",
            "neutral-model",
            CompletionOperation::Chat,
        )
        .build();
        assert_eq!(span.id(), completion_parent.id());
    });

    let Ok(captured) = captured.0.lock() else {
        panic!("captured span lock poisoned");
    };
    let Some(span) = captured.as_ref() else {
        panic!("neutral completion-parent span was not captured");
    };
    assert_eq!(span.target, "test_runtime");
    for (field, value) in [
        ("gen_ai.operation.name", "chat"),
        ("gen_ai.provider.name", "neutral-provider"),
        ("gen_ai.request.model", "neutral-model"),
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
        let completion_parent = completion_parent_span!(
            target: "test_runtime",
            name: "chat",
            operation: tracing::field::Empty,
            system_instructions: "effective agent instructions",
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
    let usage = Usage {
        input_tokens: 1,
        output_tokens: 2,
        total_tokens: 15,
        cached_input_tokens: 3,
        cache_creation_input_tokens: 4,
        tool_use_prompt_tokens: 12,
        reasoning_tokens: 5,
    };

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
