use super::*;
use crate::test_utils::TraceCapture;
use serde_json::json;

/// The provider seams are wired: an `embed_texts_response` call through
/// the shared OpenAI-compatible driver opens the embeddings span and
/// records usage — and because the vector stores' `embed_text` defaults
/// route through the same method, a `top_n` query over an
/// `InMemoryVectorIndex` records the same telemetry with no store-side
/// instrumentation.
#[test]
fn embedding_seam_and_vector_search_record_on_the_span() {
    use crate::embeddings::Embedding;
    use crate::vector_store::VectorStoreIndex as _;
    use crate::vector_store::in_memory_store::InMemoryVectorStore;
    use crate::vector_store::request::VectorSearchRequest;

    const BODY: &str = r#"{
            "object": "list",
            "model": "text-embedding-3-small",
            "usage": { "prompt_tokens": 4, "total_tokens": 4 },
            "data": [{ "object": "embedding", "index": 0, "embedding": [0.1, 0.2] }]
        }"#;

    let capture = TraceCapture::default();
    let _isolation = crate::test_utils::scoped_tracing_subscriber_guard_blocking();
    tracing::subscriber::with_default(capture.subscriber(), || {
        let model = crate::driver::Model::new(
            crate::providers::openai::wire::OpenAIConfig::with_key(
                &crate::providers::openai::wire::OPENAI,
                "test-key",
            )
            .embedding("text-embedding-3-small", Some(2)),
            crate::test_utils::RecordingHttpClient::new(BODY),
        );

        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("runtime");
        runtime.block_on(async {
            let response = model
                .call(vec!["hello".to_owned()])
                .await
                .expect("embedding succeeds");
            assert_eq!(response.usage.input_tokens, Some(4));

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
            let hits: Vec<crate::vector_store::VectorSearchResult<String>> =
                index.top_n(request).await.expect("search succeeds");
            assert_eq!(hits.len(), 1);
        });
    });

    let last = |field: &str| capture.values_of(field).last().cloned();
    assert_eq!(last("gen_ai.operation.name"), Some(json!("embeddings")));
    assert_eq!(last("gen_ai.provider.name"), Some(json!("openai")));
    assert_eq!(last("gen_ai.usage.input_tokens"), Some(json!(4)));
    assert_eq!(
        last("gen_ai.response.model"),
        Some(json!("text-embedding-3-small"))
    );
    // Two embeds ran (direct + the top_n query); both hit the same seam.
    let usage_records = capture.values_of("gen_ai.usage.input_tokens").len();
    assert_eq!(
        usage_records, 2,
        "the vector-search query embeds through the instrumented seam"
    );
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
    /// The capture has no target filter, so read its last span immediately
    /// after the span under test is created, and confirm the target before
    /// trusting the parent.
    fn captured_parent(capture: &TraceCapture) -> Option<&'static str> {
        let Some(span) = capture.last_span() else {
            panic!("completion-parent span was not captured");
        };
        assert_eq!(span.target, "third_party_runtime");
        span.parent_name
    }

    let capture = TraceCapture::default();
    let _isolation = crate::test_utils::scoped_tracing_subscriber_guard_blocking();
    tracing::subscriber::with_default(capture.subscriber(), || {
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
        assert_eq!(captured_parent(&capture), Some("ambient"));

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
        assert_eq!(captured_parent(&capture), None);
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

/// The near-miss diagnostic is the only thing that makes a rejected parent
/// visible to an operator — otherwise the sole symptom is a duplicated span
/// layer in dashboards — so its message, its `missing_fields` payload, and
/// its once-per-callsite budget all need pinning.
#[test]
fn near_miss_completion_parent_warns_once_per_callsite() {
    let capture = TraceCapture::default();
    let _isolation = crate::test_utils::scoped_tracing_subscriber_guard_blocking();
    // The warn budget is process-global; claim a clean one rather than
    // relying on this test's fixture span owning a callsite no other test
    // touches.
    reset_near_miss_warnings();
    tracing::subscriber::with_default(capture.subscriber(), || {
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
        SpanBuilder::new("openai", "gpt-5", GenAiOperation::Chat).build();
        // Second completion under the *same* span callsite: the budget is
        // per callsite, so this one must stay silent.
        SpanBuilder::new("openai", "gpt-5", GenAiOperation::Chat).build();
    });

    let captured = capture.warnings();
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

/// A runtime's completion parent (an agent's chat span) is adopted by the
/// provider's completion span, so the delivery mode reaches it the same way
/// the operation does. A runtime still naming the deprecated streaming
/// operation reports the standard `chat` operation and the stream flag.
#[test]
#[allow(deprecated)]
fn adopted_completion_parents_report_the_standard_operation_and_stream_flag() {
    let capture = TraceCapture::default();
    let _isolation = crate::test_utils::scoped_tracing_subscriber_guard_blocking();
    tracing::subscriber::with_default(capture.subscriber(), || {
        for (operation, streaming) in [
            (GenAiOperation::Chat, Some(false)),
            (GenAiOperation::Chat, Some(true)),
            (GenAiOperation::ChatStreaming, None),
        ] {
            let parent = completion_parent_span!(
                target: "third_party_runtime",
                name: "chat",
                operation: Empty,
                system_instructions: Option::<&str>::None,
            );
            let _entered = parent.enter();
            let builder = SpanBuilder::new("prov", "model", operation);
            match streaming {
                Some(streaming) => builder.streaming(streaming),
                None => builder,
            }
            .build();
        }
    });

    let reported = capture
        .spans()
        .iter()
        .map(|span| {
            (
                span.target,
                span.value("gen_ai.operation.name").cloned(),
                span.value("gen_ai.request.stream").cloned(),
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(
        reported,
        [
            (
                "third_party_runtime",
                Some(json!("chat")),
                Some(json!(false))
            ),
            (
                "third_party_runtime",
                Some(json!("chat")),
                Some(json!(true))
            ),
            (
                "third_party_runtime",
                Some(json!("chat")),
                Some(json!(true))
            ),
        ]
    );
}
