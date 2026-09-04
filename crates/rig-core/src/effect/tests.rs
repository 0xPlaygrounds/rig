use serde_json::json;

use super::*;
use crate::{
    completion::{AssistantContent, Usage},
    embeddings::Embedding,
    error::ErrorKind,
    rerank::RerankResponse,
    tool::{ToolExecutionError, ToolOutput},
    vector_store::request::VectorSearchRequest,
};

fn round_trip<T>(value: &T) -> T
where
    T: Serialize + serde::de::DeserializeOwned + PartialEq + std::fmt::Debug,
{
    let json = serde_json::to_value(value).expect("serializes");
    let back: T = serde_json::from_value(json.clone()).expect("deserializes");
    assert_eq!(&back, value, "round trip through {json}");
    back
}

fn request() -> CompletionRequest {
    CompletionRequest {
        model: None,
        chat_history: vec![Message::user("hi")],
        documents: vec![],
        tools: vec![],
        temperature: None,
        max_tokens: None,
        tool_choice: None,
        additional_params: None,
        output_schema: None,
        record_telemetry_content: false,
    }
}

#[test]
fn effect_id_is_transparent_and_ordered() {
    let id = EffectId::from_raw(7);
    assert_eq!(serde_json::to_value(id).expect("serializes"), json!(7));
    assert!(EffectId::from_raw(1) < EffectId::from_raw(2));
    assert_eq!(id.to_string(), "effect:7");
}

#[test]
fn handler_key_is_a_transparent_string() {
    let key = HandlerKey::from("tool:add");
    assert_eq!(
        serde_json::to_value(&key).expect("serializes"),
        json!("tool:add")
    );
    assert_eq!(round_trip(&key).as_str(), "tool:add");
    assert_eq!(format!("{key:?}"), "HandlerKey(\"tool:add\")");
}

#[test]
fn a_key_is_read_by_its_grammar_and_written_back() {
    use super::KeyParts;
    for (key, owner, kind, label, generation) in [
        (
            "golden/tool:add#2",
            Some("golden"),
            Some("tool"),
            "add",
            Some(2),
        ),
        (
            "golden/model:default",
            Some("golden"),
            Some("model"),
            "default",
            None,
        ),
        (
            "golden/retrieve:tools#0",
            Some("golden"),
            Some("retrieve"),
            "tools",
            Some(0),
        ),
        ("host/note", Some("host"), None, "note", None),
        ("model:fast", None, Some("model"), "fast", None),
        ("tool:lookup#7", None, Some("tool"), "lookup", Some(7)),
        ("memory", None, None, "memory", None),
        ("not-a-model", None, None, "not-a-model", None),
    ] {
        let parts = HandlerKey::from(key).parts();
        assert_eq!(parts.owner.as_deref(), owner, "{key}: owner");
        assert_eq!(parts.kind.as_deref(), kind, "{key}: kind");
        assert_eq!(parts.label.as_ref(), label, "{key}: label");
        assert_eq!(parts.generation, generation, "{key}: generation");
        assert_eq!(parts.to_key(), HandlerKey::from(key), "{key}: round trip");
        assert_eq!(parts.to_string(), key);
        assert_eq!(KeyParts::parse(key), parts);
    }
    // A generation is digits after the last `#`; anything else is label.
    let odd = HandlerKey::from("host/tool:a#b").parts();
    assert_eq!(odd.label.as_ref(), "a#b");
    assert_eq!(odd.generation, None);
}

#[test]
fn descriptor_variant_is_the_family() {
    let cases = [
        (
            FamilyDescriptor::Completion {
                model: ModelRef::new("gpt"),
                capabilities: ProviderCapabilities::new().with_native_output_tool_composition(true),
            },
            EffectFamily::Completion,
        ),
        (
            FamilyDescriptor::Tool {
                name: "add".into(),
                description: "adds".into(),
                parameters: json!({"type": "object"}),
                embedding: Some(ToolEmbeddingDescriptor {
                    context: json!({"scale": 2}),
                    embedding_docs: vec!["add numbers".into()],
                }),
            },
            EffectFamily::Tool,
        ),
        (
            FamilyDescriptor::Embed {
                model: "small".into(),
                dims: Some(3),
                max_documents: 16,
                modality: EmbedModality::Image,
            },
            EffectFamily::Embed,
        ),
        (FamilyDescriptor::Memory {}, EffectFamily::Memory),
        (FamilyDescriptor::Retrieve {}, EffectFamily::Retrieve),
        (
            FamilyDescriptor::Custom {
                kind: "host:tick".into(),
            },
            EffectFamily::Custom,
        ),
    ];
    for (family, expected) in cases {
        assert_eq!(family.family(), expected);
        let descriptor = HandlerDescriptor {
            key: HandlerKey::from("k"),
            family,
            layers: Vec::new(),
        };
        let back = round_trip(&descriptor);
        assert_eq!(back.family.family(), expected);
        let json = serde_json::to_value(&descriptor).expect("serializes");
        assert_eq!(
            json["family"]["family"],
            json!(expected.name().trim_end_matches("_call"))
        );
    }
}

#[test]
fn family_markers_name_their_family() {
    assert_eq!(family::Completion::FAMILY, EffectFamily::Completion);
    assert_eq!(family::Tool::FAMILY, EffectFamily::Tool);
    assert_eq!(family::Embed::FAMILY, EffectFamily::Embed);
    assert_eq!(family::Memory::FAMILY, EffectFamily::Memory);
    assert_eq!(family::Retrieve::FAMILY, EffectFamily::Retrieve);
}

#[test]
fn every_kind_round_trips_and_names_itself() {
    let kinds = vec![
        (
            EffectKind::Completion {
                request: request(),
                stream: true,
            },
            "completion",
            EffectFamily::Completion,
            true,
        ),
        (
            EffectKind::ToolCall {
                name: "add".into(),
                args: r#"{"a":1}"#.into(),
            },
            "tool_call",
            EffectFamily::Tool,
            false,
        ),
        (
            EffectKind::Embed {
                inputs: EmbedInputs::Texts(vec!["a".into()]),
            },
            "embed",
            EffectFamily::Embed,
            false,
        ),
        (
            EffectKind::Embed {
                inputs: EmbedInputs::Images(vec![vec![1, 2, 3]]),
            },
            "embed",
            EffectFamily::Embed,
            false,
        ),
        (
            EffectKind::Memory {
                op: MemoryOp::Append {
                    conversation: ConversationId::from("c1"),
                    messages: vec![Message::user("hello")],
                },
            },
            "memory",
            EffectFamily::Memory,
            false,
        ),
        (
            EffectKind::Memory {
                op: MemoryOp::Load {
                    conversation: ConversationId::from("c1"),
                },
            },
            "memory",
            EffectFamily::Memory,
            false,
        ),
        (
            EffectKind::Memory {
                op: MemoryOp::Clear {
                    conversation: ConversationId::from("c1"),
                },
            },
            "memory",
            EffectFamily::Memory,
            false,
        ),
        (
            EffectKind::Custom {
                kind: Arc::from("host:tick"),
                payload: json!({"frame": 3}),
            },
            "host:tick",
            EffectFamily::Custom,
            false,
        ),
    ];
    for (kind, name, family, streams) in kinds {
        assert_eq!(kind.name(), name);
        assert_eq!(kind.family(), family);
        assert_eq!(kind.streams(), streams);
        let json = serde_json::to_value(&kind).expect("serializes");
        let back: EffectKind = serde_json::from_value(json.clone()).expect("deserializes");
        assert_eq!(
            serde_json::to_value(&back).expect("serializes"),
            json,
            "kind round trip"
        );
    }
}

#[test]
fn retrieve_queries_round_trip() {
    let req = VectorSearchRequest::builder()
        .query("cats")
        .samples(3)
        .build();
    for query in [
        RetrieveQuery::TopN { req: req.clone() },
        RetrieveQuery::TopNIds { req },
    ] {
        let kind = EffectKind::Retrieve { query };
        let json = serde_json::to_value(&kind).expect("serializes");
        let back: EffectKind = serde_json::from_value(json.clone()).expect("deserializes");
        assert_eq!(
            serde_json::to_value(&back).expect("serializes"),
            json,
            "retrieve query round trip"
        );
        assert_eq!(back.family(), EffectFamily::Retrieve);
    }
}

#[test]
fn every_outcome_round_trips() {
    let tool_error = ToolExecutionError::new(crate::tool::ToolErrorKind::Timeout, "slow")
        .with_code("T1")
        .with_http_status(504)
        .with_model_feedback("try again");
    let outcomes = vec![
        (
            Outcome::Completion(CompletionResponse::new(
                vec![AssistantContent::text("hi")],
                Usage::new(),
                "mock",
            )),
            EffectFamily::Completion,
        ),
        (
            Outcome::ToolResult {
                result: ToolResult::success(ToolOutput::json(json!({"sum": 3}))),
            },
            EffectFamily::Tool,
        ),
        (
            Outcome::ToolResult {
                result: ToolResult::failed(tool_error),
            },
            EffectFamily::Tool,
        ),
        (
            Outcome::ToolResult {
                result: ToolResult::failed(ToolExecutionError::refused("no")),
            },
            EffectFamily::Tool,
        ),
        (
            Outcome::ToolResult {
                result: ToolResult::skipped("policy"),
            },
            EffectFamily::Tool,
        ),
        (
            Outcome::Embeddings(EmbedOutputs::Texts(EmbeddingResponse::new(
                vec![Embedding {
                    document: "a".into(),
                    vec: vec![0.5, 0.25],
                }],
                "mock",
            ))),
            EffectFamily::Embed,
        ),
        (
            Outcome::Memory(MemoryOutcome::Loaded {
                messages: vec![Message::user("hi")],
            }),
            EffectFamily::Memory,
        ),
        (
            Outcome::Memory(MemoryOutcome::Appended),
            EffectFamily::Memory,
        ),
        (
            Outcome::Memory(MemoryOutcome::Cleared),
            EffectFamily::Memory,
        ),
        (
            Outcome::Documents(RetrievedDocuments::Scored(vec![(
                0.9,
                "d1".into(),
                json!({"text": "cat"}),
            )])),
            EffectFamily::Retrieve,
        ),
        (
            Outcome::Documents(RetrievedDocuments::Ids(vec![(0.9, "d1".into())])),
            EffectFamily::Retrieve,
        ),
        (Outcome::Custom(json!({"ok": true})), EffectFamily::Custom),
    ];
    for (outcome, family) in outcomes {
        assert_eq!(outcome.family(), family);
        let json = serde_json::to_value(&outcome).expect("serializes");
        let back: Outcome = serde_json::from_value(json.clone()).expect("deserializes");
        assert_eq!(
            serde_json::to_value(&back).expect("serializes"),
            json,
            "outcome round trip"
        );
    }
}

#[test]
fn tool_result_wire_form_keeps_status_and_error_fields_but_not_source() {
    let error = ToolExecutionError::new(crate::tool::ToolErrorKind::Provider, "boom")
        .with_source(std::io::Error::other("disk"))
        .with_retryable(true);
    let result = ToolResult::failed(error);
    let json = serde_json::to_value(&result).expect("serializes");
    assert_eq!(json["status"], json!("error"));
    assert_eq!(json["value"]["kind"], json!("provider"));
    assert_eq!(json["value"]["retryable"], json!(true));
    assert!(json["value"].get("source").is_none());
    let back: ToolResult = serde_json::from_value(json).expect("deserializes");
    let back_error = back.error().expect("still an error");
    assert_eq!(back_error.message(), "boom");
    assert_eq!(back_error.retryable(), Some(true));
    assert!(std::error::Error::source(back_error).is_none());
}

#[test]
fn tool_output_rejects_an_empty_wire_list() {
    let error = serde_json::from_value::<ToolOutput>(json!([])).expect_err("empty is rejected");
    assert!(error.to_string().contains("no content blocks"));
}

#[test]
fn custom_kind_label_is_a_plain_string_on_the_wire() {
    let kind = EffectKind::Custom {
        kind: Arc::from("host:tick"),
        payload: json!(1),
    };
    let json = serde_json::to_value(&kind).expect("serializes");
    assert_eq!(
        json,
        json!({"effect": "custom", "kind": "host:tick", "payload": 1})
    );
}

// ---- families know their shapes ----

#[test]
fn every_family_wraps_its_request_and_unwraps_its_own_outcome() {
    let completion = family::Completion::wrap(request()).expect("a request has a wire form");
    assert_eq!(completion.family(), EffectFamily::Completion);
    let response =
        CompletionResponse::new(vec![AssistantContent::text("hi")], Usage::new(), "mock");
    let answer =
        family::Completion::unwrap(Outcome::Completion(response.clone())).expect("own family");
    assert_eq!(answer.choice, response.choice);

    let tool = family::Tool::wrap(ToolCallRequest {
        name: "add".into(),
        args: "{}".into(),
    })
    .expect("a request has a wire form");
    assert_eq!(tool.family(), EffectFamily::Tool);
    let answer = family::Tool::unwrap(Outcome::ToolResult {
        result: ToolResult::success(ToolOutput::text("3")),
    })
    .expect("own family");
    assert_eq!(answer.output().as_text(), Some("3"));

    let memory = family::Memory::wrap(MemoryOp::Clear {
        conversation: crate::id::ConversationId::new("c"),
    })
    .expect("a request has a wire form");
    assert_eq!(memory.family(), EffectFamily::Memory);
    assert!(matches!(
        family::Memory::unwrap(Outcome::Memory(MemoryOutcome::Cleared)),
        Ok(MemoryOutcome::Cleared)
    ));

    let retrieve = family::Retrieve::wrap(RetrieveQuery::TopNIds {
        req: VectorSearchRequest::builder()
            .query("q")
            .samples(1)
            .build()
            .map_filter(Filter::interpret),
    })
    .expect("a request has a wire form");
    assert_eq!(retrieve.family(), EffectFamily::Retrieve);
    assert!(matches!(
        family::Retrieve::unwrap(Outcome::Documents(RetrievedDocuments::Ids(vec![]))),
        Ok(RetrievedDocuments::Ids(ids)) if ids.is_empty()
    ));

    let rerank = family::Rerank::wrap(RerankRequest {
        query: "q".into(),
        documents: vec!["a".into()],
    })
    .expect("a request has a wire form");
    assert_eq!(rerank.family(), EffectFamily::Rerank);
    assert!(matches!(
        family::Rerank::unwrap(Outcome::Reranked(RerankResponse::new(vec![], "mock"))),
        Ok(response) if response.provider == "mock"
    ));

    let embed = family::Embed::wrap(EmbedInputs::Texts(vec!["a".into()]))
        .expect("a request has a wire form");
    assert_eq!(embed.family(), EffectFamily::Embed);
    assert!(
        family::Embed::unwrap(Outcome::Embeddings(EmbedOutputs::Texts(
            EmbeddingResponse::new(
                vec![Embedding {
                    document: "a".into(),
                    vec: vec![0.0],
                }],
                "mock",
            )
        )))
        .is_ok()
    );
}

#[test]
fn a_family_reports_another_familys_outcome_as_a_mismatch() {
    let report = family::Completion::unwrap(Outcome::Memory(MemoryOutcome::Cleared))
        .expect_err("not a completion");
    assert_eq!(report.kind, ErrorKind::Internal);
    assert!(
        report.message.contains("expected a completion outcome")
            && report.message.contains("memory"),
        "{}",
        report.message
    );
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct AskUser {
    prompt: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct Reply {
    text: String,
}

impl CustomEffect for AskUser {
    const KIND: &'static str = "test:ask_user";
    type Answer = Reply;
}

#[test]
fn a_custom_effect_travels_as_its_declared_kind_and_answer() {
    let kind = family::Custom::<AskUser>::wrap(AskUser {
        prompt: "name?".into(),
    })
    .expect("a request has a wire form");
    match &kind {
        EffectKind::Custom { kind, payload } => {
            assert_eq!(&**kind, AskUser::KIND);
            assert_eq!(payload, &json!({"prompt": "name?"}));
        }
        other => panic!("expected a custom kind, got {other:?}"),
    }
    let answer = family::Custom::<AskUser>::unwrap(Outcome::Custom(json!({"text": "Ada"})))
        .expect("the declared answer");
    assert_eq!(answer, Reply { text: "Ada".into() });

    let report = family::Custom::<AskUser>::unwrap(Outcome::Custom(json!({"nope": 1})))
        .expect_err("not a Reply");
    assert_eq!(report.kind, ErrorKind::Internal);
    assert!(report.message.contains(AskUser::KIND), "{}", report.message);

    let report = family::Custom::<AskUser>::unwrap(Outcome::Memory(MemoryOutcome::Cleared))
        .expect_err("another family");
    assert!(
        report.message.contains("expected a custom outcome"),
        "{}",
        report.message
    );

    // The marker is `Copy` for any `E` and names its kind.
    let marker = family::Custom::<AskUser>::new();
    let copied = marker;
    assert_eq!(marker, copied);
    assert_eq!(format!("{marker:?}"), "Custom<test:ask_user>");
}

/// A custom effect whose `Serialize` fails.
#[derive(Debug, Deserialize)]
struct Unserializable;

impl Serialize for Unserializable {
    fn serialize<S: serde::Serializer>(&self, _serializer: S) -> Result<S::Ok, S::Error> {
        Err(serde::ser::Error::custom("no wire form"))
    }
}

impl CustomEffect for Unserializable {
    const KIND: &'static str = "test:unserializable";
    type Answer = serde_json::Value;
}

/// A custom effect that does not serialize has no wire form: the wrap is
/// a `Request` report naming the kind and the serde error, never a
/// dispatch carrying the error as its payload.
#[test]
fn a_custom_effect_that_does_not_serialize_is_refused_at_wrap() {
    let report = family::Custom::<Unserializable>::wrap(Unserializable).expect_err("no wire form");
    assert_eq!(report.kind, ErrorKind::Request);
    assert!(
        report.message.contains("test:unserializable") && report.message.contains("no wire form"),
        "{report:?}"
    );
}

// ---- the effect row ----

fn descriptor(key: &str, family: FamilyDescriptor) -> HandlerDescriptor {
    HandlerDescriptor {
        key: HandlerKey::from(key),
        family,
        layers: Vec::new(),
    }
}

/// A row is a subset of a handler table when every key it names is served
/// as the family it needs; the first gap names the key, the family needed
/// and what serves it instead.
#[test]
fn a_row_is_checked_against_a_handler_table_and_names_its_first_gap() {
    let row: EffectRow = [
        (
            HandlerKey::from("a/model:default"),
            EffectFamily::Completion,
        ),
        (HandlerKey::from("a/tool:add#0"), EffectFamily::Tool),
    ]
    .into_iter()
    .collect();
    let served = vec![
        descriptor(
            "a/model:default",
            FamilyDescriptor::Completion {
                model: ModelRef::new("m"),
                capabilities: ProviderCapabilities::default(),
            },
        ),
        descriptor(
            "a/tool:add#0",
            FamilyDescriptor::Tool {
                name: "add".into(),
                description: String::new(),
                parameters: json!({}),
                embedding: None,
            },
        ),
        descriptor("host/note", FamilyDescriptor::Custom { kind: "n".into() }),
    ];
    assert_eq!(row.is_subset_of(&served), Ok(()));
    let gap = row
        .is_subset_of(&served[..1])
        .expect_err("the tool is not served");
    assert_eq!(gap.key, HandlerKey::from("a/tool:add#0"));
    assert_eq!(gap.needed, EffectFamily::Tool);
    assert_eq!(gap.served, None);
    assert_eq!(gap.to_string(), "`a/tool:add#0` (tool_call) is not served");
    let wrong = vec![descriptor("a/model:default", FamilyDescriptor::Memory {})];
    let gap = row
        .is_subset_of(&wrong)
        .expect_err("served as another family");
    assert_eq!(gap.served, Some(EffectFamily::Memory));
    assert_eq!(
        gap.to_string(),
        "`a/model:default` is needed as completion but served as memory"
    );
}

/// A diff names what the other row lacks, what it has extra, and the keys
/// both name as different families, in key order.
#[test]
fn a_row_diff_names_every_difference() {
    let this: EffectRow = [
        (HandlerKey::from("a"), EffectFamily::Completion),
        (HandlerKey::from("b"), EffectFamily::Tool),
    ]
    .into_iter()
    .collect();
    let other: EffectRow = [
        (HandlerKey::from("b"), EffectFamily::Memory),
        (HandlerKey::from("c"), EffectFamily::Retrieve),
    ]
    .into_iter()
    .collect();
    let diffs: Vec<String> = this.diff(&other).iter().map(ToString::to_string).collect();
    assert_eq!(
        diffs,
        [
            "`a` (completion) is missing",
            "`b` is tool_call here and memory there",
            "`c` (retrieve) is extra",
        ]
    );
    assert!(this.diff(&this).is_empty());
    let wire = serde_json::to_value(&this).expect("serializes");
    assert_eq!(wire, json!({ "a": "completion", "b": "tool" }));
}
