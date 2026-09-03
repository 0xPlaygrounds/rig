use serde_json::json;

use super::*;
use crate::{
    completion::{AssistantContent, Usage},
    embeddings::Embedding,
    error::ErrorKind,
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
                context: ToolContext::new(),
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
                context: ToolContext::new(),
            },
            EffectFamily::Tool,
        ),
        (
            Outcome::ToolResult {
                result: ToolResult::failed(tool_error),
                context: ToolContext::new(),
            },
            EffectFamily::Tool,
        ),
        (
            Outcome::ToolResult {
                result: ToolResult::failed(ToolExecutionError::refused("no")),
                context: ToolContext::new(),
            },
            EffectFamily::Tool,
        ),
        (
            Outcome::ToolResult {
                result: ToolResult::skipped("policy"),
                context: ToolContext::new(),
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
fn effect_record_and_log_round_trip() {
    let log: EffectLog = vec![
        EffectRecord {
            id: EffectId::from_raw(1),
            key: HandlerKey::from("model"),
            kind: EffectKind::Completion {
                request: request(),
                stream: false,
            },
            outcome: Ok(Outcome::Completion(CompletionResponse::new(
                vec![AssistantContent::text("hi")],
                Usage::new(),
                "mock",
            ))),
        },
        EffectRecord {
            id: EffectId::from_raw(2),
            key: HandlerKey::from("tool:add"),
            kind: EffectKind::ToolCall {
                name: "add".into(),
                args: "{}".into(),
                context: ToolContext::new(),
            },
            outcome: Err(ErrorReport::new(ErrorKind::Timeout, "slow")),
        },
    ];
    let json = serde_json::to_string(&log).expect("serializes");
    let back: EffectLog = serde_json::from_str(&json).expect("deserializes");
    assert_eq!(
        serde_json::to_string(&back).expect("serializes"),
        json,
        "log round trip"
    );
    assert_eq!(back.len(), 2);
    assert_eq!(back[0].id, EffectId::from_raw(1));
    assert_eq!(back[1].key.as_str(), "tool:add");
    assert!(matches!(&back[1].outcome, Err(report) if report.kind == ErrorKind::Timeout));
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
    let completion = family::Completion::wrap(request());
    assert_eq!(completion.family(), EffectFamily::Completion);
    let response =
        CompletionResponse::new(vec![AssistantContent::text("hi")], Usage::new(), "mock");
    let answer =
        family::Completion::unwrap(Outcome::Completion(response.clone())).expect("own family");
    assert_eq!(answer.choice, response.choice);

    let tool = family::Tool::wrap(ToolCallRequest {
        name: "add".into(),
        args: "{}".into(),
        context: ToolContext::new(),
    });
    assert_eq!(tool.family(), EffectFamily::Tool);
    let answer = family::Tool::unwrap(Outcome::ToolResult {
        result: ToolResult::success(ToolOutput::text("3")),
        context: ToolContext::new(),
    })
    .expect("own family");
    assert_eq!(answer.result.output().as_text(), Some("3"));

    let memory = family::Memory::wrap(MemoryOp::Clear {
        conversation: crate::id::ConversationId::new("c"),
    });
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
    });
    assert_eq!(retrieve.family(), EffectFamily::Retrieve);
    assert!(matches!(
        family::Retrieve::unwrap(Outcome::Documents(RetrievedDocuments::Ids(vec![]))),
        Ok(RetrievedDocuments::Ids(ids)) if ids.is_empty()
    ));

    let embed = family::Embed::wrap(EmbedInputs::Texts(vec!["a".into()]));
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
    });
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
