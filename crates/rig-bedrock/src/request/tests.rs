use super::*;
use crate::completion::{
    AMAZON_NOVA_LITE, AMAZON_NOVA_PRO, ANTHROPIC_CLAUDE_HAIKU_4_5, ANTHROPIC_CLAUDE_SONNET_4_5,
    LLAMA_3_1_70B_INSTRUCT,
};
use crate::streaming::tests::{CLAUDE, NOVA, hosted, reasoning, rich, tool_use, whole};
use rig_core::completion::ToolDefinition;
use rig_core::message::{
    AssistantMessage, CallId, Opaque, Origin, StopReason, ToolCall, ToolFunction, ToolName,
    ToolResult,
};
use rig_core::operation::Completion;
use rig_core::wire::{Mode, Operation, Wire};

const PROFILE: &str =
    "arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/a1b2c3d4";

fn name(name: &str) -> ToolName {
    ToolName::new(name).expect("a tool name")
}

fn tool(tool: &str) -> ToolDefinition {
    ToolDefinition {
        name: name(tool),
        description: format!("The {tool} tool"),
        parameters: json!({ "type": "object", "properties": {} }),
    }
}

/// The body `request` sends on `wire` in `mode`, prepared as the driver
/// prepares it.
fn encoded(wire: &Converse, request: CompletionRequest, mode: Mode) -> Value {
    let request = Completion::prepare(request, &wire.describe()).expect("prepares");
    serde_json::to_value(wire.encode(request, mode).expect("encodes").body).expect("serializes")
}

/// The body `history` sends to `model`, with a definition for every tool
/// it calls.
fn sent_on(wire: &Converse, history: Vec<Message>) -> Value {
    let mut request = CompletionRequest::new("next");
    request.tools = rig_history_conformance::tools_of(&history);
    request.chat_history = history;
    encoded(wire, request, Mode::Unary)
}

fn sent(model: &str, history: Vec<Message>) -> Value {
    sent_on(&Converse::new(model), history)
}

/// The body `history` sends to `model` in a request that declares a tool,
/// so its `toolConfig` can carry a hosted tool's use and result.
fn sent_with_tools(model: &str, history: Vec<Message>) -> Value {
    let mut request = CompletionRequest::new("next");
    request.tools = vec![tool("lookup")];
    request.chat_history = history;
    encoded(&Converse::new(model), request, Mode::Unary)
}

fn call(id: &str, tool: &str, arguments: Value) -> ToolCall {
    ToolCall::new(
        CallId::from_wire(id),
        ToolFunction::new(name(tool), arguments),
    )
}

fn answered(response: &rig_core::completion::CompletionResponse) -> Vec<Message> {
    let results = response
        .tool_calls()
        .map(|call| UserContent::ToolResult(call.result(vec![ToolResultContent::text("ok")])))
        .collect();
    vec![
        Message::user("q"),
        response.message().expect("a turn"),
        Message::User { content: results },
    ]
}

/// Each `ToolChoice` has its Converse form; `None`, or no tools, sends no
/// tool configuration.
#[test]
fn tool_choice_has_its_converse_form() {
    let config = |choice: Option<ToolChoice>, tools: Vec<ToolDefinition>| {
        let mut request = CompletionRequest::new("q");
        request.tool_choice = choice;
        request.tools = tools;
        encoded(&Converse::new(NOVA), request, Mode::Unary)
            .get("toolConfig")
            .cloned()
    };
    let tools = || vec![tool("add")];
    let spec = json!({ "toolSpec": {
        "name": "add",
        "description": "The add tool",
        "inputSchema": { "json": { "type": "object", "properties": {} } },
    } });
    assert_eq!(config(None, tools()), Some(json!({ "tools": [spec] })));
    for (choice, converse) in [
        (ToolChoice::Auto, json!({ "auto": {} })),
        (ToolChoice::Required, json!({ "any": {} })),
        (
            ToolChoice::Specific {
                function_names: vec![name("add")],
            },
            json!({ "tool": { "name": "add" } }),
        ),
    ] {
        assert_eq!(
            config(Some(choice), tools()),
            Some(json!({ "tools": [spec], "toolChoice": converse }))
        );
    }
    assert_eq!(config(Some(ToolChoice::None), tools()), None);
    assert_eq!(config(Some(ToolChoice::Auto), Vec::new()), None);
}

/// The inference configuration is always sent, with the temperature as the
/// 32-bit float Converse reads; the additional fields and the structured
/// output schema go as given.
#[test]
fn request_fields_have_their_converse_form() {
    let mut request = CompletionRequest::new("q");
    assert_eq!(
        encoded(&Converse::new(NOVA), request.clone(), Mode::Unary)["inferenceConfig"],
        json!({})
    );
    request.temperature = Some(0.7);
    request.max_tokens = Some(64);
    request.additional_params = Some(json!({ "top_k": 5 }));
    request.output_schema = Some(schemars::json_schema!({ "title": "answer", "type": "object" }));
    let body = encoded(&Converse::new(NOVA), request, Mode::Unary);
    assert_eq!(
        body["inferenceConfig"],
        json!({ "temperature": f64::from(0.7f32), "maxTokens": 64 })
    );
    assert_eq!(body["additionalModelRequestFields"], json!({ "top_k": 5 }));
    assert_eq!(
        body["outputConfig"],
        json!({ "textFormat": { "type": "json_schema", "structure": { "jsonSchema": {
            "schema": r#"{"title":"answer","type":"object"}"#,
            "name": "answer",
        } } } })
    );
}

/// Only the leading system messages are system blocks. A later one stays
/// where the history put it, as user text joined to its neighbouring user
/// content, so the cached prefix before it never changes.
#[test]
fn a_later_system_message_stays_in_place() {
    let body = sent(
        NOVA,
        vec![
            Message::system("lead"),
            Message::user("q"),
            Message::assistant("a"),
            Message::system("steer"),
            Message::user("next"),
        ],
    );
    assert_eq!(body["system"], json!([{ "text": "lead" }]));
    assert_eq!(
        body["messages"],
        json!([
            { "role": "user", "content": [{ "text": "q" }] },
            { "role": "assistant", "content": [{ "text": "a" }] },
            { "role": "user", "content": [{ "text": "steer" }, { "text": "next" }] },
        ])
    );
}

/// A short cache marks the system prompt and the last message. Bedrock
/// rejects a cache point anywhere after a reasoning turn (#1673), so that
/// one is skipped with a warning under `Ignore` and refused otherwise.
/// Reasoning another model made goes as text, so it does not.
#[test]
fn cache_points_follow_what_the_request_sends() {
    use rig_core::completion::{CacheRetention, GenerationOptions, OnUnsupported};
    let cached = |history: Vec<Message>| {
        CompletionRequest::from(history).options(
            GenerationOptions::default()
                .cache(CacheRetention::Short)
                .on_unsupported(OnUnsupported::Ignore),
        )
    };
    let sent_on = |wire: &Converse, history| encoded(wire, cached(history), Mode::Unary);
    let wire = Converse::new(CLAUDE);
    let point = json!({ "cachePoint": { "type": "default" } });
    let body = sent_on(&wire, vec![Message::system("s"), Message::user("q")]);
    assert_eq!(body["system"], json!([{ "text": "s" }, point]));
    assert_eq!(
        body["messages"][0]["content"],
        json!([{ "text": "q" }, point])
    );
    let signed = whole(
        CLAUDE,
        vec![reasoning("hm", Some("sig")), json!({ "text": "a" })],
        "end_turn",
    );
    let history = vec![
        Message::user("q"),
        signed.message().expect("a turn"),
        Message::user("more"),
    ];
    let body = sent_on(&wire, history.clone());
    assert_eq!(body["messages"][2]["content"], json!([{ "text": "more" }]));
    let other = Converse::new(ANTHROPIC_CLAUDE_HAIKU_4_5);
    let body = sent_on(&other, history);
    assert_eq!(body["messages"][1]["content"][0], json!({ "text": "hm" }));
    assert_eq!(
        body["messages"][2]["content"],
        json!([{ "text": "more" }, point])
    );
}

/// #43, #652: request documents reach the first user message through the
/// adapter, which sends a string document as its text. A document's name
/// is its content's digest, unique within the request (#404, #405).
#[test]
fn documents_are_named_by_content_and_land_in_the_first_user_message() {
    use rig_core::message::Document;
    let document = |data, media_type| Document {
        data,
        media_type: Some(media_type),
        additional_params: None,
    };
    let pdf = || {
        document(
            DocumentSourceKind::base64("aGVsbG8="),
            DocumentMediaType::PDF,
        )
    };
    let mut request = CompletionRequest::new("q");
    request.documents = vec![rig_core::completion::Document {
        id: "d1".to_owned(),
        text: "the sky is green".to_owned(),
        additional_props: Default::default(),
    }];
    request.chat_history = vec![Message::User {
        content: vec![
            UserContent::Document(document(
                DocumentSourceKind::string("plain"),
                DocumentMediaType::TXT,
            )),
            UserContent::Document(pdf()),
            UserContent::Document(pdf()),
            UserContent::text("question"),
        ],
    }];
    let body = encoded(&Converse::new(CLAUDE), request.clone(), Mode::Unary);
    assert_eq!(body, encoded(&Converse::new(CLAUDE), request, Mode::Unary));
    let content = body["messages"][0]["content"].as_array().expect("content");
    let names: Vec<&str> = content
        .iter()
        .filter_map(|block| block.pointer("/document/name")?.as_str())
        .collect();
    assert_eq!(names.len(), 2, "{body}");
    assert!(names[0].starts_with("document-"));
    assert_eq!(names[1], format!("{}-2", names[0]));
    assert!(content.contains(&json!({ "text": "plain" })), "{body}");
    assert!(
        content[0]["text"]
            .as_str()
            .is_some_and(|text| text.contains("the sky is green"))
    );
    assert_eq!(body["messages"].as_array().map(Vec::len), Some(1));
}

/// The body `history` encodes to on `wire` without being prepared, as a
/// request read back from storage reaches the encoder.
fn unprepared(history: Vec<Message>) -> Result<Value, EncodeError> {
    let mut request = CompletionRequest::new("unused");
    request.chat_history = history;
    let encoded = Converse::new(CLAUDE).encode(request, Mode::Unary)?;
    Ok(serde_json::to_value(encoded.body).expect("serializes"))
}

/// A document given as raw bytes sends the same base64 source as one given
/// as base64, a later system message goes as user text, and audio, which
/// Converse has no block for, is refused, when the request reaches the
/// encoder unprepared.
#[test]
fn unprepared_content_is_encoded_or_refused() {
    use rig_core::message::{Audio, Document};
    let document = |data| {
        unprepared(vec![Message::User {
            content: vec![UserContent::Document(Document {
                data,
                media_type: Some(DocumentMediaType::PDF),
                additional_params: None,
            })],
        }])
        .expect("encodes")
    };
    let raw = document(DocumentSourceKind::Raw(b"hello".to_vec()));
    assert_eq!(
        raw.pointer("/messages/0/content/1/document/source"),
        Some(&json!({ "bytes": "aGVsbG8=" })),
        "{raw}"
    );
    assert_eq!(raw, document(DocumentSourceKind::base64("aGVsbG8=")));

    let later = unprepared(vec![
        Message::user("q"),
        Message::assistant("a"),
        Message::system("steer"),
    ])
    .expect("encodes");
    assert_eq!(
        later.pointer("/messages/2"),
        Some(&json!({ "role": "user", "content": [{ "text": "steer" }] })),
        "{later}"
    );

    let audio = unprepared(vec![Message::User {
        content: vec![UserContent::Audio(Audio {
            data: DocumentSourceKind::base64("aGVsbG8="),
            media_type: None,
        })],
    }])
    .expect_err("audio is refused");
    assert!(
        audio.to_string().contains("Converse takes no audio"),
        "{audio}"
    );
}

/// A failed result states it with `status: error` to Nova and Claude, which
/// Converse documents the field for; any other result has no status.
#[test]
fn only_nova_and_claude_get_a_result_status() {
    let result = |id: &str, is_error| {
        UserContent::ToolResult(ToolResult {
            call: CallId::from_wire(id),
            name: name("t"),
            content: vec![ToolResultContent::text("out")],
            is_error,
        })
    };
    for (model, failed) in [
        (AMAZON_NOVA_PRO, Some(json!("error"))),
        (ANTHROPIC_CLAUDE_SONNET_4_5, Some(json!("error"))),
        (LLAMA_3_1_70B_INSTRUCT, None),
    ] {
        let calls = AssistantMessage::new(vec![
            AssistantContent::ToolCall(call("failed", "t", json!({}))),
            AssistantContent::ToolCall(call("fine", "t", json!({}))),
        ])
        .with_stop(StopReason::ToolUse);
        let body = sent(
            model,
            vec![
                Message::user("q"),
                Message::Assistant(calls),
                Message::User {
                    content: vec![result("failed", true), result("fine", false)],
                },
            ],
        );
        let statuses: Vec<_> = body["messages"][2]["content"]
            .as_array()
            .expect("results")
            .iter()
            .map(|block| block.pointer("/toolResult/status").cloned())
            .collect();
        assert_eq!(statuses, [failed, None], "{model}");
    }
}

/// #2137: structured results are Converse JSON, wrapped when not an object;
/// blank text is never sent and a message left empty says so.
#[test]
fn user_content_has_its_converse_form() {
    let c = call("t1", "list", json!({}));
    let history = vec![
        Message::User {
            content: vec![UserContent::text("q"), UserContent::text(" \n")],
        },
        Message::Assistant(AssistantMessage::new(vec![AssistantContent::ToolCall(
            c.clone(),
        )])),
        Message::User {
            content: vec![UserContent::ToolResult(c.result(vec![
                ToolResultContent::json(json!([1, 2])),
                ToolResultContent::json(json!({ "a": 1 })),
                ToolResultContent::text(" "),
            ]))],
        },
    ];
    let body = sent(NOVA, history);
    // `adapt` drops blank user text, which Converse rejects.
    assert_eq!(body["messages"][0]["content"], json!([{ "text": "q" }]));
    assert_eq!(
        body["messages"][2]["content"][0]["toolResult"]["content"],
        json!([{ "json": { "result": [1, 2] } }, { "json": { "a": 1 } }])
    );
}

/// Another model gets canonical fields only: reasoning text becomes text,
/// redacted reasoning and the hosted tool's items are dropped, and a call
/// id another provider made is one Converse accepts, on its result too.
#[test]
fn another_model_gets_canonical_fields() {
    let response = whole(CLAUDE, rich(), "tool_use");
    let body = sent(AMAZON_NOVA_LITE, answered(&response));
    assert_eq!(
        body["messages"][1]["content"],
        json!([
            { "text": "let me think" },
            { "text": "The harbor opens at nine." },
            { "text": "Calling." },
            tool_use("tooluse_1", "lookup", json!({ "q": "harbor" })),
        ])
    );
    let id = format!("call_{}|fc.{}", "a".repeat(40), "b".repeat(40));
    let foreign = call(&id, "lookup", json!({}));
    let history = vec![
        Message::user("q"),
        Message::Assistant({
            let mut message =
                AssistantMessage::new(vec![AssistantContent::ToolCall(foreign.clone())]);
            message.origin = Some(Origin::new("openai.responses", "openai", "gpt-5"));
            message
        }),
        Message::tool_results(vec![foreign.result(vec![ToolResultContent::text("ok")])]),
    ];
    let body = sent(AMAZON_NOVA_LITE, history);
    let expected: String = format!("call_{}_fc_{}", "a".repeat(40), "b".repeat(40))
        .chars()
        .take(64)
        .collect();
    assert_eq!(
        body["messages"][1]["content"][0]["toolUse"]["toolUseId"],
        json!(expected)
    );
    assert_eq!(
        body["messages"][2]["content"][0]["toolResult"]["toolUseId"],
        json!(expected)
    );
}

/// An edited block is rebuilt from its canonical fields: unsigned reasoning
/// goes to Claude as text, which it reads, and elsewhere as reasoning. A
/// Claude model behind an application inference profile, stated as Claude,
/// keeps its signatures.
#[test]
fn an_edited_block_is_rebuilt_for_the_family() {
    let wire = Converse::new(PROFILE).with_family(Family::Claude);
    let response = rig_core::test_utils::history::decode(
        &wire,
        Mode::Unary,
        [crate::completion::ConverseFrame::Whole(
            crate::streaming::tests::document(
                vec![reasoning("thought", Some("sig")), json!({ "text": "a" })],
                "end_turn",
            ),
        )],
    )
    .expect("decodes");
    let Some(Message::Assistant(mut turn)) = response.message() else {
        panic!("a turn");
    };
    let body = sent_on(
        &wire,
        vec![Message::user("q"), Message::Assistant(turn.clone())],
    );
    assert_eq!(
        body["messages"][1]["content"][0],
        reasoning("thought", Some("sig"))
    );
    if let AssistantContent::Reasoning(reasoning) = &mut turn.content[0] {
        reasoning.text = "edited".to_owned();
    }
    let edited = vec![Message::user("q"), Message::Assistant(turn.clone())];
    let body = sent_on(&wire, edited.clone());
    assert_eq!(
        body["messages"][1]["content"][0],
        json!({ "text": "edited" })
    );
    turn.origin = Some(Origin::new(
        "bedrock.converse",
        crate::completion::PROVIDER_NAME,
        NOVA,
    ));
    let body = sent(NOVA, vec![Message::user("q"), Message::Assistant(turn)]);
    assert_eq!(body["messages"][1]["content"][0], reasoning("edited", None));
}

/// Two calls sharing one id in a stored turn reach Converse distinct, each
/// answered by its own result.
#[test]
fn duplicate_stored_ids_reach_converse_distinct() {
    let (a, b) = (
        call("add", "add", json!({ "x": 1 })),
        call("add", "add", json!({ "x": 2 })),
    );
    let history = vec![
        Message::user("q"),
        Message::Assistant(AssistantMessage::new(vec![
            AssistantContent::ToolCall(a.clone()),
            AssistantContent::ToolCall(b.clone()),
        ])),
        Message::User {
            content: vec![
                UserContent::ToolResult(a.result(vec![ToolResultContent::text("1")])),
                UserContent::ToolResult(b.result(vec![ToolResultContent::text("2")])),
            ],
        },
    ];
    let body = sent(CLAUDE, history);
    let ids = |message: usize, pointer: &str| -> Vec<Value> {
        body["messages"][message]["content"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|block| block.pointer(pointer).cloned())
            .collect()
    };
    let calls = ids(1, "/toolUse/toolUseId");
    assert_ne!(calls[0], calls[1], "{body}");
    assert_eq!(ids(2, "/toolResult/toolUseId"), calls);
}

/// A tool history sent with no tools, or with `ToolChoice::None`, goes as
/// text with no `toolConfig`: Converse rejects tool blocks without one.
#[test]
fn a_tool_history_without_tools_is_text() {
    let response = whole(
        CLAUDE,
        vec![tool_use("tooluse_1", "lookup", json!({}))],
        "tool_use",
    );
    for choice in [None, Some(ToolChoice::None)] {
        let mut request = CompletionRequest::new("summarize");
        request.chat_history = answered(&response);
        if choice.is_some() {
            request.tools = vec![tool("lookup")];
        }
        request.tool_choice = choice;
        let body = encoded(&Converse::new(CLAUDE), request, Mode::Unary);
        assert!(body.get("toolConfig").is_none());
        let text = body.to_string();
        assert!(
            !text.contains("toolUse") && !text.contains("toolResult"),
            "{body}"
        );
    }
}

/// A same-model turn's decoded image goes back as a placeholder: Converse
/// reads no images in assistant turns.
#[test]
fn a_same_model_assistant_image_is_not_sent() {
    let image = json!({ "image": { "format": "png", "source": { "bytes": "cG5n" } } });
    let response = whole(
        AMAZON_NOVA_PRO,
        vec![image, json!({ "text": "drawn" })],
        "end_turn",
    );
    let body = sent(
        AMAZON_NOVA_PRO,
        vec![Message::user("draw"), response.message().expect("a turn")],
    );
    assert_eq!(body["messages"][1]["content"], json!([{ "text": "drawn" }]));
}

/// A store that writes whole numbers as floats still replays the item with
/// integers, which Converse's integer fields take.
#[test]
fn a_stored_item_goes_back_with_whole_numbers() {
    assert_eq!(
        whole_numbers(json!({ "documentIndex": 0.0, "start": 2.5, "end": [24.0, 7] })),
        json!({ "documentIndex": 0, "start": 2.5, "end": [24, 7] })
    );
    let [used, result] = hosted();
    let body = sent_with_tools(
        NOVA,
        vec![
            Message::user("q"),
            whole(NOVA, vec![used, result], "end_turn")
                .message()
                .expect("a turn"),
        ],
    );
    assert_eq!(body["messages"][1]["content"], json!(hosted()));
}

/// A hosted tool's use and its result replay together or not at all: a
/// use whose result cannot replay, or a result whose use is gone, is not
/// sent.
#[test]
fn a_hosted_use_replays_only_with_its_result() {
    let [used, result] = hosted();
    let opaque = |item: &Value, replay| {
        AssistantContent::Opaque(Opaque {
            item: item.clone(),
            replay,
        })
    };
    let origin = Origin::new("bedrock.converse", crate::completion::PROVIDER_NAME, NOVA);
    for (content, sent_items) in [
        (vec![opaque(&used, true), opaque(&result, true)], 2),
        (vec![opaque(&used, true), opaque(&result, false)], 0),
        (vec![opaque(&result, true)], 0),
    ] {
        let mut content = content;
        content.push(AssistantContent::text("done"));
        let turn = AssistantMessage::new(content)
            .with_origin(origin.clone())
            .with_stop(StopReason::Stop);
        let history = vec![Message::user("q"), Message::Assistant(turn)];
        let body = sent_with_tools(NOVA, history.clone());
        let blocks = body["messages"][1]["content"].as_array().expect("content");
        assert_eq!(blocks.len(), sent_items + 1, "{body}");
        // A request with no tools has no toolConfig to carry the pair.
        let body = sent(NOVA, history);
        assert_eq!(
            body["messages"][1]["content"],
            json!([{"text": "done"}]),
            "{body}"
        );
    }
}

/// What Converse would reject in `body`: a first message that is not a
/// user's, an empty message or blank text, two messages of one role in a
/// row, a call its next message does not answer, and tool blocks with no
/// `toolConfig`.
fn converse_violations(body: &Value) -> Vec<String> {
    let mut out = Vec::new();
    let messages = body["messages"].as_array().cloned().unwrap_or_default();
    if messages.first().map(|m| m["role"].clone()) != Some(json!("user")) {
        out.push("first message is not user".into());
    }
    let mut uses_any = false;
    for (i, m) in messages.iter().enumerate() {
        let content = m["content"].as_array().cloned().unwrap_or_default();
        if content.is_empty() {
            out.push(format!("message {i} has empty content"));
        }
        for b in &content {
            if let Some(t) = b.get("text").and_then(Value::as_str)
                && t.trim().is_empty()
            {
                out.push(format!("message {i} has blank text"));
            }
            if b.get("toolUse").is_some() || b.get("toolResult").is_some() {
                uses_any = true;
            }
        }
        if i > 0 && messages[i - 1]["role"] == m["role"] {
            out.push(format!(
                "messages {} and {i} share role {}",
                i - 1,
                m["role"]
            ));
        }
        // every toolUse is answered in the next user message
        let used: Vec<String> = content
            .iter()
            .filter_map(|b| {
                b.pointer("/toolUse/toolUseId")
                    .and_then(Value::as_str)
                    .map(str::to_owned)
            })
            .filter(|_| m["role"] == "assistant")
            .collect();
        let next = messages
            .get(i + 1)
            .map(|n| n["content"].to_string())
            .unwrap_or_default();
        for id in used {
            if !next.contains(&format!("\"toolUseId\":\"{id}\"")) {
                out.push(format!("toolUse {id} at {i} unanswered"));
            }
        }
    }
    if uses_any && body.get("toolConfig").is_none() {
        out.push("toolUse/toolResult without toolConfig".into());
    }
    out
}

/// Histories `adapt` must shape into a request Converse takes, with and
/// without declared tools: leading orphan results (round-5 F1), blank user
/// text (F8), an emptied turn holding only redacted reasoning (F7), and
/// system messages in awkward places.
#[test]
fn adversarial_histories_encode_to_requests_converse_takes() {
    let other = Some(Origin::new("openai.chat", "openai", "gpt-4.1"));
    let asst = |content: Vec<AssistantContent>, stop: StopReason| {
        Message::Assistant(
            AssistantMessage::new(content)
                .with_origin(other.clone())
                .with_stop(stop),
        )
    };
    let result = |id: &str| Message::User {
        content: vec![UserContent::ToolResult(
            call(id, "lookup", json!({})).result(vec![ToolResultContent::text("r")]),
        )],
    };
    let text = |t: &str| AssistantContent::text(t);
    let tc = |id: &str| AssistantContent::ToolCall(call(id, "lookup", json!({})));
    let cases: Vec<(&str, Vec<Message>)> = vec![
        (
            "orphan result first",
            vec![
                result("gone"),
                asst(vec![text("hello")], StopReason::Stop),
                Message::user("q"),
            ],
        ),
        (
            "system then orphan result first",
            vec![
                Message::system("sys"),
                result("gone"),
                asst(vec![text("hello")], StopReason::Stop),
                Message::user("q"),
            ],
        ),
        (
            "results of an aborted turn first",
            vec![
                Message::user("a"),
                asst(vec![tc("x")], StopReason::Aborted("cut".into())),
                result("x"),
                asst(vec![text("hello")], StopReason::Stop),
                Message::user("q"),
            ],
        ),
        (
            "system between assistants",
            vec![
                Message::user("a"),
                asst(vec![text("one")], StopReason::Stop),
                Message::system("mid"),
                asst(vec![text("two")], StopReason::Stop),
                Message::user("q"),
            ],
        ),
        (
            "orphan between assistants",
            vec![
                Message::user("a"),
                asst(vec![text("one")], StopReason::Stop),
                result("gone"),
                asst(vec![text("two")], StopReason::Stop),
                Message::user("q"),
            ],
        ),
        (
            "system while a call waits",
            vec![
                Message::user("a"),
                asst(vec![tc("x")], StopReason::ToolUse),
                Message::system("mid"),
                result("x"),
                Message::user("q"),
            ],
        ),
        (
            "blank user between",
            vec![
                Message::user("a"),
                asst(vec![text("one")], StopReason::Stop),
                Message::user("   "),
                asst(vec![text("two")], StopReason::Stop),
                Message::user("q"),
            ],
        ),
        (
            "only an emptied turn",
            vec![
                Message::user("a"),
                asst(
                    vec![AssistantContent::Reasoning(rig_core::message::Reasoning {
                        text: String::new(),
                        redacted: true,
                        native: None,
                    })],
                    StopReason::Stop,
                ),
                Message::user("q"),
            ],
        ),
        (
            "leading assistant then system",
            vec![
                asst(vec![text("hello")], StopReason::Stop),
                Message::system("later"),
                Message::user("q"),
            ],
        ),
    ];
    let mut failures = Vec::new();
    for (label, history) in cases {
        for tools in [true, false] {
            let mut request = CompletionRequest::new("next");
            if tools {
                request.tools = vec![tool("lookup")];
            }
            request.chat_history = history.clone();
            let request = match Completion::prepare(request, &Converse::new(CLAUDE).describe()) {
                Ok(request) => request,
                Err(error) => {
                    failures.push(format!("{label} tools={tools}: prepare failed: {error}"));
                    continue;
                }
            };
            let body = serde_json::to_value(
                Converse::new(CLAUDE)
                    .encode(request, Mode::Unary)
                    .expect("encodes")
                    .body,
            )
            .expect("serializes");
            let v = converse_violations(&body);
            if !v.is_empty() {
                failures.push(format!(
                    "{label} tools={tools}: {v:?}\n  {}",
                    body["messages"]
                ));
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
