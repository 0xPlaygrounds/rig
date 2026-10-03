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
    wire.encode(request, mode).expect("encodes").body
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
/// 32-bit float Converse reads; the additional fields, the structured
/// output schema and a unary request's guardrail go as given.
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
    let wire = Converse::new(NOVA).with_guardrail(
        "g1",
        "DRAFT",
        aws_sdk_bedrockruntime::types::GuardrailTrace::Enabled,
    );
    let body = encoded(&wire, request.clone(), Mode::Unary);
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
    assert_eq!(
        body["guardrailConfig"],
        json!({ "guardrailIdentifier": "g1", "guardrailVersion": "DRAFT", "trace": "enabled" })
    );
    assert!(
        encoded(&wire, request, Mode::Streaming)
            .get("guardrailConfig")
            .is_none()
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

/// Prompt caching marks the system prompt and the last message, unless the
/// request sends reasoning: Bedrock rejects a cache point anywhere after a
/// reasoning turn (#1673). Reasoning another model made goes as text, so it
/// does not.
#[test]
fn cache_points_follow_what_the_request_sends() {
    let wire = Converse::new(CLAUDE).with_prompt_caching();
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
    let other = Converse::new(ANTHROPIC_CLAUDE_HAIKU_4_5).with_prompt_caching();
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
        let calls = AssistantMessage {
            content: vec![
                AssistantContent::ToolCall(call("failed", "t", json!({}))),
                AssistantContent::ToolCall(call("fine", "t", json!({}))),
            ],
            origin: None,
            stop: Some(StopReason::ToolUse),
        };
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
            content: vec![UserContent::text(" \n")],
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
    assert_eq!(
        body["messages"][0]["content"],
        json!([{ "text": "<empty>" }])
    );
    assert_eq!(
        body["messages"][2]["content"][0]["toolResult"]["content"],
        json!([{ "json": { "result": [1, 2] } }, { "json": { "a": 1 } }])
    );
}

/// Calls rig issued ids for are one alias wherever they appear, distinct
/// from the provider's ids and from a hosted tool's, and every result
/// follows its call.
#[test]
fn issued_ids_are_one_alias_and_hosted_ids_are_reserved() {
    let issued = call("", "lookup", json!({}));
    let item = |item| AssistantContent::Opaque(Opaque { item, replay: true });
    let history = vec![
        Message::user("q"),
        Message::Assistant(AssistantMessage {
            content: vec![
                item(json!({ "toolUse": {
                    "toolUseId": "tool-0", "name": "nova_grounding", "input": {}, "type": "server_tool_use",
                } })),
                item(
                    json!({ "toolResult": { "toolUseId": "tool-0", "content": [{ "text": "found" }] } }),
                ),
                AssistantContent::ToolCall(issued.clone()),
            ],
            origin: Some(Origin::new(
                "bedrock.converse",
                crate::completion::PROVIDER_NAME,
                NOVA,
            )),
            stop: Some(StopReason::ToolUse),
        }),
        Message::User {
            content: vec![UserContent::ToolResult(
                issued.result(vec![ToolResultContent::text("done")]),
            )],
        },
    ];
    let body = sent(NOVA, history);
    let ids: Vec<&str> = body["messages"]
        .as_array()
        .into_iter()
        .flatten()
        .flat_map(|message| message["content"].as_array().into_iter().flatten())
        .filter_map(|block| {
            block
                .pointer("/toolUse/toolUseId")
                .or_else(|| block.pointer("/toolResult/toolUseId"))?
                .as_str()
        })
        .collect();
    assert_eq!(ids, ["tool-0", "tool-0", "tool-1", "tool-1"]);
}

/// The same model gets every item back as it came, in both modes: signed
/// and redacted reasoning, cited text, the hosted tool's use and result,
/// and the call, whose id its result shares.
#[test]
fn the_same_model_gets_its_items_back_verbatim() {
    let response = whole(CLAUDE, rich(), "tool_use");
    let body = sent(CLAUDE, answered(&response));
    assert_eq!(body["messages"][1]["content"], json!(rich()));
    assert_eq!(
        body["messages"][2]["content"],
        json!([{ "toolResult": { "toolUseId": "tooluse_1", "content": [{ "text": "ok" }] } }])
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
        Message::Assistant(AssistantMessage {
            origin: Some(Origin::new("openai.responses", "openai", "gpt-5")),
            ..AssistantMessage::new(vec![AssistantContent::ToolCall(foreign.clone())])
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

/// Call arguments are an object by type, so a call made with `null`
/// arguments sends `{}`, never a `null` input.
#[test]
fn null_arguments_are_sent_as_an_object() {
    let response = whole(
        NOVA,
        vec![tool_use("tooluse_2", "lookup", Value::Null)],
        "tool_use",
    );
    let body = sent(NOVA, answered(&response));
    assert_eq!(
        body["messages"][1]["content"][0],
        tool_use("tooluse_2", "lookup", json!({}))
    );
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
    let body = sent(
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
        let turn = AssistantMessage {
            content,
            origin: Some(origin.clone()),
            stop: Some(StopReason::Stop),
        };
        let body = sent(NOVA, vec![Message::user("q"), Message::Assistant(turn)]);
        let blocks = body["messages"][1]["content"].as_array().expect("content");
        assert_eq!(blocks.len(), sent_items + 1, "{body}");
    }
}
