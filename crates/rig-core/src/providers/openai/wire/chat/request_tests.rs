//! The Chat request built as JSON: the shape of each message and part, the
//! tool and format fields, and the passthrough's precedence.

use serde_json::{Value, json};

use super::Chat;
use crate::completion::{CompletionRequest, Document, Message, ToolDefinition};
use crate::message::{
    AssistantContent, AssistantMessage, CallId, DocumentMediaType, DocumentSourceKind, Image,
    ImageMediaType, ToolCall, ToolFunction, ToolName, ToolResult, ToolResultContent, UserContent,
};
use crate::providers::openai::wire::{GROQ, OPENAI, OpenAIConfig};
use crate::test_utils::json_body;
use crate::wire::{Mode, Operation, Wire};

fn wire() -> Chat {
    OpenAIConfig::new("key").chat("gpt-4o-mini")
}

fn name(name: &str) -> ToolName {
    ToolName::new(name).expect("a tool name")
}

fn tool(tool: &str) -> ToolDefinition {
    ToolDefinition::new(
        name(tool),
        "a tool",
        json!({"type": "object", "properties": {}}),
    )
}

/// The body `request` sends on `wire`, prepared as the driver prepares it.
fn body_on(wire: &Chat, request: CompletionRequest) -> Value {
    let request = crate::operation::Completion::prepare(request, &wire.describe())
        .expect("the request prepares");
    json_body(&wire.encode(request, Mode::Unary).expect("encodes").request)
}

fn body(request: CompletionRequest) -> Value {
    body_on(&wire(), request)
}

fn result(call: &str, content: Vec<ToolResultContent>) -> UserContent {
    UserContent::ToolResult(ToolResult {
        call: CallId::from_wire(call),
        name: name("tool"),
        content,
        is_error: false,
    })
}

fn calling(calls: &[&str]) -> Message {
    Message::Assistant(AssistantMessage::new(
        calls
            .iter()
            .map(|id| AssistantContent::tool_call(*id, name("tool"), json!({})))
            .collect(),
    ))
}

fn with_tool(history: Vec<Message>) -> CompletionRequest {
    let mut request = CompletionRequest::from(history);
    request.tools = vec![tool("tool")];
    request
}

#[test]
fn the_request_model_overrides_the_wire_model() {
    let request = CompletionRequest::new("Hello").model("gpt-4.1".to_owned());
    assert_eq!(body(request)["model"], "gpt-4.1");
    assert_eq!(
        body(CompletionRequest::new("Hello"))["model"],
        "gpt-4o-mini"
    );
}

/// The wire refuses a `tool_choice` beside an empty `tools`: a turn that
/// advertises none carries no choice.
#[test]
fn tool_choice_is_dropped_when_no_tool_is_advertised() {
    let request = |tools: Vec<ToolDefinition>| {
        CompletionRequest::new("Hello")
            .tool_choice(crate::message::ToolChoice::None)
            .tools(tools)
    };
    assert!(body(request(vec![])).get("tool_choice").is_none());
    assert_eq!(body(request(vec![tool("add")]))["tool_choice"], "none");
}

/// The documents open the first user message after the leading system
/// messages, so roles keep alternating.
#[test]
fn documents_open_the_first_user_message() {
    let request = CompletionRequest::new("Prompt")
        .message(Message::system("System prompt"))
        .message(Message::user("Earlier user turn"))
        .message(Message::assistant("Earlier assistant turn"))
        .document(Document {
            id: "doc1".to_owned(),
            text: "Document text.".to_owned(),
            additional_props: Default::default(),
        });
    let body = body(request);
    let messages = body["messages"].as_array().expect("messages");
    let roles: Vec<&str> = messages
        .iter()
        .filter_map(|message| message["role"].as_str())
        .collect();
    assert_eq!(roles, ["system", "user", "assistant", "user"], "{body}");
    let first = messages[1].to_string();
    assert!(
        first.find("<file id: doc1>") < first.find("Earlier user turn"),
        "{body}"
    );
}

/// A system message's content is one text part; a lone user text is a
/// plain string; a user message keeps its parts in order around results.
#[test]
fn messages_take_their_wire_shapes() {
    let history = vec![
        Message::system("Be brief."),
        Message::user("q"),
        calling(&["call-id"]),
        Message::User {
            content: vec![
                UserContent::text("before"),
                result("call-id", vec![ToolResultContent::text("tool output")]),
                UserContent::text("after"),
            ],
        },
    ];
    let body = body(with_tool(history));
    assert_eq!(
        body["messages"][0],
        json!({"role": "system", "content": [{"type": "text", "text": "Be brief."}]})
    );
    assert_eq!(body["messages"][1], json!({"role": "user", "content": "q"}));
    let roles: Vec<&str> = body["messages"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|message| message["role"].as_str())
        .collect();
    assert_eq!(
        roles,
        ["system", "user", "assistant", "tool", "user"],
        "{body}"
    );
    assert_eq!(body["messages"][3]["content"], "tool output");
}

/// A result's parts are one string, joined, unless the wire asks for arrays
/// or a part is an image, which has no string form.
#[test]
fn a_result_is_a_string_or_its_parts() {
    let parts = vec![
        ToolResultContent::text("first"),
        ToolResultContent::json(json!({"status": "ok"})),
    ];
    let history = vec![
        Message::user("q"),
        calling(&["call-id"]),
        Message::User {
            content: vec![result("call-id", parts)],
        },
    ];
    let string = body(with_tool(history.clone()));
    assert_eq!(
        string["messages"][2]["content"], "first\n{\"status\":\"ok\"}",
        "{string}"
    );
    let array = body_on(&wire().with_tool_result_array_content(), with_tool(history));
    assert_eq!(
        array["messages"][2]["content"],
        json!([
            {"type": "text", "text": "first"},
            {"type": "text", "text": "{\"status\":\"ok\"}"}
        ]),
        "array mode keeps each part (#2201): {array}"
    );

    let image = ToolResultContent::image_base64("iVBORw0KGgo=", Some(ImageMediaType::PNG), None);
    let history = vec![
        Message::user("q"),
        calling(&["call-id"]),
        Message::User {
            content: vec![result("call-id", vec![image])],
        },
    ];
    let llamacpp = OpenAIConfig::with_key(&crate::providers::openai::wire::LLAMACPP, "")
        .chat("Qwen3-VL-2B-Instruct-Q8_0");
    let sent = body_on(&llamacpp, with_tool(history));
    assert_eq!(
        sent["messages"][2]["content"][0]["image_url"]["url"], "data:image/png;base64,iVBORw0KGgo=",
        "{sent}"
    );
}

/// User parts: a PDF as a file part with its data or the URL that names
/// it, a file id, an image with `detail: auto` by default, a video by URL,
/// and a string document as text.
#[test]
fn user_parts_take_their_wire_shapes() {
    let document = |data: DocumentSourceKind, media_type| {
        UserContent::Document(crate::message::Document {
            data,
            media_type,
            additional_params: None,
        })
    };
    let content = vec![
        UserContent::text("look"),
        document(
            DocumentSourceKind::Base64("JVBERi0xLjQK".into()),
            Some(DocumentMediaType::PDF),
        ),
        document(
            DocumentSourceKind::Url("https://example.com/x.pdf".into()),
            Some(DocumentMediaType::PDF),
        ),
        document(DocumentSourceKind::FileId("file_abc".into()), None),
        UserContent::Image(Image {
            data: DocumentSourceKind::Base64("iVBORw0KGgo=".into()),
            media_type: Some(ImageMediaType::PNG),
            ..Image::default()
        }),
        UserContent::video_url("https://example.com/v.mp4".to_owned(), None),
    ];
    let openrouter = OpenAIConfig::with_key(&crate::providers::openai::wire::OPENROUTER, "key")
        .chat("google/gemini-2.5-flash");
    let sent = body_on(
        &openrouter,
        CompletionRequest::from(vec![Message::User { content }]),
    );
    assert_eq!(
        sent["messages"][0]["content"],
        json!([
            {"type": "text", "text": "look"},
            {"type": "file", "file": {"file_data": "data:application/pdf;base64,JVBERi0xLjQK",
                "filename": "document.pdf"}},
            {"type": "file", "file": {"file_data": "https://example.com/x.pdf",
                "filename": "document.pdf"}},
            {"type": "text", "text": "(document omitted: the provider cannot receive it in this form)"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo=",
                "detail": "auto"}},
            {"type": "video_url", "video_url": {"url": "https://example.com/v.mp4"}},
        ]),
        "OpenRouter takes no file ids: {sent}"
    );
    let file_id = body(CompletionRequest::from(vec![Message::User {
        content: vec![
            UserContent::text("read"),
            document(DocumentSourceKind::FileId("file_abc".into()), None),
        ],
    }]));
    assert_eq!(
        file_id["messages"][0]["content"][1],
        json!({"type": "file", "file": {"file_id": "file_abc"}})
    );
}

/// The legacy cap stays where the endpoint wants it; OpenAI's reasoning
/// families get `max_completion_tokens`, a caller's own modern cap wins,
/// and nothing else moves.
#[test]
fn the_output_cap_is_spelled_for_the_model() {
    let request = |model: &str, max_tokens: Option<u64>, params: Option<Value>| {
        let mut request = CompletionRequest::new("Hello")
            .model(model.to_owned())
            .max_tokens(max_tokens);
        request.additional_params = params;
        body(request)
    };
    let legacy = request("gpt-4o-mini", Some(4096), None);
    assert_eq!(legacy["max_tokens"], 4096);
    assert!(legacy.get("max_completion_tokens").is_none());
    let modern = request("gpt-5", Some(4096), Some(json!({"top_p": 0.5})));
    assert_eq!(modern["max_completion_tokens"], 4096);
    assert!(modern.get("max_tokens").is_none());
    assert_eq!(modern["top_p"], 0.5);
    let theirs = request(
        "gpt-5",
        Some(4096),
        Some(json!({"max_completion_tokens": 48})),
    );
    assert_eq!(theirs["max_completion_tokens"], 48);
    let upgraded = request("gpt-5", None, Some(json!({"max_tokens": 48})));
    assert_eq!(upgraded["max_completion_tokens"], 48);
    assert!(upgraded.get("max_tokens").is_none());
    let none = request("gpt-5", None, None);
    assert!(none.get("max_tokens").is_none() && none.get("max_completion_tokens").is_none());
}

/// A mixed `additional_params.tools` array splits by shape: function tools
/// join the typed `tools` (left in the flattened params they would replace
/// them), the rest stay behind, and Groq folds its native ones into
/// `compound_custom.enabled_tools`.
#[test]
fn passthrough_function_tools_join_the_typed_ones() {
    let request = || {
        CompletionRequest::new("Hello")
            .tools(vec![tool("builder_tool")])
            .additional_params(json!({"tools": [
                {"type": "function", "function": {"name": "params_tool",
                    "description": "from additional_params", "parameters": {"type": "object"}}},
                {"type": "browser_search"},
            ]}))
    };
    let groq = OpenAIConfig::with_key(&GROQ, "key").chat("compound-beta");
    let sent = body_on(&groq, request());
    let names: Vec<&str> = sent["tools"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|tool| tool["function"]["name"].as_str())
        .collect();
    assert_eq!(names, ["builder_tool", "params_tool"], "{sent}");
    assert_eq!(
        sent["compound_custom"],
        json!({"enabled_tools": [{"type": "browser_search"}]})
    );
    let openai = body(request());
    assert_eq!(
        openai["tools"],
        json!([{"type": "browser_search"}]),
        "{openai}"
    );
}

/// `additional_params` is flattened in after the typed fields, so a
/// passthrough key overrides the typed value.
#[test]
fn additional_params_override_typed_fields() {
    let request = CompletionRequest::new("hi")
        .temperature(0.1)
        .additional_params(json!({"temperature": 0.9, "top_p": 0.5}));
    let body = body(request);
    assert_eq!(body["temperature"], 0.9, "{body}");
    assert_eq!(body["top_p"], 0.5, "{body}");
}

/// A structured format waits for the first tool result unless the dialect
/// takes both at once, so the model can still call a tool.
#[test]
fn the_response_format_waits_for_a_tool_result() {
    let schema: schemars::Schema = serde_json::from_value(json!({
        "title": "WeatherResponse",
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"]
    }))
    .expect("a schema");
    let first = body(
        CompletionRequest::new("Weather in London?")
            .output_schema(schema.clone())
            .tools(vec![tool("tool")]),
    );
    assert!(first.get("response_format").is_none(), "{first}");
    let mut later = with_tool(vec![
        Message::user("Weather in London?"),
        calling(&["call_1"]),
        Message::User {
            content: vec![result("call_1", vec![ToolResultContent::text("fire")])],
        },
    ]);
    later.output_schema = Some(schema);
    let later = body(later);
    assert_eq!(
        later["response_format"]["json_schema"]["name"],
        "WeatherResponse"
    );
}

/// Calls rig issued ids for replay as `tool-<n>` aliases no provider id
/// takes, each result naming its own call, across turns and around a user
/// text split.
#[test]
fn rig_issued_ids_are_spelled_once_per_request() {
    let issued = || {
        ToolCall::new(
            CallId::from_wire(""),
            ToolFunction::new(name("tool"), json!({})),
        )
    };
    let (generated, later) = (issued(), issued());
    let real = ToolCall::from_wire("tool-0", generated.function.clone());
    let answer = |call: &ToolCall| result(&call.id.wire(), vec![ToolResultContent::text("r")]);
    let mut first_result = answer(&generated);
    if let UserContent::ToolResult(result) = &mut first_result {
        result.call = generated.id.clone();
    }
    let mut later_result = answer(&later);
    if let UserContent::ToolResult(result) = &mut later_result {
        result.call = later.id.clone();
    }
    let history = vec![
        Message::user("q"),
        Message::Assistant(AssistantMessage::new(vec![
            AssistantContent::ToolCall(generated.clone()),
            AssistantContent::ToolCall(real.clone()),
        ])),
        Message::User {
            content: vec![
                answer(&real),
                UserContent::text("between results"),
                first_result,
            ],
        },
        Message::Assistant(AssistantMessage::new(vec![AssistantContent::ToolCall(
            later.clone(),
        )])),
        Message::User {
            content: vec![later_result],
        },
    ];
    let body = body(with_tool(history));
    let messages = body["messages"].as_array().expect("messages");
    let first = &messages[1]["tool_calls"][0]["id"];
    let genuine = &messages[1]["tool_calls"][1]["id"];
    let later = &messages[5]["tool_calls"][0]["id"];
    assert_eq!(genuine, "tool-0", "{body}");
    assert_ne!(first, genuine);
    assert_ne!(later, first);
    assert_ne!(later, genuine);
    assert_eq!(&messages[2]["tool_call_id"], genuine, "{body}");
    assert_eq!(&messages[3]["tool_call_id"], first, "{body}");
    assert_eq!(&messages[6]["tool_call_id"], later, "{body}");
}

/// A reasoning-only turn is no message (pi skips one with neither content
/// nor calls), and an unedited block's own field replays beside the text.
#[test]
fn reasoning_replays_under_its_own_field() {
    let origin = crate::message::Origin::new("openai.chat", "openai", "gpt-4o-mini");
    let turn = |content| {
        Message::Assistant(AssistantMessage {
            content,
            origin: Some(origin.clone()),
            stop: Some(crate::message::StopReason::Stop),
        })
    };
    let reasoning =
        AssistantContent::reasoning("hidden").with_native(json!({"reasoning_content": "hidden"}));
    let alone = body(CompletionRequest::from(vec![
        Message::user("q"),
        turn(vec![reasoning.clone()]),
        Message::user("again"),
    ]));
    assert_eq!(
        alone["messages"],
        json!([{"role": "user", "content": "q"}, {"role": "user", "content": "again"}])
    );
    let beside = body(CompletionRequest::from(vec![
        Message::user("q"),
        turn(vec![reasoning, AssistantContent::text("visible")]),
        Message::user("again"),
    ]));
    assert_eq!(
        beside["messages"][1],
        json!({"role": "assistant", "reasoning_content": "hidden", "content": "visible"})
    );
}

/// GPT-6 models that take Chat function tools only at effort `none` are
/// refused before the request is sent, naming the fix.
#[test]
fn openai_reasoning_families_refuse_tools_they_cannot_call_on_chat() {
    let request = |model: &str, params: Option<Value>| {
        let mut request = CompletionRequest::new("q").tools(vec![tool("add")]);
        request.additional_params = params;
        OpenAIConfig::with_key(&OPENAI, "key")
            .chat(model)
            .encode(request, Mode::Unary)
    };
    assert!(request(super::models::GPT_6_ASTRA, None).is_err());
    assert!(request(super::models::GPT_6_SOL, None).is_err());
    assert!(
        request(
            super::models::GPT_6_SOL,
            Some(json!({"reasoning_effort": "none"}))
        )
        .is_ok()
    );
}
