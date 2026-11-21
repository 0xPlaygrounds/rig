//! The Chat request built as JSON: the shape of each message and part, the
//! tool and format fields, and the passthrough's precedence.

use serde_json::{Value, json};

use super::Chat;
use crate::catalog::{Catalog, ModelFacts, ModelSpec};
use crate::completion::{CompletionRequest, Message, ToolDefinition};
use crate::message::{
    AssistantContent, AssistantMessage, AudioMediaType, CallId, DocumentMediaType,
    DocumentSourceKind, Image, ImageMediaType, ToolName, ToolResult, ToolResultContent,
    UserContent, VideoMediaType,
};
use crate::providers::openai::wire::{AZURE, GROQ, OPENAI, OpenAIConfig};
use crate::providers::registry::ProviderId;
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

fn video_spec(provider: &str, model: &str, enabled: bool) -> anyhow::Result<ModelSpec> {
    let provider = ProviderId::catalog(provider)
        .ok_or_else(|| anyhow::anyhow!("{provider} is not a registered provider"))?;
    let mut spec = ModelSpec::new(provider, model);
    spec.input.video = enabled;
    Ok(spec)
}

fn video_request() -> CompletionRequest {
    CompletionRequest::from(vec![Message::User {
        content: vec![
            UserContent::text("Describe the video."),
            UserContent::video_url("https://example.com/v.mp4".to_owned(), None),
        ],
    }])
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
    request.tools = vec![tool("tool").into()];
    request
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
        UserContent::Document(crate::message::DocumentContent {
            data: data.into(),
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

/// Video requires matching facts on OpenAI and Azure. This tests local
/// adaptation of an unlisted model, without provider traffic to record.
#[test]
fn video_parts_follow_bound_model_facts() -> anyhow::Result<()> {
    let default = body_on(&wire(), video_request());
    anyhow::ensure!(
        default.pointer("/messages/0/content/1") == Some(&omitted_video()),
        "the default OpenAI model still omits video"
    );

    let model = "gateway-video-model";
    for dialect in [&OPENAI, &AZURE] {
        let bound =
            |provider, model, enabled| video_spec(provider, model, enabled).map(ModelFacts::new);
        for (case, facts, enabled) in [
            ("absent", ModelFacts::default(), false),
            ("enabled", bound(dialect.name, model, true)?, true),
            ("disabled", bound(dialect.name, model, false)?, false),
            (
                "other model",
                bound(dialect.name, "other-model", true)?,
                false,
            ),
            ("other provider", bound("anthropic", model, true)?, false),
        ] {
            let chat = OpenAIConfig::with_key(dialect, "key")
                .chat(model)
                .with_facts(facts);
            let expected = if enabled {
                json!({"type": "video_url", "video_url": {"url": "https://example.com/v.mp4"}})
            } else {
                omitted_video()
            };
            for mode in [Mode::Unary, Mode::Streaming] {
                let request =
                    crate::operation::Completion::prepare(video_request(), &chat.describe())?;
                let sent = json_body(&chat.encode(request, mode)?.request);
                anyhow::ensure!(
                    sent.pointer("/messages/0/content/1") == Some(&expected),
                    "{} with {case} facts in {mode:?}: {sent}",
                    dialect.name
                );
            }
        }
    }
    Ok(())
}

fn omitted_video() -> Value {
    json!({"type": "text", "text": "(video omitted: the provider cannot receive it in this form)"})
}

/// Video facts permit typed data but do not make untyped data encodable.
/// This checks local preparation and encoding without provider traffic.
#[test]
fn video_parts_require_a_url_or_typed_base64() -> anyhow::Result<()> {
    let model = "gateway-video-model";
    for dialect in [&OPENAI, &AZURE] {
        let chat = OpenAIConfig::with_key(dialect, "key")
            .chat(model)
            .with_facts(ModelFacts::new(video_spec(dialect.name, model, true)?));
        let request = CompletionRequest::from(vec![Message::User {
            content: vec![
                UserContent::text("Describe the videos."),
                UserContent::video_base64("AAAA", Some(VideoMediaType::MP4)),
                UserContent::video_base64("AAAA", None),
            ],
        }]);
        let sent = body_on(&chat, request);
        anyhow::ensure!(
            sent.pointer("/messages/0/content")
                == Some(&json!([
                    {"type": "text", "text": "Describe the videos."},
                    {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,AAAA"}},
                    omitted_video(),
                ])),
            "{} keeps the video source requirements: {sent}",
            dialect.name
        );
    }
    Ok(())
}

/// Bound facts take precedence and request overrides consult the catalog.
/// This tests local model selection without provider traffic to record.
#[test]
fn video_parts_follow_the_requested_models_facts() -> anyhow::Result<()> {
    let model = "gateway-video-model";
    let other = "another-video-model";
    for dialect in [&OPENAI, &AZURE] {
        let mut catalog = Catalog::builtin().clone();
        catalog.insert(video_spec(dialect.name, model, true)?);
        catalog.insert(video_spec(dialect.name, other, true)?);
        let facts = ModelFacts::new(video_spec(dialect.name, model, false)?).with_catalog(&catalog);
        let chat = OpenAIConfig::with_key(dialect, "key")
            .chat(model)
            .with_facts(facts);
        for (requested, enabled) in [(model, false), (other, true), ("unlisted-model", false)] {
            let sent = body_on(&chat, video_request().model(requested));
            let expected = if enabled {
                json!({"type": "video_url", "video_url": {"url": "https://example.com/v.mp4"}})
            } else {
                omitted_video()
            };
            anyhow::ensure!(
                sent.pointer("/messages/0/content/1") == Some(&expected),
                "{} requested {requested}: {sent}",
                dialect.name
            );
        }
    }
    Ok(())
}

/// Changing video facts must invalidate a prepared conversation's prefix.
/// This exercises the driver's local preparation cache without HTTP traffic.
#[test]
fn video_facts_refresh_prepared_history() -> anyhow::Result<()> {
    let model = "gateway-video-model";
    let request = CompletionRequest::from(vec![
        Message::User {
            content: vec![
                UserContent::text("Describe the video."),
                UserContent::video_url("https://example.com/v.mp4", None),
            ],
        },
        Message::assistant("A person is walking."),
        Message::user("Describe the scene in more detail."),
    ]);
    for dialect in [&OPENAI, &AZURE] {
        let prepared = crate::wire::Prepared::default();
        for enabled in [false, true, false] {
            let chat = OpenAIConfig::with_key(dialect, "key")
                .chat(model)
                .with_facts(ModelFacts::new(video_spec(dialect.name, model, enabled)?));
            let descriptor = crate::wire::Descriptor {
                prepared: Some(&prepared),
                ..chat.describe()
            };
            let request = crate::operation::Completion::prepare(request.clone(), &descriptor)?;
            let sent = json_body(&chat.encode(request, Mode::Unary)?.request);
            let expected = if enabled {
                json!({"type": "video_url", "video_url": {"url": "https://example.com/v.mp4"}})
            } else {
                omitted_video()
            };
            anyhow::ensure!(
                sent.pointer("/messages/0/content/1") == Some(&expected),
                "{} with video={enabled} must refresh its prepared prefix: {sent}",
                dialect.name
            );
        }
    }
    Ok(())
}

/// Other dialects keep their existing video encoding regardless of the fact.
/// This compares locally encoded requests without provider traffic.
#[test]
fn video_facts_preserve_other_chat_dialects() -> anyhow::Result<()> {
    let model = "gateway-video-model";
    for dialect in crate::providers::openai::wire::all()
        .filter(|dialect| ![OPENAI.name, AZURE.name].contains(&dialect.name))
    {
        let chat = OpenAIConfig::with_key(dialect, "key").chat(model);
        for mode in [Mode::Unary, Mode::Streaming] {
            let body = |chat: &Chat| -> anyhow::Result<Value> {
                let request =
                    crate::operation::Completion::prepare(video_request(), &chat.describe())?;
                Ok(json_body(&chat.encode(request, mode)?.request))
            };
            let default = body(&chat)?;
            for enabled in [false, true] {
                let configured = chat.clone().with_facts(ModelFacts::new(video_spec(
                    dialect.name,
                    model,
                    enabled,
                )?));
                let sent = body(&configured)?;
                anyhow::ensure!(
                    sent == default,
                    "{} changed with video={enabled} in {mode:?}: {sent}",
                    dialect.name
                );
            }
        }
    }
    Ok(())
}

/// The video fact leaves text, image, audio and document encoding unchanged.
/// This compares local request bodies without provider traffic.
#[test]
fn video_facts_preserve_non_video_content() -> anyhow::Result<()> {
    let model = "gateway-video-model";
    let request = CompletionRequest::from(vec![Message::User {
        content: vec![
            UserContent::text("Describe the media."),
            UserContent::image_base64("iVBORw0KGgo=", Some(ImageMediaType::PNG), None),
            UserContent::audio_base64("AAAA", Some(AudioMediaType::MP3)),
            UserContent::document_base64("JVBERi0xLjQK", Some(DocumentMediaType::PDF)),
        ],
    }]);
    for dialect in [&OPENAI, &AZURE] {
        for mode in [Mode::Unary, Mode::Streaming] {
            let body = |enabled| -> anyhow::Result<Value> {
                let mut spec = video_spec(dialect.name, model, enabled)?;
                spec.input.image = true;
                let chat = OpenAIConfig::with_key(dialect, "key")
                    .chat(model)
                    .with_facts(ModelFacts::new(spec));
                let request =
                    crate::operation::Completion::prepare(request.clone(), &chat.describe())?;
                Ok(json_body(&chat.encode(request, mode)?.request))
            };
            let disabled = body(false)?;
            let enabled = body(true)?;
            anyhow::ensure!(
                disabled == enabled,
                "{} changed non-video content in {mode:?}: {disabled} != {enabled}",
                dialect.name
            );
        }
    }
    Ok(())
}

/// OpenAI's default Responses route still omits video even with video facts.
/// This checks local routing and encoding without provider traffic.
#[test]
fn video_facts_preserve_the_default_responses_route() -> anyhow::Result<()> {
    let model = "gateway-video-model";
    let routed = OpenAIConfig::new("key")
        .completion(model)
        .with_facts(ModelFacts::new(video_spec("openai", model, true)?));
    for mode in [Mode::Unary, Mode::Streaming] {
        let request = crate::operation::Completion::prepare(video_request(), &routed.describe())?;
        let encoded = routed.encode(request, mode)?;
        let sent = json_body(&encoded.request);
        anyhow::ensure!(
            encoded.request.uri().path().ends_with("/responses"),
            "the default route changed"
        );
        anyhow::ensure!(
            sent.pointer("/input/1/content/0/text") == omitted_video().get("text"),
            "Responses changed its video handling in {mode:?}: {sent}"
        );
    }
    Ok(())
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

/// Every `additional_params.tools` entry, function or native, joins the
/// typed `tools` in one array. Groq's `gpt-oss` models take a native tool
/// there (`browser_search` checked live); `compound_custom.enabled_tools`
/// is ignored.
#[test]
fn passthrough_tools_join_the_typed_ones() {
    let request = || {
        CompletionRequest::new("Hello")
            .tools(vec![tool("builder_tool")])
            .additional_params(json!({"tools": [
                {"type": "function", "function": {"name": "params_tool",
                    "description": "from additional_params", "parameters": {"type": "object"}}},
                {"type": "browser_search"},
            ]}))
    };
    let groq = OpenAIConfig::with_key(&GROQ, "key").chat("openai/gpt-oss-20b");
    for sent in [body_on(&groq, request()), body(request())] {
        let kinds: Vec<&str> = sent["tools"]
            .as_array()
            .into_iter()
            .flatten()
            .map(|tool| {
                tool["function"]["name"]
                    .as_str()
                    .or_else(|| tool["type"].as_str())
                    .unwrap_or_default()
            })
            .collect();
        assert_eq!(
            kinds,
            ["builder_tool", "params_tool", "browser_search"],
            "{sent}"
        );
        assert!(sent.get("compound_custom").is_none(), "{sent}");
    }
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

/// Ollama's OpenAI-compatible API reads `reasoning_effort`, which the
/// `reasoning` option sends; a raw `think` is sent as written. It ignores
/// `num_ctx` and `options`, which are refused rather than dropped.
#[test]
fn ollama_compatible_bodies_send_reasoning_and_refuse_native_options() {
    let wire = crate::providers::ollama::OllamaConfig::new().completion("qwen3");
    let sent = |request: CompletionRequest| {
        let request = crate::operation::Completion::prepare(request, &wire.describe())
            .expect("the request prepares");
        wire.encode(request, Mode::Unary)
            .map(|encoded| json_body(&encoded.request))
    };
    let raw = |params: Value| CompletionRequest::new("hi").additional_params(params);
    let body = sent(raw(json!({"keep_alive": "10m"}))).expect("encodes");
    assert_eq!(body["keep_alive"], json!("10m"));
    assert!(body.get("reasoning_effort").is_none());
    for (reasoning, effort) in [
        (crate::completion::Reasoning::Off, json!("none")),
        (crate::completion::Effort::High.into(), json!("high")),
        (crate::completion::Effort::Low.into(), json!("low")),
    ] {
        let body = sent(
            CompletionRequest::new("hi")
                .options(crate::completion::GenerationOptions::default().reasoning(reasoning)),
        )
        .expect("encodes");
        assert_eq!(body["reasoning_effort"], effort, "{reasoning:?}");
        assert!(body.get("think").is_none());
    }
    let body = sent(raw(json!({"think": true}))).expect("a raw `think` is sent");
    assert_eq!(body["think"], json!(true));
    assert!(body.get("reasoning_effort").is_none());
    for (params, named) in [
        (json!({"num_ctx": 8192}), "`num_ctx`"),
        (json!({"options": {"num_ctx": 8192}}), "`options`"),
    ] {
        let refused = sent(raw(params.clone())).expect_err("the request is refused");
        assert!(refused.to_string().contains(named), "{params}: {refused}");
    }
}

/// Whether `value` holds a part or message whose text is exactly `text`.
fn carries_text(value: &Value, text: &str) -> bool {
    match value {
        Value::Object(fields) => {
            ["text", "content"]
                .iter()
                .any(|key| fields.get(*key).and_then(Value::as_str) == Some(text))
                || fields.values().any(|field| carries_text(field, text))
        }
        Value::Array(items) => items.iter().any(|item| carries_text(item, text)),
        _ => false,
    }
}

/// A base64 PDF is PDF data, as Anthropic sends it: the adapter's text
/// fallback never hands its base64 to the model as user text. A document's
/// text, including a persisted `"string"` PDF, goes as its text, as on
/// every wire.
///
/// Not a cassette test: the defect is the request body built before any
/// HTTP, which no recorded reply exercises.
#[test]
fn base64_pdf_is_data_and_document_text_is_text() -> anyhow::Result<()> {
    let pdf = "JVBERi0xLjQKJcfsj6IKMSAwIG9iago8PD4+CmVuZG9iagp0cmFpbGVyCjw8Pj4KJSVFT0YK";
    let wire = wire();
    let sent = |data: crate::message::DocumentData| {
        let request = CompletionRequest::new(Message::User {
            content: vec![
                UserContent::text("Summarize the file."),
                UserContent::Document(crate::message::DocumentContent {
                    data,
                    media_type: Some(DocumentMediaType::PDF),
                    additional_params: None,
                }),
            ],
        });
        crate::operation::Completion::prepare(request, &wire.describe())
            .map_err(anyhow::Error::from)
            .and_then(|request| Ok(json_body(&wire.encode(request, Mode::Unary)?.request)))
    };
    let body = sent(DocumentSourceKind::base64(pdf).into())?;
    anyhow::ensure!(
        !carries_text(&body, pdf),
        "the PDF's base64 goes to the model as text: {body:#}"
    );
    let legacy = serde_json::from_value(json!({"type": "string", "value": pdf}))?;
    let body = sent(legacy)?;
    anyhow::ensure!(
        carries_text(&body, pdf),
        "a text document goes as its text: {body:#}"
    );
    Ok(())
}
