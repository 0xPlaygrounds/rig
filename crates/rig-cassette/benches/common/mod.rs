//! The fixtures the rig-cassette benches share: a transport that replays
//! recorded replies with no network, each wire family's model over it, and
//! a coding session's requests.

use std::fmt::Display;
use std::future::Future;
use std::path::Path;
use std::sync::{Arc, Mutex};

use futures::StreamExt as _;
use rig_cassette::http::recorded_statuses_and_bodies;
use rig_core::DynModel;
use rig_core::completion::{
    AssistantContent, AssistantMessage, CompletionRequest, CompletionResponse, Message,
    ToolDefinition,
};
use rig_core::error::ProviderError;
use rig_core::http_client::{
    self, Bytes, HttpClientExt, LazyBody, MultipartForm, Request, Response, StreamingResponse,
};
use rig_core::message::{DocumentSourceKind, Image, ImageMediaType, Origin, StopReason};
use rig_core::operation::Completion;
use rig_core::providers::anthropic::AnthropicConfig;
use rig_core::providers::cohere::CohereConfig;
use rig_core::providers::gemini::GeminiConfig;
use rig_core::providers::ollama::OllamaConfig;
use rig_core::providers::openai::OpenAIConfig;
use rig_core::wasm_compat::WasmCompatSend;
use serde_json::json;

/// `result`'s value, or the process ends saying what failed: a bench whose
/// input cannot be made measures nothing.
pub fn or_exit<T, E: Display>(result: Result<T, E>, what: &str) -> T {
    result.unwrap_or_else(|error| {
        eprintln!("{what}: {error}");
        std::process::exit(1)
    })
}

/// Ends the process saying `why` unless `holds`.
pub fn ensure(holds: bool, why: impl FnOnce() -> String) {
    if !holds {
        eprintln!("{}", why());
        std::process::exit(1);
    }
}

// The transport.

/// Answers every streaming request with `reply`'s chunks, after turning the
/// request body into bytes as a real client does; with no reply it fails
/// the request once the body is built. `sent` is the last body.
#[derive(Clone, Default)]
pub struct Replay {
    pub reply: Option<Arc<Vec<Bytes>>>,
    pub sent: Arc<Mutex<Bytes>>,
}

fn refused() -> http_client::Error {
    http_client::Error::non_success_with_details(
        http_client::StatusCode::NOT_IMPLEMENTED,
        http_client::HeaderMap::new(),
        String::new(),
    )
}

impl HttpClientExt for Replay {
    fn send<T, U>(
        &self,
        _req: Request<T>,
    ) -> impl Future<Output = http_client::Result<Response<LazyBody<U>>>> + WasmCompatSend + 'static
    where
        T: Into<Bytes> + WasmCompatSend,
        U: From<Bytes> + WasmCompatSend + 'static,
    {
        std::future::ready(Err(refused()))
    }

    fn send_multipart<U>(
        &self,
        _req: Request<MultipartForm>,
    ) -> impl Future<Output = http_client::Result<Response<LazyBody<U>>>> + WasmCompatSend + 'static
    where
        U: From<Bytes> + WasmCompatSend + 'static,
    {
        std::future::ready(Err(refused()))
    }

    fn send_streaming<T>(
        &self,
        req: Request<T>,
    ) -> impl Future<Output = http_client::Result<StreamingResponse>> + WasmCompatSend
    where
        T: Into<Bytes> + WasmCompatSend,
    {
        let body: Bytes = req.into_body().into();
        if let Ok(mut sent) = self.sent.lock() {
            *sent = body;
        }
        let reply = self.reply.clone();
        async move {
            let reply = reply.ok_or_else(refused)?;
            let chunks: Vec<http_client::Result<Bytes>> = reply.iter().cloned().map(Ok).collect();
            let stream: http_client::BoxedStream = Box::pin(futures::stream::iter(chunks));
            Response::builder()
                .status(http_client::StatusCode::OK)
                .header("content-type", "text/event-stream")
                .body(stream)
                .map_err(http_client::Error::Protocol)
        }
    }
}

// The wire families.

/// The families every per-family bench runs.
pub const FAMILIES: [&str; 6] = [
    "responses",
    "anthropic",
    "chat",
    "gemini",
    "cohere",
    "ollama",
];

/// `family`'s completion model on `transport`, as the cassettes recorded it.
pub fn model(family: &str, transport: Replay) -> DynModel<Completion> {
    match family {
        "responses" => OpenAIConfig::new("bench-key")
            .connect(transport)
            .responses("gpt-5.4")
            .erase(),
        "anthropic" => AnthropicConfig::new("bench-key")
            .connect(transport)
            .completion("claude-sonnet-4-6")
            .erase(),
        "chat" => OpenAIConfig::new("bench-key")
            .connect(transport)
            .chat("gpt-4o")
            .erase(),
        "cohere" => CohereConfig::new("bench-key")
            .connect(transport)
            .completion("command-a-plus-05-2026")
            .erase(),
        "ollama" => OllamaConfig::new()
            .connect(transport)
            .native_completion("qwen3:4b")
            .erase(),
        _ => GeminiConfig::new("bench-key")
            .connect(transport)
            .completion("gemini-2.5-flash")
            .erase(),
    }
}

/// `family`'s recorded streamed text reply: its cassette, and how to tell a
/// text delta event from the others.
fn recording(family: &str) -> (&'static str, &'static str, fn(&str) -> bool) {
    match family {
        "responses" => (
            "chatgpt",
            "codex_sessions/reasoning_session_two_tool_calls_streaming",
            |event| event.contains("\"response.output_text.delta\""),
        ),
        "anthropic" => (
            "anthropic",
            "prompt_caching/matrix_automatic_prefix_unset_tools_streaming",
            |event| event.contains("\"text_delta\""),
        ),
        "chat" => (
            "openai",
            "streaming_grammar_chat/long_text_stream",
            |event| {
                event.contains("\"delta\":{\"content\":\"")
                    && event.contains("\"finish_reason\":null")
            },
        ),
        "cohere" => ("cohere", "streaming/streaming_smoke", |event| {
            event.contains("\"delta\":{\"content\":\"") && event.contains("\"finish_reason\":null")
        }),
        "ollama" => ("ollama", "native/inline_think_streamed", |event| {
            event.contains("\"done\":false")
        }),
        _ => ("gemini", "corpus_breadth/text_delta_stop", |event| {
            event.contains("\"parts\":[{\"text\"") && !event.contains("finishReason")
        }),
    }
}

/// `body`'s events: SSE events with the blank line that ends each, or
/// NDJSON records (Ollama's) with their newline.
fn events(body: &str) -> Vec<&str> {
    let end = ["\r\n\r\n", "\n\n", "\n"]
        .into_iter()
        .find(|end| body.contains(end))
        .unwrap_or("\n");
    body.split_inclusive(end).collect()
}

/// `family`'s recorded text reply with its deltas repeated to `deltas`: what
/// comes before the first delta, the deltas, and what follows the last.
pub fn long_reply(family: &str, deltas: usize) -> String {
    let (provider, scenario, is_delta) = recording(family);
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/cassettes");
    let bodies = recorded_statuses_and_bodies(&root, provider, scenario);
    let body = bodies
        .into_iter()
        .map(|(_, body)| body)
        .find(|body| events(body).iter().any(|event| is_delta(event)));
    let Some(body) = body else {
        eprintln!("{provider}/{scenario} has no streamed text");
        std::process::exit(1)
    };
    let all = events(&body);
    let recorded: Vec<&str> = all
        .iter()
        .copied()
        .filter(|event| is_delta(event))
        .collect();
    let first = all.iter().position(|event| is_delta(event)).unwrap_or(0);
    let last = all.iter().rposition(|event| is_delta(event)).unwrap_or(0);
    let mut reply: String = all.iter().take(first).copied().collect();
    reply.extend(recorded.iter().cycle().take(deltas).copied());
    reply.extend(all.iter().skip(last + 1).copied());
    reply
}

/// `reply` as a network delivers it: one SSE event per chunk (`size` 0) or
/// in chunks of `size` bytes.
pub fn chunked(reply: &str, size: usize) -> Arc<Vec<Bytes>> {
    let chunks = match size {
        0 => events(reply)
            .into_iter()
            .map(|event| Bytes::copy_from_slice(event.as_bytes()))
            .collect(),
        _ => reply
            .as_bytes()
            .chunks(size)
            .map(Bytes::copy_from_slice)
            .collect(),
    };
    Arc::new(chunks)
}

/// The model of `family` answering with `reply`.
pub fn replaying(family: &str, reply: Arc<Vec<Bytes>>) -> DynModel<Completion> {
    let transport = Replay {
        reply: Some(reply),
        ..Replay::default()
    };
    model(family, transport)
}

/// `request` streamed from `model` and drained: the items and the response.
pub fn drain(
    model: &DynModel<Completion>,
    request: CompletionRequest,
) -> Result<(usize, CompletionResponse), ProviderError> {
    futures::executor::block_on(async {
        let mut stream = model.stream(request)?;
        let mut items = 0;
        while let Some(item) = stream.next().await {
            item?;
            items += 1;
        }
        Ok((items, stream.finish().await?))
    })
}

// Requests.

const SYSTEM: &str = "You are a coding agent working in a Rust repository. Read files before \
    changing them, keep changes small, and run the tests after each change.";

/// Twenty tools shaped like the coding tools' definitions.
fn tools() -> Vec<Arc<ToolDefinition>> {
    (0..20)
        .map(|i| {
            let tool = json!({
                "name": format!("tool_{i}"),
                "description": format!("Tool number {i}: reads, writes or searches files in the \
                    workspace. It takes a path and options and returns text."),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "A path relative to the workspace root."},
                        "offset": {"type": "integer", "description": "The first line to read."},
                        "limit": {"type": "integer", "description": "How many lines to read."},
                        "pattern": {"type": "string", "description": "A regular expression."},
                        "options": {"type": "object", "properties": {
                            "case": {"type": "boolean"}, "hidden": {"type": "boolean"}}}
                    },
                    "required": ["path"]
                }
            });
            or_exit(serde_json::from_value(tool), "a tool definition")
        })
        .collect()
}

/// A request of the system prompt, `history`, a question and the tools.
pub fn request(history: &[Message]) -> CompletionRequest {
    let question = Message::user("One more question: what does handle_1_1 return?");
    let mut request = CompletionRequest::new(question.clone());
    request.chat_history = std::iter::once(Message::System {
        content: SYSTEM.to_owned(),
    })
    .chain(history.iter().cloned())
    .chain([question])
    .map(Arc::new)
    .collect();
    request.tools = tools();
    request
}

/// Base64 text of `bytes` arbitrary bytes.
pub fn base64(bytes: usize) -> String {
    use base64::Engine as _;
    let raw: Vec<u8> = (0..bytes)
        .map(|i| (i.wrapping_mul(31) % 251) as u8)
        .collect();
    base64::engine::general_purpose::STANDARD.encode(raw)
}

/// The opaque part of the history's reasoning items, repeated.
pub const SIGNATURE: &str = "c2lnbmF0dXJl";

/// What the history's assistant turns carry from `family`'s replies: who
/// made them, and the provider item of each block.
pub struct Native {
    family: &'static str,
    origin: Origin,
}

impl Native {
    /// `family`'s origin, as decoding its recorded reply to [`request`]
    /// stamps it, so the history's turns are this model's own.
    pub fn of(family: &'static str) -> Self {
        let model = replaying(family, chunked(&long_reply(family, 4), 0));
        let (_, response) = or_exit(drain(&model, request(&[])), "decoding the recorded reply");
        Self {
            family,
            origin: response.origin,
        }
    }

    /// No provider items: hand-built turns.
    pub fn of_none() -> Self {
        Self {
            family: "none",
            origin: Origin::new("bench.api", "bench", "bench-model"),
        }
    }

    /// No provider items, and call ids rig issued, as for a provider that
    /// sends none.
    pub fn issuing_ids() -> Self {
        Self {
            family: "rig",
            ..Self::of_none()
        }
    }

    /// The id of turn `i`'s call: the provider's, or one rig issued.
    fn call_id(&self, i: usize) -> serde_json::Value {
        match self.family {
            "rig" => json!({"local": format!("{i:08x}-0000-4000-8000-000000000000")}),
            _ => json!({"provider": format!("call-{i}")}),
        }
    }

    /// `block` with the provider item `family` decodes it from.
    fn item(&self, block: AssistantContent, i: usize) -> AssistantContent {
        let signature = SIGNATURE.repeat(80);
        let item = match (self.family, &block) {
            ("responses", AssistantContent::Reasoning(reasoning)) => json!({
                "type": "reasoning", "id": format!("rs_{i}"), "summary": [],
                "content": [{"type": "reasoning_text", "text": reasoning.text}],
                "encrypted_content": signature}),
            ("responses", AssistantContent::ToolCall(call)) => json!({
                "type": "function_call", "id": format!("fc_{i}"), "call_id": format!("call-{i}"),
                "name": call.function.name.as_str(), "status": "completed",
                "arguments": json!(call.function.arguments).to_string()}),
            ("responses", AssistantContent::Text(text)) => json!({
                "type": "message", "id": format!("msg_{i}"), "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text.text, "annotations": []}]}),
            ("anthropic", AssistantContent::Reasoning(reasoning)) => json!({
                "type": "thinking", "thinking": reasoning.text, "signature": signature}),
            ("anthropic", AssistantContent::ToolCall(call)) => json!({
                "type": "tool_use", "id": format!("call-{i}"),
                "name": call.function.name.as_str(), "input": call.function.arguments}),
            ("anthropic", AssistantContent::Text(text)) => {
                json!({"type": "text", "text": text.text})
            }
            ("gemini", AssistantContent::Reasoning(reasoning)) => json!({
                "text": reasoning.text, "thought": true, "thoughtSignature": signature}),
            ("gemini", AssistantContent::ToolCall(call)) => json!({
                "functionCall": {"name": call.function.name.as_str(), "args": call.function.arguments},
                "thoughtSignature": signature}),
            ("gemini", AssistantContent::Text(text)) => json!({"text": text.text}),
            _ => return block,
        };
        block.with_native(item)
    }
}

/// `turns` turns of a coding session, 4 messages each: a question (five of
/// them with a 500 KB image, spread out), a reasoning block with a `read`
/// call, its 20 KB result, and the answer. The assistant turns are
/// `native`'s model's own, with their provider items.
pub fn history(turns: usize, native: &Native, images: bool) -> Vec<Message> {
    let image = base64(375 * 1024);
    let every = (turns / 5).max(1);
    let mut messages = Vec::with_capacity(4 * turns);
    for i in 0..turns {
        let file = format!("src/module_{}.rs", i % 37);
        let lines: String = (1..=300)
            .map(|n| format!("{n:>6}\tfn handle_{i}_{n}(x: u32) -> u32 {{ x + {n} }}\n"))
            .collect();
        let question = json!({"role": "user", "content": [{"type": "text",
            "text": format!("Turn {i}: look at {file} and tell me what `handle_{i}_*` do.")}]});
        let mut asked: Message = or_exit(serde_json::from_value(question), "a question");
        if images
            && i % every == 0
            && let Message::User { content } = &mut asked
        {
            content.push(rig_core::message::UserContent::Image(Image {
                data: DocumentSourceKind::base64(image.clone()),
                media_type: Some(ImageMediaType::PNG),
                ..Image::default()
            }));
        }
        let blocks: Vec<AssistantContent> = or_exit(
            serde_json::from_value(json!([
                {"type": "reasoning", "text": format!("I should read {file} to see the handle_{i} functions.")},
                {"type": "toolcall", "id": native.call_id(i),
                 "function": {"name": "tool_0", "arguments": {"path": file, "offset": 1, "limit": 300}}}
            ])),
            "a call",
        );
        let answer: Vec<AssistantContent> = or_exit(
            serde_json::from_value(json!([{"type": "text",
                "text": format!("The functions `handle_{i}_1` to `handle_{i}_300` in {file} each add \
                    their own number to the argument.")}])),
            "an answer",
        );
        let turn = |blocks: Vec<AssistantContent>, stop| {
            Message::Assistant(
                AssistantMessage::new(blocks.into_iter().map(|b| native.item(b, i)).collect())
                    .with_origin(native.origin.clone())
                    .with_stop(stop),
            )
        };
        let result: Message = or_exit(
            serde_json::from_value(json!({"role": "user", "content": [{"type": "toolresult",
                "call": native.call_id(i), "name": "tool_0",
                "content": [{"type": "text", "text": lines}]}]})),
            "a tool result",
        );
        messages.extend([
            asked,
            turn(blocks, StopReason::ToolUse),
            result,
            turn(answer, StopReason::Stop),
        ]);
    }
    messages
}
