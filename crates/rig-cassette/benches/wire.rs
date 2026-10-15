//! A provider exchange's hot paths, per wire family (OpenAI Responses,
//! Anthropic Messages, OpenAI Chat, Gemini), with no network: a transport
//! hands the driver recorded replies from `fixtures/cassettes` and synthetic
//! ones, and takes request bodies without sending them.
//!
//! - `stream_decode`: a recorded text reply with its deltas repeated to
//!   2,000, one SSE event per transport chunk, in 16 KB chunks and in 1 MB
//!   chunks (the whole reply at once);
//! - `stream_serve`: the same reply in 16 KB chunks served by a model
//!   handler to a recorded dispatch, as an agent's model call is;
//! - `stream_decode_tool_args_30k`: a Responses function call and a custom
//!   call whose 30 KB of arguments stream in 3,000 deltas;
//! - `fold_fragmented_call`: a Chat tool call in 3,000 fragments, with an
//!   index, with one and a fresh id in every fragment, without one, and
//!   without one but restating the call's id;
//! - `prepare_encode`: a request of 200 and 3,000 messages (reasoning, calls
//!   and text carrying their provider items, 20 KB tool results, five
//!   500 KB images) and 20 tools, prepared and encoded to its body;
//! - `check`: `check` of that request with 3,000 messages;
//! - `check_next_call`: a model handler's `check` of that request with one
//!   more message, after it checked the request, as on every model call;
//! - `fingerprint`: a 4 KB text, a call with 20 KB of arguments and a 1 MB
//!   image;
//! - `effect_log_append`: appending the completion record of a request of
//!   100, 1,000 and 3,000 messages to an effect log whose last record is the
//!   same request four messages earlier;
//! - `gemini_caching`: a Gemini request of 200 messages through the
//!   `Caching` transport, which reads, digests and plans its body.
//!
//! `cargo bench -p rig-cassette --bench wire --features http,jsonl`.

use std::fmt::Display;
use std::future::Future;
use std::path::Path;
use std::sync::{Arc, Mutex};

use futures::StreamExt as _;
use rig_cassette::effect_log::{EffectLog, jsonl};
use rig_cassette::http::recorded_statuses_and_bodies;
use rig_core::DynModel;
use rig_core::completion::{
    AssistantContent, AssistantMessage, CompletionRequest, CompletionResponse, Message,
    ToolDefinition, Usage,
};
use rig_core::effect::{Answered, EffectId, EffectKind, EffectRecord, HandlerKey};
use rig_core::error::RigError;
use rig_core::http_client::{
    self, Bytes, HttpClientExt, LazyBody, MultipartForm, Request, Response, StreamingResponse,
};
use rig_core::message::{DocumentSourceKind, Image, ImageMediaType, Origin, StopReason};
use rig_core::operation::Completion;
use rig_core::providers::anthropic::AnthropicConfig;
use rig_core::providers::gemini::GeminiConfig;
use rig_core::providers::openai::OpenAIConfig;
use rig_core::serve::{
    Dispatch, DispatchContext, ErasedHandler, RecordBuffer, Recorder, Serve, adapters::ModelAdapter,
};
use rig_core::wasm_compat::WasmCompatSend;
use serde_json::{Value, json};

#[global_allocator]
static ALLOC: divan::AllocProfiler = divan::AllocProfiler::system();

fn main() {
    divan::main();
}

/// `result`'s value, or the process ends saying what failed: a bench whose
/// input cannot be made measures nothing.
fn or_exit<T, E: Display>(result: Result<T, E>, what: &str) -> T {
    result.unwrap_or_else(|error| {
        eprintln!("{what}: {error}");
        std::process::exit(1)
    })
}

/// Ends the process saying `why` unless `holds`.
fn ensure(holds: bool, why: impl FnOnce() -> String) {
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
struct Replay {
    reply: Option<Arc<Vec<Bytes>>>,
    sent: Arc<Mutex<Bytes>>,
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
const FAMILIES: [&str; 4] = ["responses", "anthropic", "chat", "gemini"];

/// `family`'s completion model on `transport`, as the cassettes recorded it.
fn model(family: &str, transport: Replay) -> DynModel<Completion> {
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
        _ => ("gemini", "corpus_breadth/text_delta_stop", |event| {
            event.contains("\"parts\":[{\"text\"") && !event.contains("finishReason")
        }),
    }
}

/// `body`'s SSE events, each with the blank line that ends it.
fn events(body: &str) -> Vec<&str> {
    let end = match body.contains("\r\n\r\n") {
        true => "\r\n\r\n",
        false => "\n\n",
    };
    body.split_inclusive(end).collect()
}

/// `family`'s recorded text reply with its deltas repeated to `deltas`: what
/// comes before the first delta, the deltas, and what follows the last.
fn long_reply(family: &str, deltas: usize) -> String {
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
fn chunked(reply: &str, size: usize) -> Arc<Vec<Bytes>> {
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
fn replaying(family: &str, reply: Arc<Vec<Bytes>>) -> DynModel<Completion> {
    let transport = Replay {
        reply: Some(reply),
        ..Replay::default()
    };
    model(family, transport)
}

/// `request` streamed from `model` and drained: the items and the response.
fn drain(
    model: &DynModel<Completion>,
    request: CompletionRequest,
) -> Result<(usize, CompletionResponse), RigError> {
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
fn tools() -> Vec<ToolDefinition> {
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
fn request(history: &[Message]) -> CompletionRequest {
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
fn base64(bytes: usize) -> String {
    use base64::Engine as _;
    let raw: Vec<u8> = (0..bytes)
        .map(|i| (i.wrapping_mul(31) % 251) as u8)
        .collect();
    base64::engine::general_purpose::STANDARD.encode(raw)
}

/// The opaque part of the history's reasoning items, repeated.
const SIGNATURE: &str = "c2lnbmF0dXJl";

/// What the history's assistant turns carry from `family`'s replies: who
/// made them, and the provider item of each block.
struct Native {
    family: &'static str,
    origin: Origin,
}

impl Native {
    /// `family`'s origin, as decoding its recorded reply to [`request`]
    /// stamps it, so the history's turns are this model's own.
    fn of(family: &'static str) -> Self {
        let model = replaying(family, chunked(&long_reply(family, 4), 0));
        let (_, response) = or_exit(drain(&model, request(&[])), "decoding the recorded reply");
        Self {
            family,
            origin: response.origin,
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
fn history(turns: usize, native: &Native, images: bool) -> Vec<Message> {
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
                {"type": "toolcall", "id": {"provider": format!("call-{i}")},
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
                "call": {"provider": format!("call-{i}")}, "name": "tool_0",
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

// Decoding.

/// A recorded text reply with 2,000 deltas, streamed and folded, per family,
/// one event per chunk (`CHUNK` 0), in 16 KB chunks or in 1 MB chunks.
#[divan::bench(args = FAMILIES, consts = [0, 16_384, 1_048_576], sample_count = 30)]
fn stream_decode<const CHUNK: usize>(bencher: divan::Bencher, family: &str) {
    let model = replaying(family, chunked(&long_reply(family, 2_000), CHUNK));
    let request = CompletionRequest::new("Tell me about Rust.");
    let (items, response) = or_exit(drain(&model, request.clone()), "the long reply");
    ensure(items > 1_900 && !response.choice.is_empty(), || {
        format!("{family}: {items} items")
    });
    bencher
        .with_inputs(|| request.clone())
        .bench_local_values(|request| drain(&model, request));
}

/// The recorded text reply in 16 KB chunks, served by a model handler to a
/// dispatch a record buffer records, drained, and its record taken: the
/// decode plus what serving and recording a model call add.
#[divan::bench(args = FAMILIES, sample_count = 30)]
fn stream_serve(bencher: divan::Bencher, family: &str) {
    let model = replaying(family, chunked(&long_reply(family, 2_000), 16_384));
    let handler = ErasedHandler::new(ModelAdapter::new("bench/model", model));
    let request = Arc::new(CompletionRequest::new("Tell me about Rust."));
    let serve = || {
        futures::executor::block_on(async {
            let buffer = RecordBuffer::new();
            let id = EffectId::from_raw(1);
            let kind = EffectKind::Completion {
                request: request.clone(),
                stream: true,
            };
            let context = DispatchContext::default();
            buffer.begin(id, handler.key().clone(), kind.clone(), context);
            let dispatch = Dispatch::new(id, true).recorded_by(Arc::new(buffer.clone()));
            let mut stream = handler.handle(kind, dispatch).await.into_stream();
            let mut items = 0;
            while let Some(item) = stream.next().await {
                items += usize::from(item.is_ok());
            }
            (items, buffer.take(id).map(|record| record.outcome.is_ok()))
        })
    };
    let (items, recorded) = serve();
    ensure(items > 1_900 && recorded == Some(true), || {
        format!("{family}: {items} items, recorded {recorded:?}")
    });
    bencher.bench_local(serve);
}

/// A Responses reply whose one call's 30 KB of arguments stream in 3,000
/// deltas: a function call, or a custom call whose input is raw text.
fn tool_args_reply(custom: bool) -> String {
    let code: String = (0..600)
        .map(|n| format!("    let line_{n} = \"value {n}\";\n"))
        .collect();
    let (kind, delta_type, done_type, field, full) = match custom {
        true => (
            "custom_tool_call",
            "response.custom_tool_call_input.delta",
            "response.custom_tool_call_input.done",
            "input",
            format!("*** Begin Patch\n*** Add File: src/big.rs\n{code}*** End Patch\n"),
        ),
        false => (
            "function_call",
            "response.function_call_arguments.delta",
            "response.function_call_arguments.done",
            "arguments",
            json!({"path": "src/big.rs", "content": code}).to_string(),
        ),
    };
    let chars: Vec<char> = full.chars().collect();
    let pieces: Vec<String> = chars
        .chunks(chars.len().div_ceil(3_000))
        .map(|piece| piece.iter().collect())
        .collect();
    let item = |status: &str, value: &str| {
        json!({"type": kind, "id": "fc_1", "call_id": "call_1", "name": "write",
            "status": status, field: value})
    };
    let response = |status: &str, output: Vec<Value>| {
        json!({"id": "resp_1", "object": "response", "created_at": 0, "status": status,
            "model": "gpt-5.4", "output": output,
            "usage": {"input_tokens": 10, "output_tokens": 3_000, "total_tokens": 3_010,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens_details": {"reasoning_tokens": 0}}})
    };
    let mut events = vec![
        json!({"type": "response.created", "sequence_number": 0,
            "response": response("in_progress", Vec::new())}),
        json!({"type": "response.output_item.added", "sequence_number": 1, "output_index": 0,
            "item": item("in_progress", "")}),
    ];
    events.extend(pieces.iter().enumerate().map(|(n, piece)| {
        json!({"type": delta_type, "sequence_number": n + 2, "item_id": "fc_1",
            "output_index": 0, "delta": piece})
    }));
    events.extend([
        json!({"type": done_type, "item_id": "fc_1", "output_index": 0, field: full}),
        json!({"type": "response.output_item.done", "output_index": 0,
            "item": item("completed", &full)}),
        json!({"type": "response.completed",
            "response": response("completed", vec![item("completed", &full)])}),
    ]);
    events
        .iter()
        .map(|event| {
            let name = event
                .get("type")
                .and_then(Value::as_str)
                .unwrap_or_default();
            format!("event: {name}\ndata: {event}\n\n")
        })
        .collect()
}

/// A Responses call whose 30 KB of arguments stream in 3,000 deltas.
#[divan::bench(args = ["function", "custom"], sample_count = 20)]
fn stream_decode_tool_args_30k(bencher: divan::Bencher, call: &str) {
    let reply = chunked(&tool_args_reply(call == "custom"), 0);
    let model = replaying("responses", reply);
    let request = CompletionRequest::new("Write src/big.rs.");
    let (_, response) = or_exit(drain(&model, request.clone()), "the tool-argument reply");
    ensure(
        response
            .choice
            .iter()
            .any(|block| matches!(block, AssistantContent::ToolCall(_))),
        || format!("{call}: no call in {:?}", response.choice),
    );
    bencher
        .with_inputs(|| request.clone())
        .bench_local_values(|request| drain(&model, request));
}

/// A Chat reply of one call whose arguments come in 3,000 fragments: each
/// with the call's index (`index`), with neither index nor id (`noindex`),
/// or without an index but restating the call's id (`noindex_id`).
fn fragmented_reply(how: &str) -> String {
    let code: String = (0..600)
        .map(|n| format!("    let line_{n} = \"value {n}\";\n"))
        .collect();
    let arguments = json!({"path": "src/big.rs", "content": code}).to_string();
    let chars: Vec<char> = arguments.chars().collect();
    let chunk = |delta: Value, finish: Value| {
        json!({"id": "chatcmpl-1", "object": "chat.completion.chunk", "created": 0,
            "model": "gpt-4o", "usage": null,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish, "logprobs": null}]})
    };
    let call = |mut call: Value, n: usize| {
        if let Some(fields) = call.as_object_mut() {
            match how {
                "index" => {
                    fields.insert("index".into(), json!(0));
                }
                "index_churn" => {
                    fields.insert("index".into(), json!(0));
                    fields.insert("id".into(), json!(format!("call_{n}")));
                }
                "noindex_id" => {
                    fields.insert("id".into(), json!("call_1"));
                }
                _ => {}
            }
        }
        chunk(json!({"tool_calls": [call]}), Value::Null)
    };
    let mut chunks = vec![
        chunk(json!({"role": "assistant", "content": null}), Value::Null),
        call(
            json!({"id": "call_1", "type": "function",
            "function": {"name": "write", "arguments": ""}}),
            0,
        ),
    ];
    chunks.extend(
        chars
            .chunks(chars.len().div_ceil(3_000))
            .enumerate()
            .map(|(n, piece)| {
                call(
                    json!({"function": {"arguments": piece.iter().collect::<String>()}}),
                    n + 1,
                )
            }),
    );
    chunks.push(chunk(json!({}), json!("tool_calls")));
    chunks.push(
        json!({"id": "chatcmpl-1", "object": "chat.completion.chunk", "created": 0,
        "model": "gpt-4o", "choices": [],
        "usage": {"prompt_tokens": 10, "completion_tokens": 3_000, "total_tokens": 3_010}}),
    );
    let mut reply: String = chunks
        .iter()
        .map(|chunk| format!("data: {chunk}\n\n"))
        .collect();
    reply.push_str("data: [DONE]\n\n");
    reply
}

/// A Chat call in 3,000 fragments, folded into one call.
#[divan::bench(
    args = ["index", "index_churn", "noindex", "noindex_id"],
    sample_count = 20
)]
fn fold_fragmented_call(bencher: divan::Bencher, how: &str) {
    let model = replaying("chat", chunked(&fragmented_reply(how), 0));
    let request = CompletionRequest::new("Write src/big.rs.");
    let (_, response) = or_exit(drain(&model, request.clone()), "the fragmented call");
    let calls = response
        .choice
        .iter()
        .filter(|block| matches!(block, AssistantContent::ToolCall(_)))
        .count();
    ensure(calls == 1, || format!("{how}: {calls} calls"));
    bencher
        .with_inputs(|| request.clone())
        .bench_local_values(|request| drain(&model, request));
}

// Encoding.

/// A request of `MESSAGES` messages of the family's own turns, prepared and
/// encoded to its body (the transport refuses to send it).
#[divan::bench(args = FAMILIES, consts = [200, 3_000], sample_count = 20)]
fn prepare_encode<const MESSAGES: usize>(bencher: divan::Bencher, family: &str) {
    let family = FAMILIES
        .into_iter()
        .find(|known| *known == family)
        .unwrap_or("responses");
    let native = Native::of(family);
    let request = request(&history(MESSAGES / 4, &native, true));
    let transport = Replay::default();
    let sent = transport.sent.clone();
    let model = model(family, transport);
    let refused = drain(&model, request.clone());
    let body = sent.lock().map(|body| body.clone()).unwrap_or_default();
    // The turns replay their provider items (Chat keeps none).
    let replayed = String::from_utf8_lossy(&body).contains(SIGNATURE) || family == "chat";
    ensure(
        refused.is_err() && body.len() > 2_500_000 && replayed,
        || {
            format!(
                "{family}: a {}-byte body, items replayed: {replayed}, {:?}",
                body.len(),
                refused.map(|(items, _)| items)
            )
        },
    );
    bencher
        .with_inputs(|| request.clone())
        .bench_local_values(|request| drain(&model, request).is_err());
}

/// `check` of the 3,000-message request with 20 tools.
#[divan::bench(args = FAMILIES, sample_count = 30)]
fn check(bencher: divan::Bencher, family: &str) {
    let request = request(&history(750, &Native::of_none(), true));
    let model = model(family, Replay::default());
    or_exit(model.check(&request), "the check");
    bencher.bench_local(|| model.check(&request));
}

/// A model handler's `check` of the 3,000-message request with 20 tools
/// and one more message, after it checked the request: what an agent asks
/// on every model call.
#[divan::bench(args = FAMILIES, sample_count = 30)]
fn check_next_call(bencher: divan::Bencher, family: &str) {
    let request = Arc::new(request(&history(750, &Native::of_none(), true)));
    let handler = ModelAdapter::new("bench/model", model(family, Replay::default()));
    let kind = |request: Arc<CompletionRequest>| EffectKind::Completion {
        request,
        stream: true,
    };
    let refused = handler.check(&kind(request.clone()));
    ensure(refused.is_empty(), || format!("{family}: {refused:?}"));
    bencher
        .with_inputs(|| {
            let mut next = CompletionRequest::clone(&request);
            next.chat_history.push(Arc::new(Message::user("And then?")));
            kind(Arc::new(next))
        })
        .bench_local_refs(|kind| handler.check(kind));
}

impl Native {
    /// No provider items: hand-built turns.
    fn of_none() -> Self {
        Self {
            family: "none",
            origin: Origin::new("bench.api", "bench", "bench-model"),
        }
    }
}

/// The fingerprint of a block: 4 KB of text, a call with 20 KB of nested
/// arguments, a 1 MB image.
#[divan::bench(args = ["text4k", "args20k", "image1m"], sample_count = 50)]
fn fingerprint(bencher: divan::Bencher, block: &str) {
    let block: AssistantContent = match block {
        "text4k" => AssistantContent::text("word ".repeat(4 * 1024 / 5)),
        "args20k" => {
            let entries: Vec<Value> = (0..200)
                .map(|n| {
                    json!({"zeta": n, "alpha": format!("value {n} {}", "x".repeat(60)),
                    "mid": {"b": 20.0, "a": [1, 2.5, "three"]}})
                })
                .collect();
            or_exit(
                serde_json::from_value(json!({"type": "toolcall", "id": {"provider": "call_1"},
                    "function": {"name": "write", "arguments": {"path": "a.rs", "entries": entries}}})),
                "a call",
            )
        }
        _ => AssistantContent::Image(Image {
            data: DocumentSourceKind::base64(base64(768 * 1024)),
            media_type: Some(ImageMediaType::PNG),
            ..Image::default()
        }),
    };
    bencher.bench_local(|| block.fingerprint());
}

// The effect log.

/// The completion record of `id` with `request`, answered by a short reply.
fn completion_record(id: u64, request: CompletionRequest) -> EffectRecord {
    EffectRecord {
        purpose: None,
        turn: None,
        started_ms: None,
        finished_ms: None,
        reviewed: None,
        disposition: Default::default(),
        decided_by: None,
        stream_origin: None,
        published: None,
        parent: None,
        scope: Some(Arc::from("agent")),
        id: EffectId::from_raw(id),
        key: HandlerKey::from("model"),
        kind: EffectKind::Completion {
            request: request.into(),
            stream: true,
        },
        outcome: Ok(Answered::Completion(CompletionResponse::new(
            vec![AssistantContent::text("ok")],
            Usage::default(),
            Origin::new("openai.responses", "openai", "gpt-5.4"),
            json!({"id": "resp_1", "usage": {"input_tokens": 3}}),
        ))),
        events: None,
    }
}

/// Appending the completion record of a request of `MESSAGES` messages to
/// an effect log whose last record is the same request four messages
/// earlier: the continuation every model call writes. The two requests
/// share their messages, as a conversation's requests do. Each sample
/// appends to a fresh log holding that earlier record.
#[divan::bench(consts = [100, 1_000, 3_000], sample_count = 20, sample_size = 1)]
fn effect_log_append<const MESSAGES: usize>(bencher: divan::Bencher) {
    let mut sent = request(&history(MESSAGES / 4, &Native::of("responses"), false));
    sent.chat_history.pop();
    let mut before = sent.clone();
    before
        .chat_history
        .truncate(sent.chat_history.len().saturating_sub(4));
    let earlier = EffectLog::from_records(vec![completion_record(1, before)]);
    let next = EffectLog::from_records(vec![completion_record(2, sent)]);
    let primed = || {
        let store = rig_memory::MemoryStore::default();
        let mut writer = jsonl::Writer::on(Arc::new(store.clone()), jsonl::SESSION_EFFECT_LOG);
        or_exit(
            writer.append(&earlier.header, &earlier.records),
            "the earlier record",
        );
        (store, writer)
    };
    let (store, mut writer) = primed();
    or_exit(writer.append(&next.header, &next.records), "the record");
    let read = or_exit(jsonl::read_session(&store), "reading the log back");
    ensure(read.records.len() == 2, || {
        format!("{} records read back", read.records.len())
    });
    bencher
        .with_inputs(primed)
        .bench_local_values(|(store, mut writer)| {
            writer
                .append(&next.header, &next.records)
                .map(|()| (store, writer))
        });
}

/// A Gemini request of `MESSAGES` messages through the `Caching` transport:
/// its body parsed, every content digested and estimated, and the book's
/// plan made, before it goes out inline (the transport refuses it).
#[divan::bench(consts = [200], sample_count = 20)]
fn gemini_caching<const MESSAGES: usize>(bencher: divan::Bencher) {
    use rig_core::providers::gemini::{AutoCache, CacheBook};
    let request = request(&history(MESSAGES / 4, &Native::of("gemini"), false));
    let book = CacheBook::new(AutoCache::default());
    let model = GeminiConfig::new("bench-key")
        .connect(Replay::default())
        .completion("gemini-2.5-flash")
        .caching(&book)
        .erase();
    ensure(drain(&model, request.clone()).is_err(), || {
        "the caching transport sent the request".to_owned()
    });
    bencher
        .with_inputs(|| request.clone())
        .bench_local_values(|request| drain(&model, request).is_err());
}
