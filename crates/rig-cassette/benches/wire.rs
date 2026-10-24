//! A provider exchange's hot paths, per wire family (OpenAI Responses,
//! Anthropic Messages, OpenAI Chat, Gemini, Cohere, Ollama's native chat),
//! with no network: a transport hands the driver recorded replies from
//! `fixtures/cassettes` and synthetic ones, and takes request bodies
//! without sending them.
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
//! - `prepare_encode_anthropic`: the 200-message Anthropic request for a
//!   model whose provider items bind to the request's tools and system
//!   prompt, and with four message cache points;
//! - `check`: `check` of that request with 3,000 messages;
//! - `check_next_call`: a model handler's `check` of that request with one
//!   more message, after it checked the request, as on every model call;
//! - `fingerprint`: a 4 KB text, a call with 20 KB of arguments and a 1 MB
//!   image;
//! - `effect_log_append`: appending the completion record of a request of
//!   100, 1,000 and 3,000 messages to an effect log whose last record is the
//!   same request four messages earlier;
//! - `effect_log_append_tools`: the same at 100 messages with ten tools of
//!   about 3 KB of schema each;
//! - `effect_log_replay`: a replay answering the completion of a request of
//!   100 and 1,000 messages, whose calls carry ids rig issued, from its
//!   recorded log;
//! - `gemini_caching`: a Gemini request of 200 messages through the
//!   `Caching` transport, which reads, digests and plans its body.
//!
//! Beside them, three of rig-core's own paths: `catalog_builtin`, parsing
//! the built-in model catalog as a process's first use of it does,
//! `tool_call`, a typed tool's call with its handler built for it as the
//! harness builds one per call, and `vector_search`, the in-memory vector
//! store's scan, unfiltered and filtered.
//!
//! `cargo bench -p rig-cassette --bench wire --features http,jsonl`.

use std::sync::Arc;

use futures::StreamExt as _;
use rig_cassette::effect_log::{EffectLog, RecordedAnswers, jsonl};
use rig_core::completion::{
    AssistantContent, CachePoint, CompletionRequest, CompletionResponse, Message, ToolDefinition,
    Usage,
};
use rig_core::driver::{Exchange, Local, Opened, Opening, Step, Transport};
use rig_core::effect::{Answered, EffectId, EffectKind, EffectRecord, HandlerKey};
use rig_core::embeddings::{Embedding, EmbeddingResponse};
use rig_core::message::{DocumentSourceKind, Image, ImageMediaType, Origin};
use rig_core::providers::anthropic::AnthropicConfig;
use rig_core::providers::gemini::GeminiConfig;
use rig_core::serve::adapters::{ModelAdapter, ToolAdapter};
use rig_core::serve::{
    Dispatch, DispatchContext, ErasedHandler, RecordBuffer, Recorder, Serve, serve_inline,
};
use rig_core::tool::{Tool, ToolContext, ToolExecutionError, args_schema};
use rig_core::wire::Capabilities;
use serde_json::{Value, json};

mod common;
use common::{
    FAMILIES, Native, Replay, SIGNATURE, base64, chunked, drain, ensure, history, long_reply,
    model, or_exit, replaying, request,
};

#[global_allocator]
static ALLOC: divan::AllocProfiler = divan::AllocProfiler::system();

fn main() {
    divan::main();
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
    // The turns replay their provider items (Chat, Cohere and Ollama keep none).
    let replayed = String::from_utf8_lossy(&body).contains(SIGNATURE)
        || matches!(family, "chat" | "cohere" | "ollama");
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

/// The 200-message Anthropic request on a model whose provider items bind
/// to the request's tools and system prompt (`bound`), or with four message
/// cache points (`cached`).
#[divan::bench(args = ["bound", "cached"], sample_count = 20)]
fn prepare_encode_anthropic(bencher: divan::Bencher, how: &str) {
    let mut request = request(&history(50, &Native::of("anthropic"), true));
    if how == "cached" {
        request.cache_points = (1..=4).map(|n| CachePoint::Messages(n * 40)).collect();
    }
    let transport = Replay::default();
    let sent = transport.sent.clone();
    let model = AnthropicConfig::new("bench-key")
        .connect(transport)
        .completion(match how {
            "bound" => "claude-sonnet-5-5",
            _ => "claude-sonnet-4-6",
        })
        .erase();
    let refused = drain(&model, request.clone());
    let body = sent.lock().map(|body| body.len()).unwrap_or_default();
    ensure(refused.is_err() && body > 2_500_000, || {
        format!(
            "{how}: a {body}-byte body, {:?}",
            refused.map(|(items, _)| items)
        )
    });
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
        scope: Some(Arc::from("agent")),
        ..EffectRecord::new(
            EffectId::from_raw(id),
            HandlerKey::from("model"),
            EffectKind::Completion {
                request: request.into(),
                stream: true,
            },
            Ok(Answered::Completion(
                CompletionResponse::new(
                    vec![AssistantContent::text("ok")],
                    Usage::default(),
                    Origin::new("openai.responses", "openai", "gpt-5.4"),
                    json!({"id": "resp_1", "usage": {"input_tokens": 3}}),
                )
                .into(),
            )),
        )
    }
}

/// Appending the completion record of a request of `MESSAGES` messages to
/// an effect log whose last record is the same request four messages
/// earlier: the continuation every model call writes. The two requests
/// share their messages, as a conversation's requests do. Each sample
/// appends to a fresh log holding that earlier record.
#[divan::bench(consts = [100, 1_000, 3_000], sample_count = 20, sample_size = 1)]
fn effect_log_append<const MESSAGES: usize>(bencher: divan::Bencher) {
    appending(
        bencher,
        request(&history(MESSAGES / 4, &Native::of("responses"), false)),
    );
}

/// [`effect_log_append`] of a request of `MESSAGES` messages with ten
/// tools of about 3 KB of schema each, as a coding agent sends them, the
/// same in both records.
#[divan::bench(consts = [100], sample_count = 20, sample_size = 1)]
fn effect_log_append_tools<const MESSAGES: usize>(bencher: divan::Bencher) {
    let mut sent = request(&history(MESSAGES / 4, &Native::of("responses"), false));
    sent.tools = (0..10).map(heavy_tool).collect();
    appending(bencher, sent);
}

/// Tool `i` with about 3 KB of schema: 24 described parameters.
fn heavy_tool(i: usize) -> Arc<ToolDefinition> {
    let properties: serde_json::Map<String, Value> = (0..24)
        .map(|p| {
            let about = format!(
                "Parameter {p} of tool {i}: what it selects, which values it \
                takes and what the tool does when it is left out."
            );
            (
                format!("option_{p}"),
                json!({"type": "string", "description": about}),
            )
        })
        .collect();
    let tool = json!({
        "name": format!("heavy_{i}"),
        "description": format!("Tool number {i}: reads, writes or searches files in the workspace."),
        "parameters": {"type": "object", "properties": properties, "required": ["option_0"]}
    });
    or_exit(serde_json::from_value(tool), "a tool definition")
}

/// Appends the completion record of `sent`, less its question, to a fresh
/// log holding the record of the same request four messages earlier.
fn appending(bencher: divan::Bencher, mut sent: CompletionRequest) {
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

/// A replay answering a model call from its recorded log: the completion
/// of a request of `MESSAGES` messages whose calls carry ids rig issued
/// (as for a provider that sends none, which a replay renames), matched
/// against the log's record of the same request.
#[divan::bench(consts = [100, 1_000], sample_count = 20)]
fn effect_log_replay<const MESSAGES: usize>(bencher: divan::Bencher) {
    let sent = request(&history(MESSAGES / 4, &Native::issuing_ids(), false));
    let record = completion_record(1, sent);
    let kind = record.kind.clone();
    let answers = RecordedAnswers::new(&EffectLog::from_records(vec![record]));
    let answer = |mut answers: RecordedAnswers| {
        let answered = answers.answer(Some("agent"), &kind);
        answered
            .map(|record| record.id)
            .map_err(|divergence| divergence.report)
    };
    or_exit(answer(answers.clone()), "the recorded answer");
    bencher
        .with_inputs(|| answers.clone())
        .bench_local_values(answer);
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

// rig-core beside the wires.

/// The catalog this build ships, parsed from its models.dev data, as a
/// process's first `Catalog::builtin` does.
#[divan::bench(sample_count = 20)]
fn catalog_builtin(bencher: divan::Bencher) {
    use rig_core::catalog::Catalog;
    const MODELS: &str = include_str!("../../rig-core/src/catalog/models.json");
    let parse =
        || Catalog::from_models_dev(MODELS).map(|(catalog, skipped)| (catalog, skipped.len()));
    let (catalog, skipped) = or_exit(parse(), "the built-in catalog");
    ensure(skipped == 0 && catalog.iter().count() > 1_000, || {
        format!("{} models, {skipped} skipped", catalog.iter().count())
    });
    bencher.bench_local(parse);
}

#[derive(serde::Deserialize, schemars::JsonSchema)]
struct EditArgs {
    /// The file to edit, relative to the workspace root.
    path: String,
    /// The exact text to replace.
    old: String,
    /// The text that replaces it.
    new: String,
    /// Whether every occurrence is replaced, not only the first.
    all: Option<bool>,
}

/// A typed tool with an edit tool's arguments.
struct Edit;

impl Tool for Edit {
    const NAME: &'static str = "edit";
    type Args = EditArgs;
    type Output = String;
    type Error = ToolExecutionError;

    fn description(&self) -> String {
        "Replace text in a file.".to_owned()
    }

    fn parameters(&self) -> Value {
        args_schema::<EditArgs>()
    }

    async fn call(&self, _: &mut ToolContext, args: EditArgs) -> Result<String, Self::Error> {
        let all = args.all.unwrap_or(false);
        Ok(format!(
            "{} -> {} in {} (all: {all})",
            args.old, args.new, args.path
        ))
    }
}

/// One tool call as the harness runs it: the call's handler built for it
/// (`ErasedHandler::new(ToolAdapter::new(tool))`), then the call served
/// inline.
#[divan::bench(sample_count = 50)]
fn tool_call(bencher: divan::Bencher) {
    let call = || {
        let handler = ErasedHandler::new(ToolAdapter::new(Edit));
        let kind = EffectKind::ToolCall {
            name: Edit::NAME.to_owned(),
            args: r#"{"path":"src/lib.rs","old":"a","new":"b"}"#.to_owned(),
        };
        futures::executor::block_on(serve_inline(&handler, kind)).is_ok()
    };
    ensure(call(), || "the edit call failed".to_owned());
    bencher.bench_local(call);
}

/// Embeds every text as one fixed vector.
#[derive(Clone)]
struct FixedEmbedding(Arc<Vec<f64>>);

impl Transport<Local<rig_core::operation::Embedding>> for FixedEmbedding {
    fn send(
        &self,
        texts: Vec<String>,
        _exchange: Exchange,
    ) -> Opening<Step<rig_core::operation::Embedding>> {
        let vec = self.0.to_vec();
        let embeddings = texts.into_iter().map(|document| Embedding {
            document,
            vec: vec.clone(),
        });
        let response = EmbeddingResponse::new(embeddings.collect());
        Opening::ready(Opened::new(futures::stream::iter([Ok(Step::End(
            response,
        ))])))
    }
}

/// The in-memory vector store's ten nearest of 10,000 documents of one
/// 1,536-dimension embedding each: unfiltered, and filtered on a metadata
/// field a quarter of the documents match.
#[divan::bench(args = ["unfiltered", "filtered"], sample_count = 20)]
fn vector_search(bencher: divan::Bencher, how: &str) {
    use rig_core::vector_store::request::{Filter, SearchFilter as _, VectorSearchRequest};
    use rig_core::vector_store::{VectorStoreIndex as _, in_memory_store::InMemoryVectorStore};
    const DIMENSIONS: usize = 1_536;
    let vector = |seed: usize| -> Vec<f64> {
        (0..DIMENSIONS)
            .map(|d| ((seed * 31 + d * 17) % 101) as f64 / 101.0 - 0.5)
            .collect()
    };
    let documents = (0..10_000).map(|i| {
        let text = format!("document {i}");
        let embedding = Embedding {
            document: text.clone(),
            vec: vector(i),
        };
        (json!({"text": text, "kind": i % 4}), vec![embedding])
    });
    let model = rig_core::Model::new(
        Local::new("bench").with_capabilities(Capabilities::embedding(16, DIMENSIONS)),
        FixedEmbedding(Arc::new(vector(7_777))),
    );
    let index = InMemoryVectorStore::from_documents(documents).index(model);
    let request = VectorSearchRequest::builder()
        .query("the query")
        .samples(10);
    let request = match how {
        "filtered" => request.filter(Filter::eq("kind", json!(1))).build(),
        _ => request.build(),
    };
    let search = || futures::executor::block_on(index.top_n_ids(request.clone()));
    let found = or_exit(search(), "the search");
    ensure(found.len() == 10, || {
        format!("{how}: {} found", found.len())
    });
    bencher.bench_local(search);
}
