//! Helpers for the typed provider-options tests: the body a request encodes
//! to, a recorded reply read out of a cassette and decoded as a live call
//! decodes it, and the check that an `Options` type writes no leaf the
//! mapped options or the request own.

use std::collections::BTreeSet;

use serde_json::{Value, json};

use crate::completion::{CacheRetention, Effort, Reasoning, ServiceTier, Verbosity};
use crate::completion::{
    CompletionRequest, CompletionResponse, GenerationOptions, OnUnsupported, ProviderExtension,
    ProviderOptions, ToolDefinition,
};
use crate::error::ProviderError;
use crate::message::{ToolChoice, ToolName};
use crate::operation::Completion;
use crate::test_utils::{RecordingHttpClient, json_body};
use crate::wire::{Encoded, Mode, Operation, Wire};

/// `request` prepared as the driver prepares it, then encoded by `wire` in
/// `mode`, as the JSON body it sends.
///
/// # Errors
///
/// When `prepare` or the encoder refuses the request.
pub(crate) fn encoded_body<W>(
    wire: &W,
    request: CompletionRequest,
    mode: Mode,
) -> Result<Value, ProviderError>
where
    W: Wire<Op = Completion, Payload = Encoded>,
{
    let request = Completion::prepare(request, &wire.describe())?;
    let encoded = wire.encode(request, mode)?;
    Ok(json_body(&encoded.request))
}

/// A one-message request carrying `options` as `P`'s entry.
pub(crate) fn request_with<P: ProviderExtension>(options: &P::Options) -> CompletionRequest {
    let options = ProviderOptions::new()
        .with::<P>(options)
        .unwrap_or_else(|error| panic!("the options are sections: {error}"));
    CompletionRequest::new("hi").provider_options(options)
}

/// The unary body a one-message request with `options` as `P`'s entry
/// sends on `wire`.
pub(crate) fn body_with<P, W>(wire: &W, options: &P::Options) -> Value
where
    P: ProviderExtension,
    W: Wire<Op = Completion, Payload = Encoded>,
{
    encoded_body(wire, request_with::<P>(options), Mode::Unary)
        .unwrap_or_else(|error| panic!("the request encodes: {error}"))
}

/// The reply body of recorded interaction `index` of
/// `rig-cassette/fixtures/cassettes/<provider>/<scenario>.yaml`, a
/// single-quoted scalar. Read here rather than with a YAML dependency, and
/// never written.
pub(crate) fn recorded_reply(provider: &str, scenario: &str, index: usize) -> String {
    let file = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../rig-cassette/fixtures/cassettes")
        .join(provider)
        .join(format!("{scenario}.yaml"));
    let text = std::fs::read_to_string(&file)
        .unwrap_or_else(|error| panic!("{} should be readable: {error}", file.display()));
    let interaction = text
        .split("\n---\n")
        .nth(index)
        .unwrap_or_else(|| panic!("{} should record interaction {index}", file.display()));
    let reply = interaction
        .split_once("\nthen:")
        .unwrap_or_else(|| panic!("{} should record a reply", file.display()))
        .1;
    let body = reply
        .split_once("  body: '")
        .unwrap_or_else(|| panic!("{} should record a single-quoted body", file.display()))
        .1
        .lines()
        .next()
        .unwrap_or_default()
        .trim_end()
        .trim_end_matches('\'');
    body.replace("''", "'")
}

/// `body`, a unary reply, decoded by `wire` as a live call decodes it.
pub(crate) async fn reply_of<W>(wire: W, body: impl Into<String>) -> CompletionResponse
where
    W: Wire<Op = Completion>,
    RecordingHttpClient: crate::driver::Transport<W>,
{
    let body: String = body.into();
    crate::driver::Model::new(wire, RecordingHttpClient::new(bytes::Bytes::from(body)))
        .call(CompletionRequest::new("hi"))
        .await
        .unwrap_or_else(|error| panic!("the reply decodes: {error}"))
}

/// The event stream of recorded interaction `index` of
/// `rig-cassette/fixtures/cassettes/<provider>/<scenario>.yaml`, a `|+`
/// literal block. Read here rather than with a YAML dependency, and never
/// written.
pub(crate) fn recorded_stream(provider: &str, scenario: &str, index: usize) -> String {
    let file = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../rig-cassette/fixtures/cassettes")
        .join(provider)
        .join(format!("{scenario}.yaml"));
    let text = std::fs::read_to_string(&file)
        .unwrap_or_else(|error| panic!("{} should be readable: {error}", file.display()));
    let interaction = text
        .split("\n---\n")
        .nth(index)
        .unwrap_or_else(|| panic!("{} should record interaction {index}", file.display()));
    let reply = interaction
        .split_once("\nthen:")
        .unwrap_or_else(|| panic!("{} should record a reply", file.display()))
        .1;
    let block = reply
        .split_once("  body: |+\n")
        .unwrap_or_else(|| panic!("{} should record an event stream", file.display()))
        .1;
    let mut body = String::new();
    for line in block.lines() {
        if !line.is_empty() && !line.starts_with("    ") {
            break;
        }
        body.push_str(line.get(4..).unwrap_or_default());
        body.push('\n');
    }
    body
}

/// `body`, an event stream, folded by `wire` as a live stream folds it.
pub(crate) async fn streamed_reply_of<W>(wire: W, body: impl Into<String>) -> CompletionResponse
where
    W: Wire<Op = Completion>,
    crate::test_utils::MockStreamingClient: crate::driver::Transport<W>,
{
    use futures::StreamExt;

    let body: String = body.into();
    let model = crate::driver::Model::new(
        wire,
        crate::test_utils::MockStreamingClient {
            sse_bytes: bytes::Bytes::from(body),
        },
    );
    let mut stream = model
        .stream(CompletionRequest::new("hi"))
        .unwrap_or_else(|error| panic!("the stream opens: {error}"));
    while let Some(item) = stream.next().await {
        item.unwrap_or_else(|error| panic!("the stream yields no error: {error}"));
    }
    stream
        .finish()
        .await
        .unwrap_or_else(|error| panic!("the stream ends: {error}"))
}

/// The event stream of a Chat Completions reply: each of `chunks` as one
/// `data:` event, then `[DONE]`, for a provider no recording covers.
pub(crate) fn chat_stream(chunks: &[Value]) -> String {
    let mut body: String = chunks
        .iter()
        .map(|chunk| format!("data: {chunk}\n\n"))
        .collect();
    body.push_str("data: [DONE]\n\n");
    body
}

/// A Chat Completions reply holding one assistant text, with `extra` merged
/// over it, for a provider no recording covers.
pub(crate) fn chat_reply(extra: Value) -> String {
    let mut reply = json!({
        "id": "reply-1",
        "object": "chat.completion",
        "created": 0,
        "model": "model",
        "choices": [{
            "index": 0,
            "finish_reason": "stop",
            "message": {"role": "assistant", "content": "pong"}
        }],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
    });
    merge(&mut reply, extra);
    reply.to_string()
}

fn merge(into: &mut Value, over: Value) {
    match (into, over) {
        (Value::Object(into), Value::Object(over)) => {
            for (key, value) in over {
                match into.get_mut(&key) {
                    Some(slot) => merge(slot, value),
                    None => {
                        into.insert(key, value);
                    }
                }
            }
        }
        (Value::Array(into), Value::Array(over)) => {
            for (index, value) in over.into_iter().enumerate() {
                match into.get_mut(index) {
                    Some(slot) => merge(slot, value),
                    None => into.push(value),
                }
            }
        }
        (into, over) => *into = over,
    }
}

/// Every generation option one at a time, each a request a wire may refuse.
fn option_variants() -> Vec<GenerationOptions> {
    let mut variants: Vec<GenerationOptions> = [
        Reasoning::Off,
        Reasoning::Budget { tokens: 1024 },
        Effort::Minimal.into(),
        Effort::Low.into(),
        Effort::Medium.into(),
        Effort::High.into(),
        Effort::XHigh.into(),
        Effort::Max.into(),
    ]
    .into_iter()
    .map(|reasoning| GenerationOptions::default().reasoning(reasoning))
    .collect();
    for cache in [
        CacheRetention::None,
        CacheRetention::Short,
        CacheRetention::Long,
    ] {
        variants.push(GenerationOptions::default().cache(cache));
    }
    for tier in [
        ServiceTier::Auto,
        ServiceTier::Default,
        ServiceTier::Flex,
        ServiceTier::Priority,
    ] {
        variants.push(GenerationOptions::default().service_tier(tier));
    }
    variants.push(GenerationOptions::default().verbosity(Verbosity::Low));
    variants.push(GenerationOptions::default().parallel_tool_calls(true));
    variants.push(GenerationOptions::default().top_p(0.5));
    variants.push(GenerationOptions::default().seed(7));
    variants.push(GenerationOptions::default().stop(["END"]));
    variants
}

/// A request setting every field `CompletionRequest` owns.
fn full_request(options: GenerationOptions) -> CompletionRequest {
    let tool = ToolDefinition::new(
        ToolName::new("lookup").unwrap_or_else(|error| panic!("{error}")),
        "look a thing up",
        json!({"type": "object", "properties": {"q": {"type": "string"}}}),
    );
    CompletionRequest::new("hi")
        .preamble("be brief")
        .tool(tool)
        .tool_choice(ToolChoice::Required)
        .temperature(0.5)
        .max_tokens(8192)
        .options(options.on_unsupported(OnUnsupported::Ignore))
}

/// The JSON pointer of every leaf of `value` under `at`: a scalar, an
/// array or an empty object.
fn leaves(value: &Value, at: &str, out: &mut BTreeSet<String>) {
    match value {
        Value::Object(fields) if !fields.is_empty() => {
            for (key, field) in fields {
                let key = key.replace('~', "~0").replace('/', "~1");
                leaves(field, &format!("{at}/{key}"), out);
            }
        }
        _ => {
            out.insert(at.to_owned());
        }
    }
}

/// The leaves the request's own fields and every mapped generation option
/// write on each of `wires`, in both modes. A request a wire refuses adds
/// nothing.
fn reserved_leaves<W>(wires: &[W]) -> BTreeSet<String>
where
    W: Wire<Op = Completion, Payload = Encoded>,
{
    let mut reserved = BTreeSet::new();
    let mut requests = vec![full_request(GenerationOptions::default())];
    requests.extend(option_variants().into_iter().map(full_request));
    for wire in wires {
        for request in &requests {
            for mode in [Mode::Unary, Mode::Streaming] {
                if let Ok(body) = encoded_body(wire, request.clone(), mode) {
                    leaves(&body, "", &mut reserved);
                }
            }
        }
    }
    reserved
}

/// Assert `options`, fully set, writes no leaf the request or a mapped
/// generation option writes on any of `wires`: no two typed knobs set the
/// same JSON leaf. A provider leaf may sit beside a reserved one under the
/// same object (`/reasoning/exclude` beside `/reasoning/effort`), never on
/// it, above it or below it.
pub(crate) fn assert_no_reserved_leaf<P, W>(wires: &[W], options: &P::Options)
where
    P: ProviderExtension,
    W: Wire<Op = Completion, Payload = Encoded>,
{
    let reserved = reserved_leaves(wires);
    let sections = serde_json::to_value(options).unwrap_or_else(|error| panic!("{error}"));
    let Value::Object(sections) = sections else {
        panic!("{} options are an object of sections", P::PROVIDER);
    };
    assert!(!sections.is_empty(), "the options set no field");
    for (section, fields) in &sections {
        if fields.as_object().is_some_and(|fields| fields.is_empty()) {
            continue;
        }
        let mut written = BTreeSet::new();
        leaves(fields, "", &mut written);
        for leaf in &written {
            let clash = reserved.iter().find(|owned| {
                *owned == leaf
                    || owned.starts_with(&format!("{leaf}/"))
                    || leaf.starts_with(&format!("{owned}/"))
            });
            assert!(
                clash.is_none(),
                "{}.{section} writes `{leaf}`, which `{}` owns",
                P::PROVIDER,
                clash.map(String::as_str).unwrap_or_default()
            );
        }
    }
}
