//! Replays every recorded 200-status chat-completions reply through the chat
//! wire and projects each outcome onto the fields a response is compared on:
//! choice, usage, identifiers, finish reason, model, `raw` and the provider
//! request id. A reply that fails records its error kind instead.
//!
//! The projection is written against the public fields, not the response's
//! serde shape, so a change to how messages serialize does not move the
//! snapshot. Tool-call ids rig minted are projected as `"rig-issued"`, and a
//! provider's id is kept as sent.
//!
//! The snapshot pins one reply per reply shape and mode: the smallest, then
//! the first in path order. Every other reply is decoded too, and checked in
//! the test: `call` and `stream().finish()` fold its frames to the same
//! response.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use futures::StreamExt;
use serde::Deserialize;
use serde_json::{Value, json};

use rig::completion::CompletionResponse;
use rig::driver::{Exchange, Model, Opened, Opening, Transport};
use rig::error::ErrorKind;
use rig::message::AssistantContent;
use rig::providers::openai::wire::{
    COHERE, Chat, DEEPSEEK, DOUBLEWORD, Dialect, GROQ, LLAMACPP, MISTRAL, OLLAMA, OPENAI,
    OPENROUTER, OpenAIConfig, PERPLEXITY, VENICE,
};
use rig::test_utils::{MockHttpResponse, SequencedHttpClient};
use rig::wire::{Encoded, Framing, WireFrame};

/// Setting this regenerates the snapshot instead of comparing against it.
const REGENERATE: &str = "RIG_REGENERATE_PARITY";

/// The provider directories whose recorded chat-completions replies decode
/// through the chat wire.
pub const PROVIDERS: &[&str] = &[
    "cohere",
    "copilot",
    "deepseek",
    "doubleword",
    "groq",
    "llamacpp",
    "mistral",
    "mistralrs",
    "ollama",
    "openai",
    "openrouter",
    "perplexity",
    "venice",
];

/// The dialect each provider directory was recorded under.
pub fn dialect(provider: &str) -> Dialect {
    match provider {
        "cohere" => COHERE,
        "copilot" => rig::providers::copilot::wire::DIALECT,
        "deepseek" => DEEPSEEK,
        "doubleword" => DOUBLEWORD,
        "groq" => GROQ,
        "llamacpp" => LLAMACPP,
        "mistral" => MISTRAL,
        "mistralrs" | "openai" => OPENAI,
        "ollama" => OLLAMA,
        "openrouter" => OPENROUTER,
        "perplexity" => PERPLEXITY,
        "venice" => VENICE,
        other => panic!("no chat dialect recorded for `{other}`"),
    }
}

#[derive(Deserialize)]
struct RecordedInteraction {
    when: RecordedRequest,
    then: RecordedResponse,
}

#[derive(Deserialize)]
struct RecordedRequest {
    path: String,
    #[serde(default)]
    body: Option<String>,
}

#[derive(Deserialize)]
struct RecordedResponse {
    status: u16,
    #[serde(default)]
    header: Vec<RecordedHeader>,
    #[serde(default)]
    body: Option<String>,
}

#[derive(Deserialize)]
struct RecordedHeader {
    name: String,
    value: String,
}

/// One recorded chat-completions reply.
pub struct Interaction {
    /// `<scenario>.yaml#<index in the cassette>`.
    pub key: String,
    /// Whether rig asked for a streamed reply.
    pub streaming: bool,
    pub headers: http::HeaderMap,
    pub body: String,
}

fn cassette_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/cassettes")
}

fn snapshot_path(provider: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures/parity")
        .join(format!("{provider}.json"))
}

fn yaml_files(dir: &Path, found: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(dir).expect("cassette directory is readable") {
        let path = entry.expect("cassette directory entry").path();
        if path.is_dir() {
            yaml_files(&path, found);
        } else if path
            .extension()
            .is_some_and(|extension| extension == "yaml")
        {
            found.push(path);
        }
    }
}

/// Every 200-status chat-completions reply recorded for `provider`, in path
/// order.
pub fn interactions(provider: &str) -> Vec<Interaction> {
    interactions_at(provider, "/chat/completions")
}

/// Every 200-status reply recorded for `provider` on a path ending in
/// `endpoint`, in path order.
pub fn interactions_at(provider: &str, endpoint: &str) -> Vec<Interaction> {
    let root = cassette_root().join(provider);
    let mut files = Vec::new();
    yaml_files(&root, &mut files);
    files.sort();
    let mut found = Vec::new();
    for file in files {
        let scenario = file
            .strip_prefix(&root)
            .expect("cassette under its provider directory")
            .to_string_lossy()
            .replace('\\', "/");
        let contents = std::fs::read_to_string(&file).expect("cassette is readable");
        for (index, document) in serde_yaml::Deserializer::from_str(&contents).enumerate() {
            let interaction = RecordedInteraction::deserialize(document)
                .unwrap_or_else(|error| panic!("{scenario} should deserialize: {error}"));
            if interaction.then.status != 200 || !interaction.when.path.ends_with(endpoint) {
                continue;
            }
            let streaming = interaction
                .when
                .body
                .as_deref()
                .and_then(|body| serde_json::from_str::<Value>(body).ok())
                .is_some_and(|body| body["stream"] == Value::Bool(true));
            let mut headers = http::HeaderMap::new();
            for header in interaction.then.header {
                if let (Ok(name), Ok(value)) = (
                    http::HeaderName::from_bytes(header.name.as_bytes()),
                    http::HeaderValue::from_str(&header.value),
                ) {
                    headers.append(name, value);
                }
            }
            found.push(Interaction {
                key: format!("{scenario}#{index}"),
                streaming,
                headers,
                body: interaction.then.body.unwrap_or_default(),
            });
        }
    }
    found
}

/// Decode one recorded reply the way rig read it: a unary call, or a stream
/// drained to its end and finished. The first error fails either.
pub async fn decode(
    provider: &str,
    interaction: &Interaction,
) -> Result<CompletionResponse, ErrorKind> {
    let http = SequencedHttpClient::new([MockHttpResponse::SuccessWithHeaders(
        interaction.body.clone().into(),
        interaction.headers.clone(),
    )]);
    let wire = Chat::new(
        OpenAIConfig::with_key(&dialect(provider), "parity-key"),
        "parity",
    );
    let model = Model::new(wire, http);
    if !interaction.streaming {
        return model.call("parity").await.map_err(|error| error.kind());
    }
    let mut stream = model.stream("parity").map_err(|error| error.kind())?;
    while let Some(item) = stream.next().await {
        item.map_err(|error| error.kind())?;
    }
    stream.finish().await.map_err(|error| error.kind())
}

/// A recorded reply's frames, as they arrived, whatever mode reads them.
#[derive(Clone)]
struct Frames(Vec<WireFrame>, http::HeaderMap);

impl Transport<Chat> for Frames {
    fn send(&self, _payload: Encoded, _exchange: Exchange) -> Opening<WireFrame> {
        let frames = futures::stream::iter(self.0.clone().into_iter().map(Ok));
        Opening::ready(Opened::new(frames).with_http(http::StatusCode::OK, self.1.clone()))
    }
}

/// One recorded reply's frames folded by `call` and by `stream().finish()`:
/// one decoder and one fold, so the two must agree.
pub async fn both_paths(
    provider: &str,
    interaction: &Interaction,
) -> (
    Result<CompletionResponse, ErrorKind>,
    Result<CompletionResponse, ErrorKind>,
) {
    let framing = if interaction.streaming {
        Framing::Sse
    } else {
        Framing::Whole
    };
    let frames = Frames(
        framing.split(interaction.body.as_bytes()),
        interaction.headers.clone(),
    );
    let wire = Chat::new(
        OpenAIConfig::with_key(&dialect(provider), "parity-key"),
        "parity",
    );
    let model = Model::new(wire, frames);
    let called = model.call("parity").await.map_err(|error| error.kind());
    let streamed = match model.stream("parity") {
        Ok(mut stream) => {
            while stream.next().await.is_some() {}
            stream.finish().await.map_err(|error| error.kind())
        }
        Err(error) => Err(error.kind()),
    };
    (called, streamed)
}

/// Keys whose string values a reply shape keeps, as the coverage gate's
/// reply shapes do: bounded enums that select a decoder branch.
const DISCRIMINATORS: &[&str] = &[
    "finishReason",
    "finish_reason",
    "object",
    "reason",
    "role",
    "status",
    "stop_reason",
    "type",
];

/// The skeleton of `value`: keys sorted, scalars reduced to their type,
/// arrays to the set of their elements' skeletons, discriminators kept.
fn skeleton(value: &Value) -> String {
    match value {
        Value::Null => "null".to_owned(),
        Value::Bool(_) => "bool".to_owned(),
        Value::Number(_) => "num".to_owned(),
        Value::String(_) => "str".to_owned(),
        Value::Array(items) => {
            let items: BTreeSet<String> = items.iter().map(skeleton).collect();
            format!("[{}]", items.into_iter().collect::<Vec<_>>().join("|"))
        }
        Value::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            let fields: Vec<String> = keys
                .into_iter()
                .map(|key| match map.get(key) {
                    Some(Value::String(text)) if DISCRIMINATORS.contains(&key.as_str()) => {
                        format!("{key}:{text:?}")
                    }
                    Some(child) => format!("{key}:{}", skeleton(child)),
                    None => key.clone(),
                })
                .collect();
            format!("{{{}}}", fields.join(","))
        }
    }
}

/// A recorded reply's shape: its mode and the skeleton of its body, or of
/// each of its stream's events.
pub fn reply_shape(interaction: &Interaction) -> String {
    let frame_shape = |text: &str| match serde_json::from_str::<Value>(text) {
        Ok(value) => skeleton(&value),
        Err(_) if text == "[DONE]" => text.to_owned(),
        Err(_) => "text".to_owned(),
    };
    if interaction.streaming {
        let events: BTreeSet<String> = Framing::Sse
            .split(interaction.body.as_bytes())
            .iter()
            .map(|frame| frame_shape(&frame.as_str()))
            .collect();
        format!(
            "streaming[{}]",
            events.into_iter().collect::<Vec<_>>().join("|")
        )
    } else {
        format!("unary{}", frame_shape(&interaction.body))
    }
}

/// The replies the snapshot pins: per reply shape, the smallest, then the
/// first in path order.
pub fn pins(interactions: &[Interaction]) -> BTreeSet<String> {
    let mut chosen: BTreeMap<String, (usize, &str)> = BTreeMap::new();
    for interaction in interactions {
        let candidate = (interaction.body.len(), interaction.key.as_str());
        chosen
            .entry(reply_shape(interaction))
            .and_modify(|best| *best = (*best).min(candidate))
            .or_insert(candidate);
    }
    chosen
        .into_values()
        .map(|(_, key)| key.to_owned())
        .collect()
}

/// The committed snapshot of `provider`'s replies.
pub fn snapshot(provider: &str) -> BTreeMap<String, Value> {
    let path = snapshot_path(provider);
    serde_json::from_str(&std::fs::read_to_string(&path).unwrap_or_else(|error| {
        panic!(
            "{} is unreadable ({error}); run with {REGENERATE}=1",
            path.display()
        )
    }))
    .expect("snapshot is JSON")
}

/// The compared fields of one outcome.
pub fn project(streaming: bool, outcome: &Result<CompletionResponse, ErrorKind>) -> Value {
    let mode = if streaming { "streaming" } else { "unary" };
    match outcome {
        Err(kind) => json!({ "mode": mode, "error": kind }),
        Ok(response) => json!({
            "mode": mode,
            "choice": response.choice.iter().map(project_content).collect::<Vec<_>>(),
            "usage": response.usage,
            "response_id": response.response_id(),
            "finish_reason": response.finish_reason(),
            "model": response.model(),
            "provider_request_id": response.provider_request_id,
            "raw": response.raw,
        }),
    }
}

fn project_content(content: &AssistantContent) -> Value {
    match content {
        AssistantContent::Text(text) => {
            let mut projected = json!({
                "text": text.text,
                "native": text.native,
            });
            if !text.citations().is_empty() {
                projected["citations"] = json!(text.citations());
            }
            projected
        }
        AssistantContent::ToolCall(call) => {
            let id = match call.id.provider() {
                Some(provider) => provider.as_str().to_owned(),
                None => "rig-issued".to_owned(),
            };
            json!({
                "tool_call": {
                    "id": id,
                    "name": call.function.name,
                    "arguments": call.function.arguments,
                    "native": call.native,
                }
            })
        }
        AssistantContent::Reasoning(reasoning) => json!({
            "reasoning": {
                "text": reasoning.text,
                "redacted": reasoning.redacted,
                "native": reasoning.native,
            }
        }),
        AssistantContent::Image(image) => json!({ "image": image }),
        AssistantContent::Opaque(opaque) => json!({ "opaque": opaque }),
        other => json!({ "other": other }),
    }
}

/// Decode every recorded reply for `provider` and compare the pinned ones'
/// projections with the committed snapshot, or rewrite it when
/// [`REGENERATE`] is set.
pub async fn check(provider: &str) {
    let recorded = interactions(provider);
    assert!(
        !recorded.is_empty(),
        "no chat-completions replies recorded for {provider}"
    );
    let pinned = pins(&recorded);
    let mut actual = BTreeMap::new();
    let mut seen = BTreeSet::new();
    for interaction in &recorded {
        assert!(
            seen.insert(interaction.key.clone()),
            "duplicate key {}",
            interaction.key
        );
        let outcome = decode(provider, interaction).await;
        if pinned.contains(&interaction.key) {
            actual.insert(
                interaction.key.clone(),
                project(interaction.streaming, &outcome),
            );
        }
    }
    let path = snapshot_path(provider);
    if std::env::var_os(REGENERATE).is_some() {
        std::fs::create_dir_all(path.parent().expect("snapshot directory"))
            .expect("snapshot directory is writable");
        let mut text = serde_json::to_string_pretty(&actual).expect("snapshot serializes");
        text.push('\n');
        std::fs::write(&path, text).expect("snapshot is writable");
        return;
    }
    let expected = snapshot(provider);
    let mut diffs = Vec::new();
    for key in expected.keys().chain(actual.keys()) {
        let (before, after) = (expected.get(key), actual.get(key));
        if before == after || diffs.iter().any(|(seen, _)| seen == key) {
            continue;
        }
        let fields = match (before, after) {
            (Some(Value::Object(before)), Some(Value::Object(after))) => before
                .keys()
                .chain(after.keys())
                .filter(|field| before.get(*field) != after.get(*field))
                .cloned()
                .collect::<std::collections::BTreeSet<_>>()
                .into_iter()
                .collect::<Vec<_>>()
                .join(", "),
            (None, _) => "not in the snapshot".to_owned(),
            (_, None) => "no longer recorded".to_owned(),
            _ => "shape".to_owned(),
        };
        diffs.push((key.clone(), fields));
    }
    assert!(
        diffs.is_empty(),
        "{provider}: {} of {} replies differ from {}:\n{}",
        diffs.len(),
        actual.len(),
        path.display(),
        diffs
            .iter()
            .map(|(key, fields)| format!("  {key}: {fields}"))
            .collect::<Vec<_>>()
            .join("\n")
    );
}

/// The chat wire `provider` was recorded under.
pub fn wire(provider: &str) -> Chat {
    Chat::new(
        OpenAIConfig::with_key(&dialect(provider), "parity-key"),
        "parity",
    )
}

/// `text` in two pieces, as a stream delivers it.
fn halves(text: &str) -> [String; 2] {
    let at = text
        .char_indices()
        .map(|(at, _)| at)
        .nth(text.chars().count() / 2)
        .unwrap_or(text.len());
    [text[..at].to_owned(), text[at..].to_owned()]
}

/// A whole `chat.completion` body restated as the chunks a stream of the
/// same turn carries: the message's other fields first, then its reasoning
/// in two pieces (every reasoning key in one delta, as a relaying gateway
/// sends them), each reasoning detail whole, its text in two pieces, each
/// call opened then its arguments in two pieces, and the finish reason and
/// usage last.
pub fn restate_chat(body: &Value) -> Vec<WireFrame> {
    let chunk = |delta: Value, finish: &Value| {
        let mut choice = json!({"index": 0, "delta": delta, "finish_reason": finish});
        if let Some(native) = body["choices"][0]
            .get("native_finish_reason")
            .filter(|_| !finish.is_null())
        {
            choice["native_finish_reason"] = native.clone();
        }
        json!({
            "id": body["id"],
            "model": body["model"],
            "object": "chat.completion.chunk",
            "choices": [choice],
        })
    };
    let mut message = body["choices"][0]["message"]
        .as_object()
        .cloned()
        .unwrap_or_default();
    let mut deltas = Vec::new();
    let mut reasoning = [serde_json::Map::new(), serde_json::Map::new()];
    for key in ["reasoning_content", "reasoning", "reasoning_text"] {
        if let Some(Value::String(text)) = message.get_mut(key) {
            for (delta, piece) in reasoning.iter_mut().zip(halves(text)) {
                delta.insert(key.to_owned(), json!(piece));
            }
            text.clear();
        }
    }
    deltas.extend(
        reasoning
            .into_iter()
            .filter(|delta| !delta.is_empty())
            .map(Value::Object),
    );
    if let Some(Value::Array(details)) = message.shift_remove("reasoning_details") {
        deltas.extend(
            details
                .into_iter()
                .map(|detail| json!({ "reasoning_details": [detail] })),
        );
    }
    if let Some(Value::String(text)) = message.get_mut("content") {
        deltas.extend(halves(text).map(|piece| json!({ "content": piece })));
        text.clear();
    }
    if let Some(Value::Array(calls)) = message.shift_remove("tool_calls") {
        for (index, call) in calls.into_iter().enumerate() {
            let mut opening = call.clone();
            opening["index"] = json!(index);
            let arguments = call["function"]["arguments"].as_str().map(halves);
            if arguments.is_some() {
                opening["function"]["arguments"] = json!("");
            }
            deltas.push(json!({ "tool_calls": [opening] }));
            deltas.extend(arguments.into_iter().flatten().map(|piece| {
                json!({ "tool_calls": [{"index": index, "function": {"arguments": piece}}] })
            }));
        }
    }
    let mut frames = vec![chunk(Value::Object(message), &Value::Null)];
    frames.extend(deltas.into_iter().map(|delta| chunk(delta, &Value::Null)));
    let mut last = chunk(json!({}), &body["choices"][0]["finish_reason"]);
    last["usage"] = body["usage"].clone();
    // The fields a reply states beside its choices, such as Perplexity's
    // `citations` or Venice's `cost`, as a stream states them on a chunk.
    for (key, value) in body.as_object().into_iter().flatten() {
        if !matches!(
            key.as_str(),
            "id" | "model" | "object" | "choices" | "usage"
        ) {
            last[key] = value.clone();
        }
    }
    frames.push(last);
    frames
        .into_iter()
        .map(|frame| WireFrame::Text(frame.to_string()))
        .chain([WireFrame::Text("[DONE]".to_owned())])
        .collect()
}
