//! Corpus census: what rig's `Usage` reports for every recorded completion.
//!
//! Every committed 200-status completion reply, unary and streamed, is
//! decoded through the provider's real wire exactly as it was recorded, and
//! rig's `Usage` for it is set beside the total the provider reported, when
//! it reported one. The census costs no provider traffic and covers every
//! provider directory in the corpus.
//!
//! It prints one `USAGE_CENSUS <provider>/<wire> {json}` line per provider
//! and wire: calls, calls with a cache read or write, calls with reasoning,
//! and the calls where cached plus written input exceeds input, where
//! reasoning exceeds output, where rig's total is not input plus output, and
//! where the provider's reported total is not input plus output. Set
//! `RIG_USAGE_CENSUS` to a path to also write every call as one JSON line.
//!
//! [`every_recorded_completion_is_censused`] fails closed on coverage, as
//! the cache-prefix census does: every 200-status reply is either a
//! completion the census decodes or an endpoint it names as carrying none,
//! and every provider directory yields decoded completions.
//! [`every_recorded_call_keeps_the_usage_contract`] asserts `Usage`'s
//! contract on every decoded call of every provider, and that rig's total is
//! the provider's wherever the provider reports one.

#![allow(clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use aws_sdk_bedrockruntime::config::{BehaviorVersion, Credentials, Region};
use aws_smithy_runtime_api::client::http::{
    HttpClient, HttpConnector, HttpConnectorFuture, HttpConnectorSettings, SharedHttpConnector,
};
use aws_smithy_runtime_api::client::orchestrator::{HttpRequest, HttpResponse};
use aws_smithy_runtime_api::client::result::ConnectorError;
use aws_smithy_runtime_api::client::runtime_components::RuntimeComponents;
use aws_smithy_runtime_api::http::StatusCode;
use aws_smithy_types::body::SdkBody;
use base64::{Engine, prelude::BASE64_STANDARD};
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use rig::DynModel;
use rig::bedrock::client::BedrockRuntime;
use rig::bedrock::completion::Converse;
use rig::completion::{CompletionRequest, Usage};
use rig::driver::Model;
use rig::operation::Completion;
use rig::providers::anthropic::wire::AnthropicConfig;
use rig::providers::copilot::CopilotConfig;
use rig::providers::gemini::GeminiConfig;
use rig::providers::openai::wire::{
    COHERE, DEEPSEEK, DOUBLEWORD, Dialect, GROQ, LLAMACPP, MISTRAL, OLLAMA, OPENAI, OPENROUTER,
    OpenAIConfig, PERPLEXITY, VENICE,
};
use rig::test_utils::{MockHttpResponse, SequencedHttpClient};
use rig::wire::{Framing, WireFrame};

/// Endpoints whose 200-status replies carry no completion, with the reason.
/// A reply at any other endpoint that is not a completion route fails the
/// coverage check rather than being skipped.
const NOT_COMPLETIONS: &[(&str, &str)] = &[
    ("/models", "model listing"),
    ("/api/tags", "Ollama model listing"),
    ("/props", "llama.cpp server properties"),
    ("/embeddings", "embeddings"),
    ("/embed", "embeddings"),
    (":batchEmbedContents", "Gemini embeddings"),
    (
        "/invoke",
        "Bedrock InvokeModel, recorded only for embeddings",
    ),
    ("/rerank", "reranking"),
    ("/images/generations", "image generation"),
    ("/image/generate", "image generation"),
    ("/audio/", "speech and transcription"),
    ("/files", "file upload"),
    (
        "/cachedContents",
        "Gemini cache resources: creation reports the tokens stored, not a completion",
    ),
];

#[derive(Deserialize)]
struct RecordedInteraction {
    when: RecordedRequest,
    then: RecordedResponse,
}

#[derive(Deserialize)]
struct RecordedRequest {
    path: String,
    method: String,
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
    #[serde(default)]
    body_encoding: Option<String>,
}

#[derive(Clone, Deserialize)]
struct RecordedHeader {
    name: String,
    value: String,
}

/// The wire a recorded completion was spoken on.
#[derive(Clone, Copy, Debug)]
enum Route {
    Messages,
    Chat(&'static Dialect),
    Responses(&'static Dialect),
    /// Copilot's wire picks Chat Completions or Responses by model; the
    /// recorded path says which it spoke.
    Copilot {
        responses: bool,
    },
    GenerateContent,
    Interactions,
    Converse,
    /// Cohere's native `/v2/chat`.
    CohereChat,
}

impl Route {
    /// The column this route is reported under.
    fn wire(self) -> &'static str {
        match self {
            Self::Messages => "messages",
            Self::Chat(_) | Self::Copilot { responses: false } => "chat",
            Self::Responses(_) | Self::Copilot { responses: true } => "responses",
            Self::GenerateContent => "generate_content",
            Self::Interactions => "interactions",
            Self::Converse => "converse",
            Self::CohereChat => "cohere_chat",
        }
    }
}

/// What the census makes of one 200-status reply.
enum Endpoint {
    Completion(Route),
    NotCompletion,
    Unknown,
}

fn chat_dialect(provider: &str) -> Option<&'static Dialect> {
    Some(match provider {
        "cohere" => &COHERE,
        "deepseek" => &DEEPSEEK,
        "doubleword" => &DOUBLEWORD,
        "groq" => &GROQ,
        "llamacpp" => &LLAMACPP,
        "mistral" => &MISTRAL,
        "mistralrs" | "openai" => &OPENAI,
        "ollama" => &OLLAMA,
        "openrouter" => &OPENROUTER,
        "perplexity" => &PERPLEXITY,
        "venice" => &VENICE,
        _ => return None,
    })
}

fn responses_dialect(provider: &str) -> Option<&'static Dialect> {
    Some(match provider {
        "chatgpt" => &rig::providers::chatgpt::DIALECT,
        "llamacpp" | "mistralrs" | "openai" => &OPENAI,
        "openrouter" => &OPENROUTER,
        "xai" => &rig::providers::xai::DIALECT,
        _ => return None,
    })
}

fn classify(provider: &str, method: &str, path: &str) -> Endpoint {
    if method != "POST" {
        // Reads and deletes of stored resources, model listings: nothing
        // here reads a completion back.
        return if method == "GET" && path.contains("/interactions/") {
            Endpoint::Unknown
        } else {
            Endpoint::NotCompletion
        };
    }
    let route = match provider {
        "anthropic" if path.ends_with("/messages") => Some(Route::Messages),
        "bedrock" if path.ends_with("/converse") || path.ends_with("/converse-stream") => {
            Some(Route::Converse)
        }
        "copilot" if path.ends_with("/chat/completions") => {
            Some(Route::Copilot { responses: false })
        }
        "copilot" if path.ends_with("/responses") => Some(Route::Copilot { responses: true }),
        "gemini"
            if path.ends_with(":generateContent") || path.ends_with(":streamGenerateContent") =>
        {
            Some(Route::GenerateContent)
        }
        "gemini" if path.ends_with("/interactions") => Some(Route::Interactions),
        "cohere" if path.ends_with("/v2/chat") => Some(Route::CohereChat),
        _ if path.ends_with("/chat/completions") => chat_dialect(provider).map(Route::Chat),
        _ if path.ends_with("/responses") => responses_dialect(provider).map(Route::Responses),
        _ => None,
    };
    match route {
        Some(route) => Endpoint::Completion(route),
        None if NOT_COMPLETIONS
            .iter()
            .any(|(fragment, _)| path.contains(fragment)) =>
        {
            Endpoint::NotCompletion
        }
        None => Endpoint::Unknown,
    }
}

/// One decoded completion.
#[derive(Serialize)]
struct Call {
    key: String,
    provider: String,
    wire: &'static str,
    streamed: bool,
    usage: Option<Usage>,
    reported_total: Option<u64>,
    error: Option<String>,
}

/// Per provider and wire counts.
#[derive(Default, Serialize)]
struct Row {
    calls: usize,
    streamed: usize,
    errors: usize,
    with_cache: usize,
    with_reasoning: usize,
    cache_exceeds_input: usize,
    reasoning_exceeds_output: usize,
    total_not_input_plus_output: usize,
    with_reported_total: usize,
    reported_total_not_input_plus_output: usize,
}

/// Cached plus written input exceeds input, or cache counts come without
/// an input count.
fn cache_exceeds_input(usage: &Usage) -> bool {
    let cache =
        usage.cached_input_tokens.unwrap_or(0) + usage.cache_creation_input_tokens.unwrap_or(0);
    match usage.input_tokens {
        Some(input) => cache > input,
        None => cache > 0,
    }
}

/// Reasoning exceeds output, or reasoning comes without an output count.
fn reasoning_exceeds_output(usage: &Usage) -> bool {
    let reasoning = usage.reasoning_tokens.unwrap_or(0);
    match usage.output_tokens {
        Some(output) => reasoning > output,
        None => reasoning > 0,
    }
}

/// Input plus output, when both were reported.
fn input_plus_output(usage: &Usage) -> Option<u64> {
    usage
        .input_tokens
        .zip(usage.output_tokens)
        .map(|(input, output)| input + output)
}

impl Row {
    fn add(&mut self, call: &Call) {
        self.calls += 1;
        self.streamed += usize::from(call.streamed);
        let Some(usage) = &call.usage else {
            self.errors += 1;
            return;
        };
        let cache =
            usage.cached_input_tokens.unwrap_or(0) + usage.cache_creation_input_tokens.unwrap_or(0);
        self.with_cache += usize::from(cache > 0);
        self.with_reasoning += usize::from(usage.reasoning_tokens.unwrap_or(0) > 0);
        self.cache_exceeds_input += usize::from(cache_exceeds_input(usage));
        self.reasoning_exceeds_output += usize::from(reasoning_exceeds_output(usage));
        self.total_not_input_plus_output +=
            usize::from(usage.total_tokens != input_plus_output(usage));
        if let Some(total) = call.reported_total {
            self.with_reported_total += 1;
            self.reported_total_not_input_plus_output +=
                usize::from(Some(total) != input_plus_output(usage));
        }
    }
}

/// The census of the whole corpus.
struct Census {
    calls: Vec<Call>,
    rows: BTreeMap<String, Row>,
    unknown: Vec<String>,
    provider_dirs: Vec<String>,
}

fn cassette_root() -> PathBuf {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/cassettes");
    assert!(root.is_dir(), "cassette root moved: {}", root.display());
    root
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

/// The model a recorded request addressed, so a wire that derives request
/// defaults from it encodes.
fn recorded_model(route: Route, path: &str, body: &Value) -> String {
    let from_path = match route {
        Route::GenerateContent => path
            .rsplit_once("/models/")
            .and_then(|(_, rest)| rest.split(':').next()),
        Route::Converse => path
            .strip_prefix("/model/")
            .and_then(|rest| rest.split('/').next()),
        _ => None,
    };
    from_path
        .map(|model| model.replace("%3A", ":"))
        .or_else(|| body["model"].as_str().map(str::to_owned))
        .unwrap_or_else(|| "census".to_owned())
}

/// Whether rig asked for this reply streamed.
fn recorded_streamed(route: Route, path: &str, body: &Value) -> bool {
    match route {
        Route::GenerateContent => path.ends_with(":streamGenerateContent"),
        Route::Converse => path.ends_with("/converse-stream"),
        _ => body["stream"] == Value::Bool(true),
    }
}

/// A Bedrock HTTP client that answers every request with one recorded reply.
#[derive(Clone, Debug)]
struct RecordedBedrockReply {
    status: u16,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
}

impl HttpConnector for RecordedBedrockReply {
    fn call(&self, _request: HttpRequest) -> HttpConnectorFuture {
        let reply = self.clone();
        HttpConnectorFuture::new(async move {
            let status = StatusCode::try_from(reply.status)
                .map_err(|error| ConnectorError::user(Box::new(error)))?;
            let mut response = HttpResponse::new(status, SdkBody::from(reply.body));
            for (name, value) in reply.headers {
                response
                    .headers_mut()
                    .try_append(name, value)
                    .map_err(|error| ConnectorError::user(Box::new(error)))?;
            }
            Ok(response)
        })
    }
}

impl HttpClient for RecordedBedrockReply {
    fn http_connector(
        &self,
        _settings: &HttpConnectorSettings,
        _components: &RuntimeComponents,
    ) -> SharedHttpConnector {
        SharedHttpConnector::new(self.clone())
    }
}

/// An HTTP client that answers the one request with the recorded reply.
fn recorded_http(headers: &[RecordedHeader], body: Vec<u8>) -> SequencedHttpClient {
    let mut header_map = http::HeaderMap::new();
    for header in headers {
        if let (Ok(name), Ok(value)) = (
            http::HeaderName::from_bytes(header.name.as_bytes()),
            http::HeaderValue::from_str(&header.value),
        ) {
            header_map.append(name, value);
        }
    }
    SequencedHttpClient::new([MockHttpResponse::SuccessWithHeaders(
        body.into(),
        header_map,
    )])
}

/// The Bedrock runtime over an SDK client that answers with the recorded reply.
fn recorded_bedrock(headers: &[RecordedHeader], body: Vec<u8>) -> BedrockRuntime {
    let reply = RecordedBedrockReply {
        status: 200,
        headers: headers
            .iter()
            .map(|header| (header.name.clone(), header.value.clone()))
            .collect(),
        body,
    };
    let config = aws_sdk_bedrockruntime::Config::builder()
        .behavior_version(BehaviorVersion::latest())
        .region(Region::new("us-east-1"))
        .credentials_provider(Credentials::new("census", "census", None, None, "census"))
        .http_client(reply)
        .build();
    BedrockRuntime::from(aws_sdk_bedrockruntime::Client::from_conf(config))
}

/// The recorded reply's route as a model whose transport replays it.
fn replaying_model(
    route: Route,
    model: String,
    headers: &[RecordedHeader],
    body: Vec<u8>,
) -> DynModel<Completion> {
    match route {
        Route::Converse => {
            Model::new(Converse::new(model), recorded_bedrock(headers, body)).erase()
        }
        Route::Messages => AnthropicConfig::new("census")
            .connect(recorded_http(headers, body))
            .completion(model)
            .erase(),
        Route::Chat(dialect) => OpenAIConfig::with_key(dialect, "census")
            .connect(recorded_http(headers, body))
            .chat(model)
            .erase(),
        Route::Responses(dialect) => {
            let mut config = OpenAIConfig::with_key(dialect, "census");
            if dialect.name == rig::providers::chatgpt::DIALECT.name {
                config = config.with_account_id("census");
            }
            config
                .connect(recorded_http(headers, body))
                .responses(model)
                .erase()
        }
        Route::Copilot { .. } => CopilotConfig::new("census")
            .connect(recorded_http(headers, body))
            .completion(model)
            .erase(),
        Route::GenerateContent => GeminiConfig::new("census")
            .connect(recorded_http(headers, body))
            .completion(model)
            .erase(),
        Route::Interactions => GeminiConfig::new("census")
            .connect(recorded_http(headers, body))
            .interactions(model)
            .erase(),
        Route::CohereChat => {
            let mut model = rig::providers::cohere::CohereConfig::new("census")
                .connect(recorded_http(headers, body))
                .completion(model);
            model.wire = model
                .wire
                .with_route(rig::providers::cohere::ChatRoute::Native);
            model.erase()
        }
    }
}

/// Decode one recorded reply the way rig read it.
async fn decode(model: DynModel<Completion>, streamed: bool) -> Result<Usage, String> {
    let mut request = CompletionRequest::from("census");
    // A wire that cannot default `max_tokens` for an unknown model still encodes.
    request.max_tokens = Some(1024);
    if !streamed {
        return model
            .call(request)
            .await
            .map(|response| response.usage)
            .map_err(|error| error.to_string());
    }
    let mut stream = model.stream(request).map_err(|error| error.to_string())?;
    while let Some(item) = stream.next().await {
        item.map_err(|error| error.to_string())?;
    }
    stream
        .finish()
        .await
        .map(|response| response.usage)
        .map_err(|error| error.to_string())
}

/// The payloads of a recorded reply, one JSON document per frame.
fn reply_documents(route: Route, streamed: bool, body: &[u8]) -> Vec<Value> {
    let texts: Vec<String> = match (route, streamed) {
        (Route::Converse, true) => {
            let mut input = bytes::Bytes::copy_from_slice(body);
            let mut payloads = Vec::new();
            while !input.is_empty() {
                let Ok(message) = aws_smithy_eventstream::frame::read_message_from(&mut input)
                else {
                    break;
                };
                payloads.push(String::from_utf8_lossy(message.payload()).into_owned());
            }
            payloads
        }
        (_, true) => frame_texts(Framing::Sse.split(body)),
        (_, false) => frame_texts(Framing::Whole.split(body)),
    };
    texts
        .iter()
        .filter_map(|text| serde_json::from_str(text).ok())
        .collect()
}

fn frame_texts(frames: Vec<WireFrame>) -> Vec<String> {
    frames
        .iter()
        .map(|frame| frame.as_str().into_owned())
        .collect()
}

/// The first usage object in `value`, searching depth first.
fn usage_object(value: &Value) -> Option<&serde_json::Map<String, Value>> {
    match value {
        Value::Object(map) => ["usage", "usageMetadata"]
            .iter()
            .find_map(|key| map.get(*key)?.as_object())
            .or_else(|| map.values().find_map(usage_object)),
        Value::Array(items) => items.iter().find_map(usage_object),
        _ => None,
    }
}

/// The total the provider reported on the last frame carrying usage, as
/// the decoders keep the last usage snapshot.
fn reported_total(documents: &[Value]) -> Option<u64> {
    let usage = documents.iter().rev().find_map(usage_object)?;
    ["total_tokens", "totalTokens", "totalTokenCount"]
        .iter()
        .find_map(|key| usage.get(*key)?.as_u64())
}

/// The census, decoded once and shared by both tests.
async fn census() -> &'static Census {
    static CENSUS: tokio::sync::OnceCell<Census> = tokio::sync::OnceCell::const_new();
    CENSUS.get_or_init(take_census).await
}

async fn take_census() -> Census {
    let root = cassette_root();
    let mut provider_dirs: Vec<String> = std::fs::read_dir(&root)
        .expect("cassette root is readable")
        .map(|entry| entry.expect("provider directory entry").path())
        .filter(|path| path.is_dir())
        .map(|path| {
            path.file_name()
                .expect("directory name")
                .to_string_lossy()
                .into_owned()
        })
        .collect();
    provider_dirs.sort();

    let mut calls = Vec::new();
    let mut unknown = Vec::new();
    for provider in &provider_dirs {
        let mut files = Vec::new();
        yaml_files(&root.join(provider), &mut files);
        files.sort();
        for file in files {
            let scenario = file
                .strip_prefix(&root)
                .expect("cassette under the root")
                .to_string_lossy()
                .replace('\\', "/");
            let contents = std::fs::read_to_string(&file).expect("cassette is readable");
            for (index, document) in serde_yaml::Deserializer::from_str(&contents).enumerate() {
                let Ok(interaction) = RecordedInteraction::deserialize(document) else {
                    continue;
                };
                if interaction.then.status != 200 {
                    continue;
                }
                let key = format!("{scenario}#{index}");
                let (method, path) = (&interaction.when.method, &interaction.when.path);
                let route = match classify(provider, method, path) {
                    Endpoint::Completion(route) => route,
                    Endpoint::NotCompletion => continue,
                    Endpoint::Unknown => {
                        unknown.push(format!("{key}: {method} {path}"));
                        continue;
                    }
                };
                let request: Value = interaction
                    .when
                    .body
                    .as_deref()
                    .and_then(|body| serde_json::from_str(body).ok())
                    .unwrap_or(Value::Null);
                let streamed = recorded_streamed(route, path, &request);
                let text = interaction.then.body.unwrap_or_default();
                let body = if interaction.then.body_encoding.as_deref() == Some("base64") {
                    BASE64_STANDARD
                        .decode(text.trim())
                        .expect("base64 cassette body decodes")
                } else {
                    text.into_bytes()
                };
                let reported_total = reported_total(&reply_documents(route, streamed, &body));
                let model = replaying_model(
                    route,
                    recorded_model(route, path, &request),
                    &interaction.then.header,
                    body,
                );
                let decoded = decode(model, streamed).await;
                calls.push(Call {
                    key,
                    provider: provider.clone(),
                    wire: route.wire(),
                    streamed,
                    usage: decoded.as_ref().ok().copied(),
                    reported_total,
                    error: decoded.err(),
                });
            }
        }
    }

    let mut rows: BTreeMap<String, Row> = BTreeMap::new();
    for call in &calls {
        rows.entry(format!("{}/{}", call.provider, call.wire))
            .or_default()
            .add(call);
    }
    Census {
        calls,
        rows,
        unknown,
        provider_dirs,
    }
}

fn report(census: &Census) {
    for (row, counts) in &census.rows {
        println!(
            "USAGE_CENSUS {row} {}",
            serde_json::to_string(counts).expect("row serializes")
        );
    }
    if let Some(path) = std::env::var_os("RIG_USAGE_CENSUS") {
        let lines = census
            .calls
            .iter()
            .map(|call| serde_json::to_string(call).expect("call serializes"))
            .collect::<Vec<_>>()
            .join("\n");
        std::fs::write(&path, lines + "\n").expect("census file is writable");
    }
}

/// Every provider directory is examined and every 200-status reply is either
/// decoded or named as carrying no completion.
#[tokio::test]
async fn every_recorded_completion_is_censused() {
    for (fragment, reason) in NOT_COMPLETIONS {
        assert!(
            !reason.trim().is_empty(),
            "NOT_COMPLETIONS entry `{fragment}` needs a reason"
        );
    }
    let census = census().await;
    report(census);

    assert!(
        census.unknown.is_empty(),
        "200-status replies at endpoints the census neither decodes nor names as carrying no \
         completion:\n{}",
        census.unknown.join("\n")
    );
    let unexamined: Vec<&String> = census
        .provider_dirs
        .iter()
        .filter(|provider| {
            !census
                .calls
                .iter()
                .any(|call| &call.provider == *provider && call.usage.is_some())
        })
        .collect();
    assert!(
        unexamined.is_empty(),
        "provider directories with no decoded completion: {unexamined:?}"
    );
    let decoded = census
        .calls
        .iter()
        .filter(|call| call.usage.is_some())
        .count();
    println!(
        "USAGE_CENSUS_TOTAL {}",
        json!({ "calls": census.calls.len(), "decoded": decoded })
    );
}

/// `Usage`'s contract on every decoded call of every provider: cached plus
/// written input within input, reasoning within output, the total input plus
/// output, and that total the provider's own wherever it reports one.
#[tokio::test]
async fn every_recorded_call_keeps_the_usage_contract() {
    let census = census().await;
    let mut violations = Vec::new();
    for call in &census.calls {
        let Some(usage) = &call.usage else {
            continue;
        };
        let mut broken = Vec::new();
        if cache_exceeds_input(usage) {
            broken.push("cached + written > input");
        }
        if reasoning_exceeds_output(usage) {
            broken.push("reasoning > output");
        }
        if usage.total_tokens != input_plus_output(usage) {
            broken.push("total != input + output");
        }
        if call.reported_total.is_some() && call.reported_total != input_plus_output(usage) {
            broken.push("the provider's total != input + output");
        }
        if !broken.is_empty() {
            violations.push(format!(
                "{} ({}/{}): {}; usage {usage:?}, provider total {:?}",
                call.key,
                call.provider,
                call.wire,
                broken.join(", "),
                call.reported_total
            ));
        }
    }
    let decoded = census
        .calls
        .iter()
        .filter(|call| call.usage.is_some())
        .count();
    assert!(decoded > 0, "the census decoded no completion");
    assert!(
        violations.is_empty(),
        "{} of {decoded} recorded calls break `Usage`'s contract; fix the provider's usage \
         mapping:\n{}",
        violations.len(),
        violations.join("\n")
    );
}
