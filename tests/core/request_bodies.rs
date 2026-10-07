//! What every completion wire and dialect sends, pinned. For a corpus of
//! model ids (every catalog id and public model constant, and the other
//! spellings gateways, Bedrock, Vertex and callers use for them) and a set
//! of request shapes, each wire encodes the request and the canonical JSON
//! of its body (and of the headers and URL, which the catalog can drive) is
//! compared with the committed golden. A body that changes fails here, so a
//! change to what rig sends must be deliberate: list it in
//! `crates/rig-core/TYPED_OPTIONS.md` section 12.0, then regenerate with
//! `RIG_REGENERATE_REQUEST_BODIES=1` (all features, so every companion wire
//! is encoded) and review the golden's diff.
//!
//! The fixtures live in `tests/fixtures/request_bodies/`: `corpus.txt`, one
//! model id per line, and `bodies.txt`, the golden. The golden is
//! deduplicated: a model id inside an output reads `{model}`, each distinct
//! output is stored once under a stable hash, each wire's distinct rows of
//! outputs (one per shape) are its classes, and each model id names its
//! class on every wire.

#[path = "request_bodies/tests.rs"]
mod tests;

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use rig::completion::{
    CacheRetention, CompletionRequest, Effort, GenerationOptions, Message, Reasoning,
    ToolDefinition,
};
use rig::message::{
    AssistantContent, AssistantMessage, DocumentSourceKind, Image, ImageMediaType, ToolChoice,
    ToolName, ToolResultContent, UserContent,
};
use rig_core::operation::Completion;
use rig_core::wire::{Body, Encoded, Mode, Operation, Wire};
use serde_json::{Value, json};

/// Setting this rewrites the golden (and adds new catalog ids to the
/// corpus) instead of comparing.
const REGENERATE: &str = "RIG_REGENERATE_REQUEST_BODIES";

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/request_bodies")
        .join(name)
}

/// The committed corpus of model ids.
fn corpus() -> Vec<String> {
    let path = fixture("corpus.txt");
    std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("{} is unreadable: {error}", path.display()))
        .lines()
        .filter(|line| !line.is_empty())
        .map(str::to_owned)
        .collect()
}

/// `value` with every object's keys sorted, so key order never shows.
fn canonical(value: &Value) -> Value {
    match value {
        Value::Object(map) => {
            let sorted: BTreeMap<&String, Value> = map
                .iter()
                .map(|(key, value)| (key, canonical(value)))
                .collect();
            Value::Object(
                sorted
                    .into_iter()
                    .map(|(key, value)| (key.clone(), value))
                    .collect(),
            )
        }
        Value::Array(items) => Value::Array(items.iter().map(canonical).collect()),
        other => other.clone(),
    }
}

/// An HTTP request as data: its method, URL, headers (with the per-request
/// random ids blanked) and JSON body.
fn http(encoded: &Encoded) -> Value {
    let request = &encoded.request;
    let mut headers: BTreeMap<String, String> = BTreeMap::new();
    for (name, value) in request.headers() {
        let value = match name.as_str() {
            "session_id" | "x-request-id" | "x-interaction-id" => "<random>".to_owned(),
            _ => String::from_utf8_lossy(value.as_bytes()).into_owned(),
        };
        headers
            .entry(name.as_str().to_owned())
            .and_modify(|joined| {
                joined.push_str(" | ");
                joined.push_str(&value);
            })
            .or_insert(value);
    }
    let body = match request.body() {
        Body::Bytes(bytes) if bytes.is_empty() => Value::Null,
        Body::Bytes(bytes) => serde_json::from_slice(bytes)
            .unwrap_or_else(|_| Value::String(String::from_utf8_lossy(bytes).into_owned())),
        _ => Value::String("<not bytes>".to_owned()),
    };
    json!({
        "method": request.method().as_str(),
        "uri": request.uri().to_string(),
        "headers": headers,
        "body": body,
    })
}

/// What `wire` sends for `request`, prepared as the driver prepares it, as
/// one line: canonical JSON, or `ERR` and the refusal.
fn sent<W: Wire<Op = Completion>>(
    wire: &W,
    request: CompletionRequest,
    show: &dyn Fn(&W::Payload) -> Value,
) -> String {
    let result = Completion::prepare(request, &wire.describe())
        .and_then(|request| Ok(wire.encode(request, Mode::Unary)?))
        .map(|payload| show(&payload));
    match result {
        Ok(value) => canonical(&value).to_string(),
        Err(error) => format!("ERR {error}"),
    }
}

fn png() -> Image {
    Image {
        data: DocumentSourceKind::base64("iVBORw0KGgo="),
        media_type: Some(ImageMediaType::PNG),
        ..Image::default()
    }
}

fn lookup_tool() -> ToolDefinition {
    ToolDefinition {
        name: ToolName::new("lookup").expect("a tool name"),
        description: "Look something up.".into(),
        parameters: json!({"type": "object", "properties": {"q": {"type": "string"}}}),
    }
}

/// The request shapes, by name: no option set, then sampling, tools, an
/// image and reasoning in history, and the reasoning and cache options.
fn shapes() -> Vec<(&'static str, CompletionRequest)> {
    let base = || CompletionRequest::new("hi");
    let call = AssistantContent::tool_call(
        "call_1",
        ToolName::new("lookup").expect("a tool name"),
        json!({"q": "x"}),
    );
    let AssistantContent::ToolCall(tool_call) = &call else {
        unreachable!("a tool call")
    };
    let result = tool_call.result(vec![ToolResultContent::text("found")]);
    let image_history = vec![
        Message::User {
            content: vec![UserContent::text("look"), UserContent::Image(png())],
        },
        Message::assistant("ok"),
    ];
    let reasoning_history = vec![
        Message::user("find x"),
        Message::Assistant(AssistantMessage::new(vec![
            AssistantContent::reasoning("I should look it up."),
            call.clone(),
        ])),
        Message::tool_results(vec![result]),
        Message::Assistant(AssistantMessage::new(vec![
            AssistantContent::reasoning("Done."),
            AssistantContent::text("x is found"),
        ])),
        Message::system("later system"),
    ];
    let with = |options: GenerationOptions| base().max_tokens(4096).options(options);
    vec![
        ("plain", base()),
        ("max", base().max_tokens(1000)),
        (
            "sampling",
            base().max_tokens(1000).temperature(0.5).top_p(0.9),
        ),
        ("temp", base().temperature(0.5)),
        ("tools", base().max_tokens(1000).tool(lookup_tool())),
        (
            "required",
            base()
                .max_tokens(1000)
                .tool(lookup_tool())
                .tool_choice(ToolChoice::Required),
        ),
        (
            "image",
            CompletionRequest::new("next")
                .max_tokens(1000)
                .messages(image_history)
                .preamble("sys"),
        ),
        (
            "reasoning_history",
            CompletionRequest::new("next")
                .max_tokens(1000)
                .tool(lookup_tool())
                .messages(reasoning_history)
                .preamble("sys"),
        ),
        (
            "off",
            with(GenerationOptions::default().reasoning(Reasoning::Off)),
        ),
        (
            "high",
            with(GenerationOptions::default().reasoning(Effort::High)),
        ),
        (
            "budget",
            with(GenerationOptions::default().reasoning(Reasoning::Budget { tokens: 2048 })),
        ),
        (
            "cache_long",
            with(GenerationOptions::default().cache(CacheRetention::Long)),
        ),
    ]
}

/// One wire and dialect: its key, and how it encodes a request for a model.
struct Encoder {
    key: String,
    encode: Box<dyn Fn(&str, CompletionRequest) -> String + Send + Sync>,
}

fn encoder<W, F, S>(key: impl Into<String>, make: F, show: S) -> Encoder
where
    W: Wire<Op = Completion> + 'static,
    F: Fn(&str) -> W + Send + Sync + 'static,
    S: Fn(&W::Payload) -> Value + Send + Sync + 'static,
{
    Encoder {
        key: key.into(),
        encode: Box::new(move |model, request| sent(&make(model), request, &show)),
    }
}

/// A transport the client-built wires are created over; nothing is sent.
fn transport() -> rig_core::test_utils::RecordingHttpClient {
    rig_core::test_utils::RecordingHttpClient::new("{}")
}

/// Every completion wire and dialect this build has, keyed `wire|dialect`.
fn encoders() -> Vec<Encoder> {
    use rig::providers::anthropic::wire as anthropic;
    use rig::providers::openai::responses_api::wire::Responses;
    use rig::providers::openai::wire as openai;
    use rig::providers::{cohere, copilot, gemini, ollama};

    let mut chat: Vec<(String, &'static openai::Dialect)> = openai::all()
        .map(|dialect| (dialect.name.to_owned(), dialect))
        .collect();
    chat.push(("zai-coding".to_owned(), &openai::ZAI_CODING));
    chat.push(("minimax-china".to_owned(), &openai::MINIMAX_CHINA));
    chat.push(("moonshot-china".to_owned(), &openai::MOONSHOT_CHINA));
    let mut out = Vec::new();
    for (name, dialect) in &chat {
        let dialect: &'static openai::Dialect = dialect;
        out.push(encoder(
            format!("chat|{name}"),
            move |model| openai::Chat::new(openai::OpenAIConfig::with_key(dialect, "k"), model),
            http,
        ));
    }
    for (name, dialect) in &chat {
        if ![
            "openai",
            "azure.openai",
            "openrouter",
            "xai",
            "copilot",
            "chatgpt",
        ]
        .contains(&name.as_str())
        {
            continue;
        }
        let dialect: &'static openai::Dialect = dialect;
        out.push(encoder(
            format!("resp|{name}"),
            move |model| Responses::new(openai::OpenAIConfig::with_key(dialect, "k"), model),
            http,
        ));
    }
    out.push(encoder(
        "route|openai",
        |model| {
            openai::OpenAiWire::new(openai::OpenAIConfig::with_key(&openai::OPENAI, "k"), model)
        },
        http,
    ));
    out.push(encoder(
        "copilot|copilot",
        |model| {
            copilot::CopilotConfig::new("k")
                .connect(transport())
                .completion(model)
                .wire
        },
        http,
    ));
    for dialect in anthropic::all() {
        out.push(encoder(
            format!("msgs|{}", dialect.name),
            move |model| {
                anthropic::AnthropicConfig::with_key(dialect, "k")
                    .connect(transport())
                    .completion(model)
                    .wire
            },
            http,
        ));
    }
    out.push(encoder(
        "gemini|rest",
        |model| gemini::completion::GenerateContent::new(gemini::GeminiConfig::new("k"), model),
        http,
    ));
    out.push(encoder(
        "gemini|interactions",
        |model| gemini::interactions_api::Interactions::new(gemini::GeminiConfig::new("k"), model),
        http,
    ));
    out.push(encoder(
        "cohere|chat",
        |model| cohere::wire::CohereConfig::new("k").completion(model),
        http,
    ));
    out.push(encoder(
        "cohere|native",
        |model| cohere::chat::NativeChat::new(cohere::wire::CohereConfig::new("k"), model),
        http,
    ));
    out.push(encoder(
        "ollama|native",
        |model| ollama::Chat::new(ollama::OllamaConfig::new(), model),
        http,
    ));
    companions(&mut out);
    out
}

/// The companion crates' wires this build enables.
fn companions(out: &mut Vec<Encoder>) {
    #[cfg(feature = "bedrock")]
    out.push(encoder(
        "bedrock|converse",
        |model: &str| rig::bedrock::completion::Converse::new(model),
        |payload: &rig::bedrock::completion::ConverseRequest| {
            json!({
                "model": payload.model,
                "body": serde_json::to_value(&payload.body).unwrap_or(Value::Null),
            })
        },
    ));
    #[cfg(feature = "vertexai")]
    out.push(encoder(
        "vertex|generate",
        |model: &str| rig::vertexai::completion::GenerateContent::new(model),
        |payload| {
            serde_json::to_value(payload)
                .unwrap_or_else(|error| Value::String(format!("unserializable: {error}")))
        },
    ));
    #[cfg(feature = "gemini-grpc")]
    out.push(encoder(
        "grpc|generate",
        |model: &str| rig::gemini_grpc::completion::GenerateContent::new(model),
        |payload| Value::String(format!("{payload:?}")),
    ));
    #[cfg(feature = "candle")]
    out.push(encoder(
        "candle|qwen3",
        |model| rig::candle::Generation {
            model: model.to_owned(),
            protocol: rig::candle::ConversationProtocol::Qwen3,
        },
        |payload: &rig::candle::CandleRequest| {
            json!({ "params": serde_json::to_value(&payload.params).unwrap_or(Value::Null) })
        },
    ));
    let _ = out;
}

/// Whether this build encodes every wire the golden holds.
fn every_wire_built() -> bool {
    cfg!(all(
        feature = "bedrock",
        feature = "vertexai",
        feature = "gemini-grpc",
        feature = "candle"
    ))
}

/// Every output, by wire, model id and shape: `{model}` stands for the id.
type Outputs = BTreeMap<String, BTreeMap<String, Vec<String>>>;

/// Encode the corpus on every wire, one thread per wire at a time.
fn encode_all(models: &[String], encoders: &[Encoder]) -> Outputs {
    let shapes = shapes();
    let next = std::sync::atomic::AtomicUsize::new(0);
    let results = std::sync::Mutex::new(Outputs::new());
    let threads = std::thread::available_parallelism().map_or(4, usize::from);
    std::thread::scope(|scope| {
        for _ in 0..threads {
            scope.spawn(|| {
                loop {
                    let index = next.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    let Some(encoder) = encoders.get(index) else {
                        break;
                    };
                    let mut by_model = BTreeMap::new();
                    for model in models {
                        let quoted = serde_json::to_string(model).expect("a string");
                        let escaped = &quoted[1..quoted.len() - 1];
                        let outputs = shapes
                            .iter()
                            .map(|(_, request)| {
                                (encoder.encode)(model, request.clone()).replace(escaped, "{model}")
                            })
                            .collect();
                        by_model.insert(model.clone(), outputs);
                    }
                    results
                        .lock()
                        .expect("results")
                        .insert(encoder.key.clone(), by_model);
                }
            });
        }
    });
    results.into_inner().expect("results")
}

/// A stable 64-bit FNV-1a hash of `text`, as 16 hex digits.
fn digest(text: &str) -> String {
    let hash = text.bytes().fold(0xcbf2_9ce4_8422_2325_u64, |hash, byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3)
    });
    format!("{hash:016x}")
}

/// Render `outputs` as the golden.
fn render(outputs: &Outputs, models: &[String]) -> String {
    let shape_names: Vec<&str> = shapes().into_iter().map(|(name, _)| name).collect();
    let mut bodies = BTreeMap::new();
    let mut classes: BTreeMap<&str, Vec<Vec<String>>> = BTreeMap::new();
    let mut class_of: BTreeMap<&str, BTreeMap<&str, usize>> = BTreeMap::new();
    for (wire, by_model) in outputs {
        let rows: BTreeSet<Vec<String>> = by_model
            .values()
            .map(|row| {
                row.iter()
                    .map(|output| {
                        let hash = digest(output);
                        bodies.insert(hash.clone(), output.clone());
                        hash
                    })
                    .collect()
            })
            .collect();
        let rows: Vec<Vec<String>> = rows.into_iter().collect();
        let index: BTreeMap<&str, usize> = by_model
            .iter()
            .map(|(model, row)| {
                let hashes: Vec<String> = row.iter().map(|output| digest(output)).collect();
                let class = rows.iter().position(|known| *known == hashes).unwrap_or(0);
                (model.as_str(), class)
            })
            .collect();
        class_of.insert(wire, index);
        classes.insert(wire, rows);
    }
    let mut text = String::from(
        "# Generated by tests/core/request_bodies.rs with RIG_REGENERATE_REQUEST_BODIES=1.\n\
         # What each completion wire sends per model id and request shape: see that file.\n",
    );
    text.push_str(&format!("shapes\t{}\n", shape_names.join(" ")));
    text.push_str(&format!(
        "wires\t{}\n",
        outputs.keys().cloned().collect::<Vec<_>>().join(" ")
    ));
    text.push_str("\n[outputs]\n");
    for (hash, output) in &bodies {
        text.push_str(&format!("{hash}\t{output}\n"));
    }
    text.push_str("\n[classes]\n");
    for (wire, rows) in &classes {
        for (index, row) in rows.iter().enumerate() {
            text.push_str(&format!("{wire}\t{index}\t{}\n", row.join(" ")));
        }
    }
    text.push_str("\n[models]\n");
    for model in models {
        let row: Vec<String> = outputs
            .keys()
            .map(|wire| {
                class_of
                    .get(wire.as_str())
                    .and_then(|index| index.get(model.as_str()))
                    .map_or_else(|| "-".to_owned(), ToString::to_string)
            })
            .collect();
        text.push_str(&format!("{model}\t{}\n", row.join(" ")));
    }
    text
}

/// The golden read back into outputs, by wire, model id and shape.
fn parse(text: &str) -> (Vec<String>, Outputs) {
    let mut shapes = Vec::new();
    let mut wires = Vec::new();
    let mut bodies = BTreeMap::new();
    let mut classes: BTreeMap<(String, usize), Vec<String>> = BTreeMap::new();
    let mut outputs = Outputs::new();
    let mut section = "";
    for line in text.lines() {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if line.starts_with('[') {
            section = line;
            continue;
        }
        let (head, rest) = line.split_once('\t').expect("a tab-separated golden line");
        match section {
            "" if head == "shapes" => shapes = rest.split(' ').map(str::to_owned).collect(),
            "" if head == "wires" => wires = rest.split(' ').map(str::to_owned).collect(),
            "[outputs]" => {
                bodies.insert(head.to_owned(), rest.to_owned());
            }
            "[classes]" => {
                let (index, row) = rest.split_once('\t').expect("a class row");
                let row = row
                    .split(' ')
                    .map(|hash| bodies.get(hash).cloned().expect("a known output"))
                    .collect();
                classes.insert((head.to_owned(), index.parse().expect("an index")), row);
            }
            "[models]" => {
                for (wire, class) in wires.iter().zip(rest.split(' ')) {
                    let Ok(class) = class.parse::<usize>() else {
                        continue;
                    };
                    let row = classes
                        .get(&(wire.clone(), class))
                        .cloned()
                        .expect("a known class");
                    outputs
                        .entry(wire.clone())
                        .or_default()
                        .insert(head.to_owned(), row);
                }
            }
            other => panic!("unexpected golden line in {other:?}: {line}"),
        }
    }
    (shapes, outputs)
}
