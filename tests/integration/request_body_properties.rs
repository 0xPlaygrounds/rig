//! What every completion wire and dialect sends, checked by property. Every
//! model id the built-in catalog lists goes through every completion wire
//! this build has, for a few request shapes, and each request either
//! encodes to a JSON body or is refused with a reason. A body addresses the
//! requested model, sends no `temperature` or reasoning the model's catalog
//! entry says it does not take, and goes to the dialect's completion path with the credential in
//! the dialect's header, marked sensitive.
//!
//! The ids are the catalog's: `core::catalog_constants` holds every public
//! model constant to a catalog entry, so the constants are among them. A
//! wire given an id its vendor does not list finds no catalog entry, and
//! only the properties that need none apply.

use std::panic::{AssertUnwindSafe, catch_unwind};

use rig::catalog::{Catalog, ModelSpec, ReasoningSupport, Sampling};
use rig::completion::{
    CompletionRequest, Effort, GenerationOptions, Message, OnUnsupported, ToolDefinition,
};
use rig::message::{DocumentSourceKind, Image, ImageMediaType, ToolName, UserContent};
use rig_core::operation::Completion;
use rig_core::wire::{Body, Encoded, Mode, Operation, Wire};
use serde_json::{Value, json};

/// The credential every wire is built with, so a header or URL carrying it
/// is found by value.
const KEY: &str = "props-credential-1234";

/// What one wire did with one request.
enum Outcome {
    /// The wire refused it, for this reason.
    Refused(String),
    /// The wire built a request.
    Sent(Sent),
}

struct Sent {
    /// The URL, for an HTTP wire.
    uri: Option<String>,
    /// Each header's name, value and whether it is marked sensitive.
    headers: Vec<(String, String, bool)>,
    /// The body as JSON, or why it is not JSON.
    body: Result<Value, String>,
}

/// Where an HTTP wire's settings say a request goes, and the header its
/// credential travels in.
#[derive(Clone)]
struct Route {
    base: String,
    /// The path the settings name, when they name one.
    path: Option<&'static str>,
    credential: Option<&'static str>,
}

type Encode = Box<dyn Fn(&str, CompletionRequest) -> (Option<ModelSpec>, Outcome) + Send + Sync>;

/// One wire and dialect.
struct Encoder {
    name: String,
    route: Option<Route>,
    encode: Encode,
}

fn http(encoded: &Encoded) -> Sent {
    let request = &encoded.request;
    let headers = request
        .headers()
        .iter()
        .map(|(name, value)| {
            (
                name.as_str().to_owned(),
                String::from_utf8_lossy(value.as_bytes()).into_owned(),
                value.is_sensitive(),
            )
        })
        .collect();
    let body = match request.body() {
        Body::Bytes(bytes) => serde_json::from_slice(bytes).map_err(|error| error.to_string()),
        Body::Multipart(_) => Err("a multipart body".to_owned()),
    };
    Sent {
        uri: Some(request.uri().to_string()),
        headers,
        body,
    }
}

/// A payload sent by a client library rather than as an HTTP request.
#[cfg(any(feature = "bedrock", feature = "vertexai", feature = "gemini-grpc"))]
fn data(body: Result<Value, impl std::fmt::Display>) -> Sent {
    Sent {
        uri: None,
        headers: Vec::new(),
        body: body.map_err(|error| error.to_string()),
    }
}

/// `make`'s wire for a model, encoding a request as the driver prepares it.
fn encoder<W, F, S>(name: impl Into<String>, route: Option<Route>, make: F, show: S) -> Encoder
where
    W: Wire<Op = Completion> + 'static,
    F: Fn(&str) -> W + Send + Sync + 'static,
    S: Fn(&W::Payload) -> Sent + Send + Sync + 'static,
{
    Encoder {
        name: name.into(),
        route,
        encode: Box::new(move |model, request| {
            let wire = make(model);
            let described = wire.describe();
            let spec = described.spec().cloned();
            let outcome = Completion::prepare(request, &described)
                .and_then(|request| Ok(wire.encode(request, Mode::Unary)?))
                .map_or_else(
                    |error| Outcome::Refused(error.to_string()),
                    |payload| Outcome::Sent(show(&payload)),
                );
            (spec, outcome)
        }),
    }
}

/// The OpenAI dialects that serve the Responses API.
const RESPONSES: [&str; 6] = [
    "openai",
    "azure.openai",
    "openrouter",
    "xai",
    "copilot",
    "chatgpt",
];
/// The OpenAI dialects whose wire picks Chat or Responses per model.
const ROUTED: [&str; 4] = ["openai", "copilot", "opencode", "opencode-go"];

/// Every completion wire and dialect this build has.
fn encoders() -> Vec<Encoder> {
    use rig::providers::anthropic::wire as anthropic;
    use rig::providers::openai::responses_api::wire::Responses;
    use rig::providers::openai::wire as openai;
    use rig::providers::{cohere, gemini, ollama};

    fn openai_route(dialect: &openai::Dialect, path: &'static str) -> Option<Route> {
        let config = openai::OpenAIConfig::with_key(dialect, KEY);
        Some(Route {
            base: config.base_url,
            path: Some(path),
            credential: Some(match config.auth {
                openai::Auth::Bearer | openai::Auth::OptionalBearer => "authorization",
                openai::Auth::ApiKeyHeader => "api-key",
            }),
        })
    }

    let mut chat: Vec<&'static openai::Dialect> = openai::all().collect();
    chat.extend([
        &openai::ZAI_CODING,
        &openai::MINIMAX_CHINA,
        &openai::MOONSHOT_CHINA,
    ]);
    let mut out = Vec::new();
    for dialect in chat {
        out.push(encoder(
            format!("chat|{}", dialect.name),
            openai_route(dialect, dialect.quirks.completion_path),
            move |model| openai::Chat::new(openai::OpenAIConfig::with_key(dialect, KEY), model),
            http,
        ));
        if RESPONSES.contains(&dialect.name) {
            out.push(encoder(
                format!("responses|{}", dialect.name),
                openai_route(dialect, dialect.quirks.responses.path),
                move |model| Responses::new(openai::OpenAIConfig::with_key(dialect, KEY), model),
                http,
            ));
        }
        // The routing wire picks Chat or Responses per model, so only the
        // base and the credential are the dialect's.
        if ROUTED.contains(&dialect.name) {
            out.push(encoder(
                format!("route|{}", dialect.name),
                openai_route(dialect, "").map(|route| Route {
                    path: None,
                    ..route
                }),
                move |model| {
                    openai::OpenAiWire::new(openai::OpenAIConfig::with_key(dialect, KEY), model)
                },
                http,
            ));
        }
    }
    for dialect in anthropic::all() {
        let config = anthropic::AnthropicConfig::with_key(dialect, KEY);
        let credential = match dialect.quirks.auth {
            anthropic::Auth::ApiKey => "x-api-key",
            anthropic::Auth::OptionalBearer => "authorization",
        };
        out.push(encoder(
            format!("messages|{}", dialect.name),
            Some(Route {
                base: config.base_url.clone(),
                path: Some("/v1/messages"),
                credential: Some(credential),
            }),
            move |model| config.completion(model),
            http,
        ));
    }
    let gemini_route = Some(Route {
        base: gemini::GeminiConfig::new(KEY).base_url,
        path: None,
        credential: Some("x-goog-api-key"),
    });
    out.push(encoder(
        "gemini|generate",
        gemini_route.clone(),
        |model| gemini::completion::GenerateContent::new(gemini::GeminiConfig::new(KEY), model),
        http,
    ));
    out.push(encoder(
        "gemini|interactions",
        gemini_route,
        |model| gemini::interactions_api::Interactions::new(gemini::GeminiConfig::new(KEY), model),
        http,
    ));
    let cohere_route = Some(Route {
        base: cohere::wire::CohereConfig::new(KEY).base_url,
        path: None,
        credential: Some("authorization"),
    });
    out.push(encoder(
        "cohere|chat",
        cohere_route.clone(),
        |model| cohere::wire::CohereConfig::new(KEY).completion(model),
        http,
    ));
    out.push(encoder(
        "cohere|native",
        cohere_route,
        |model| cohere::chat::NativeChat::new(cohere::wire::CohereConfig::new(KEY), model),
        http,
    ));
    out.push(encoder(
        "ollama|chat",
        Some(Route {
            base: ollama::OllamaConfig::new().base_url,
            path: None,
            credential: Some("authorization"),
        }),
        |model| ollama::Chat::new(ollama::OllamaConfig::new().with_api_key(KEY), model),
        http,
    ));
    companions(&mut out);
    out
}

/// The companion crates' wires this build enables. They send no HTTP
/// request of their own, so only their bodies are checked.
#[cfg_attr(
    not(any(feature = "bedrock", feature = "vertexai", feature = "gemini-grpc")),
    allow(unused_variables)
)]
fn companions(out: &mut Vec<Encoder>) {
    #[cfg(feature = "bedrock")]
    out.push(encoder(
        "bedrock|converse",
        None,
        |model: &str| rig::bedrock::completion::Converse::new(model),
        |payload: &rig::bedrock::completion::ConverseRequest| {
            data(
                serde_json::to_value(&payload.body)
                    .map(|body| json!({ "model": payload.model, "body": body })),
            )
        },
    ));
    #[cfg(feature = "vertexai")]
    out.push(encoder(
        "vertex|generate",
        None,
        |model: &str| rig::vertexai::completion::GenerateContent::new(model),
        |payload| data(serde_json::to_value(payload)),
    ));
    #[cfg(feature = "gemini-grpc")]
    out.push(encoder(
        "grpc|generate",
        None,
        |model: &str| rig::gemini_grpc::completion::GenerateContent::new(model),
        |payload| data(rig::gemini_grpc::rest::to_rest(payload)),
    ));
}

/// The request shapes, by name.
fn shapes() -> anyhow::Result<Vec<(&'static str, CompletionRequest)>> {
    let tool = ToolDefinition {
        name: ToolName::new("lookup")?,
        description: "Look something up.".into(),
        parameters: json!({"type": "object", "properties": {"q": {"type": "string"}}}),
        annotations: Default::default(),
    };
    let image = Image {
        data: DocumentSourceKind::base64("iVBORw0KGgo="),
        media_type: Some(ImageMediaType::PNG),
        ..Image::default()
    };
    let base = || CompletionRequest::new("hi");
    Ok(vec![
        ("plain", base()),
        // A request that sets no option is sent as built by design
        // (`crates/rig-core/TYPED_OPTIONS.md`), so this one sets the policy.
        (
            "temperature",
            base()
                .temperature(0.5)
                .options(GenerationOptions::default().on_unsupported(OnUnsupported::Error)),
        ),
        ("tools", base().tool(tool)),
        (
            "effort",
            base().options(GenerationOptions::default().reasoning(Effort::High)),
        ),
        (
            "image",
            CompletionRequest::new("next").messages(vec![
                Message::User {
                    content: vec![UserContent::text("look"), UserContent::Image(image)],
                },
                Message::assistant("ok"),
            ]),
        ),
    ])
}

/// Whether `value` holds an object key `key` anywhere inside.
fn has_key(value: &Value, key: &str) -> bool {
    match value {
        Value::Array(items) => items.iter().any(|item| has_key(item, key)),
        Value::Object(map) => map
            .iter()
            .any(|(name, item)| name == key || has_key(item, key)),
        _ => false,
    }
}

/// `url` split into its scheme and authority, and its path without the
/// query.
fn split_url(url: &str) -> Option<(&str, &str)> {
    let after_scheme = url.find("://")? + 3;
    let path_at = url
        .get(after_scheme..)?
        .find('/')
        .map_or(url.len(), |at| after_scheme + at);
    let path = url.get(path_at..)?;
    Some((
        url.get(..path_at)?,
        path.split('?').next().unwrap_or_default(),
    ))
}

/// The model a request addresses: its body's `model` field, else a whole
/// segment of the URL path, which may end in a `:method` suffix.
fn names_model(sent: &Sent, body: &Value, model: &str) -> bool {
    let needle = format!("/{model}");
    match body.get("model").and_then(Value::as_str) {
        Some(field) => field == model || field.ends_with(&needle),
        None => sent
            .uri
            .as_deref()
            .and_then(split_url)
            .is_some_and(|(_, path)| {
                path.match_indices(&needle).any(|(at, _)| {
                    path.get(at + needle.len()..)
                        .is_some_and(|rest| rest.is_empty() || rest.starts_with(['/', ':']))
                })
            }),
    }
}

/// Whether `url` has `base`'s scheme and authority exactly, and a path at or
/// under `base`'s path. An empty base (Azure has no default endpoint) leaves
/// a URL that is only a path.
fn under_base(url: &str, base: &str) -> bool {
    let base = base.trim_end_matches('/');
    if base.is_empty() {
        return url.starts_with('/');
    }
    match (split_url(url), split_url(base)) {
        (Some((origin, path)), Some((base_origin, base_path))) => {
            origin == base_origin
                && path
                    .strip_prefix(base_path)
                    .is_some_and(|rest| rest.is_empty() || rest.starts_with('/'))
        }
        _ => false,
    }
}

/// Why `shape`'s request should not reach `spec`'s model, if the catalog
/// says so. An unknown sampling or reasoning rule is no reason: the
/// property is skipped there.
fn unsupported(spec: &ModelSpec, shape: &str) -> Option<&'static str> {
    match shape {
        "temperature" if spec.sampling == Some(Sampling::Never) => Some("temperature"),
        "effort" => match &spec.reasoning {
            ReasoningSupport::None => Some("reasoning"),
            ReasoningSupport::Listed { levels, .. } if !levels.contains(&Effort::High) => {
                Some("effort high")
            }
            _ => None,
        },
        _ => None,
    }
}

/// Whether `body` sends what `unsupported` names: a `temperature` key, or a
/// body that differs from the same request without reasoning.
fn sends(body: &Value, shape: &str, plain: Option<&Value>) -> bool {
    match shape {
        "temperature" => has_key(body, "temperature"),
        // A refused plain request leaves nothing to compare, so any effort
        // body counts as sending the reasoning the catalog refuses.
        _ => plain != Some(body),
    }
}

/// The properties one sent request breaks, each naming the wire, model and
/// shape.
fn check(
    encoder: &Encoder,
    model: &str,
    shape: &str,
    spec: Option<&ModelSpec>,
    sent: &Sent,
    plain: Option<&Value>,
    failures: &mut Failures,
) {
    let at = format!("{} {model} {shape}", encoder.name);
    let body = match &sent.body {
        Ok(body) if body.is_object() => body,
        Ok(other) => {
            return failures
                .json
                .push(format!("{at}: body is not an object: {other}"));
        }
        Err(error) => {
            return failures
                .json
                .push(format!("{at}: body is not JSON: {error}"));
        }
    };
    if !names_model(sent, body, model) {
        let field = body.get("model");
        failures
            .model
            .push(format!("{at}: model field {field:?}, URL {:?}", sent.uri));
    }
    if let Some(what) = spec.and_then(|spec| unsupported(spec, shape))
        && sends(body, shape, plain)
    {
        failures.unsupported.push(format!(
            "{at}: the catalog says no {what}, the body sends it: {body}"
        ));
    }
    if let (Some(route), Some(uri)) = (&encoder.route, &sent.uri) {
        let carriers: Vec<&(String, String, bool)> = sent
            .headers
            .iter()
            .filter(|(_, value, _)| value.contains(KEY))
            .collect();
        let path_ok = route.path.is_none_or(|path| {
            uri.split('?')
                .next()
                .is_some_and(|before| before.ends_with(path))
        });
        let header_ok = matches!(
            carriers.as_slice(),
            [(name, _, true)] if Some(name.as_str()) == route.credential
        );
        if !under_base(uri, &route.base) || !path_ok || !header_ok || uri.contains(KEY) {
            failures.route.push(format!(
                "{at}: sent to {uri} with credential headers {carriers:?}; the settings say \
                 base {}, path {:?}, header {:?} marked sensitive",
                route.base, route.path, route.credential
            ));
        }
    }
}

#[derive(Default)]
struct Failures {
    json: Vec<String>,
    model: Vec<String>,
    unsupported: Vec<String>,
    route: Vec<String>,
}

fn report(failures: &[String]) -> String {
    let shown: Vec<&str> = failures.iter().take(20).map(String::as_str).collect();
    format!(
        "{} failures, the first {}:\n{}",
        failures.len(),
        shown.len(),
        shown.join("\n")
    )
}

#[test]
fn chat_wires_send_what_the_catalog_allows() -> anyhow::Result<()> {
    wires_send_what_the_catalog_allows(&["chat|"])
}

#[test]
fn responses_and_route_wires_send_what_the_catalog_allows() -> anyhow::Result<()> {
    wires_send_what_the_catalog_allows(&["responses|", "route|"])
}

#[test]
fn messages_wires_send_what_the_catalog_allows() -> anyhow::Result<()> {
    wires_send_what_the_catalog_allows(&["messages|"])
}

#[test]
fn gemini_wires_send_what_the_catalog_allows() -> anyhow::Result<()> {
    wires_send_what_the_catalog_allows(&["gemini|"])
}

#[test]
fn cohere_and_ollama_wires_send_what_the_catalog_allows() -> anyhow::Result<()> {
    wires_send_what_the_catalog_allows(&["cohere|", "ollama|"])
}

#[test]
fn companion_wires_send_what_the_catalog_allows() -> anyhow::Result<()> {
    wires_send_what_the_catalog_allows(&["bedrock|", "vertex|", "grpc|"])
}

/// Every check on the wires whose names start with one of `families`. The
/// families run as separate tests, so a parallel runner runs them at once.
fn wires_send_what_the_catalog_allows(families: &[&str]) -> anyhow::Result<()> {
    let mut models: Vec<String> = Catalog::builtin()
        .iter()
        .map(|spec| spec.id.clone())
        .collect();
    models.sort_unstable();
    models.dedup();
    let shapes = shapes()?;
    let encoders: Vec<Encoder> = encoders()
        .into_iter()
        .filter(|encoder| {
            families
                .iter()
                .any(|family| encoder.name.starts_with(family))
        })
        .collect();
    let mut failures = Failures::default();
    for encoder in &encoders {
        for model in &models {
            let mut plain = None;
            for (shape, request) in &shapes {
                let encoded = catch_unwind(AssertUnwindSafe(|| {
                    (encoder.encode)(model, request.clone())
                }));
                let at = format!("{} {model} {shape}", encoder.name);
                match encoded {
                    Err(_) => failures.json.push(format!("{at}: encoding panicked")),
                    Ok((_, Outcome::Refused(reason))) if reason.trim().is_empty() => {
                        failures.json.push(format!("{at}: refused with no reason"));
                    }
                    Ok((_, Outcome::Refused(_))) => {}
                    Ok((spec, Outcome::Sent(sent))) => {
                        check(
                            encoder,
                            model,
                            shape,
                            spec.as_ref(),
                            &sent,
                            plain.as_ref(),
                            &mut failures,
                        );
                        if *shape == "plain" {
                            plain = sent.body.ok();
                        }
                    }
                }
            }
        }
    }
    anyhow::ensure!(
        failures.json.is_empty(),
        "a request neither encodes to a JSON object nor is refused with a reason. {}",
        report(&failures.json)
    );
    anyhow::ensure!(
        failures.model.is_empty(),
        "a request does not address the model it was built for. {}",
        report(&failures.model)
    );
    anyhow::ensure!(
        failures.unsupported.is_empty(),
        "a request sends what the model's catalog entry says it does not take. {}",
        report(&failures.unsupported)
    );
    anyhow::ensure!(
        failures.route.is_empty(),
        "a request is not sent where its dialect's settings say. {}",
        report(&failures.route)
    );
    Ok(())
}
