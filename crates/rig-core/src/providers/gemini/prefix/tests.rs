use std::time::Duration;

use serde_json::{Value, json};

use super::*;
use crate::completion::CompletionRequest;
use crate::providers::gemini::{GeminiConfig, GenerateContent};
use crate::test_utils::{MockHttpResponse, SequencedHttpClient};
use crate::wire::{Mode, Wire};

const PREAMBLE: &str = "You are a support agent.";

fn lookup() -> ToolDefinition {
    ToolDefinition {
        name: "lookup_order".to_owned(),
        description: "Status of an order by id.".to_owned(),
        parameters: json!({"type": "object", "properties": {"order_id": {"type": "string"}}}),
    }
}

fn search() -> api::HostedTool {
    api::HostedTool {
        google_search: Some(api::GoogleSearch::default()),
        ..Default::default()
    }
}

fn prefix(new: NewCachedContent) -> CachedPrefix {
    CachedPrefix {
        resource: api::CachedContent {
            name: Some("cachedContents/abc".to_owned()),
            ..Default::default()
        },
        request: new.render().expect("renders"),
    }
}

fn cache() -> NewCachedContent {
    NewCachedContent::new("gemini-3.8-flash")
        .system_instruction(PREAMBLE)
        .tools([lookup()])
        .hosted_tools([search()])
}

fn wire(prefix: CachedPrefix) -> GenerateContent {
    GenerateContent {
        cached_content: Some(prefix),
        ..GenerateContent::new(GeminiConfig::new("test-key"), "gemini-3.8-flash").with_settings(
            api::RequestSettings {
                tools: vec![search()],
                ..Default::default()
            },
        )
    }
}

fn request() -> CompletionRequest {
    CompletionRequest::new("where is A-1?")
        .preamble(PREAMBLE)
        .tool(lookup())
}

fn encode(wire: &GenerateContent, request: CompletionRequest) -> Result<Value, EncodeError> {
    let encoded = wire.encode(request, Mode::Unary)?;
    Ok(json_body(encoded.request.body()))
}

fn json_body(body: &crate::wire::Body) -> Value {
    let crate::wire::Body::Bytes(bytes) = body else {
        panic!("a JSON body");
    };
    serde_json::from_slice(bytes).expect("JSON")
}

#[test]
fn a_matching_request_reads_its_prefix_from_the_cache() {
    let body = encode(&wire(prefix(cache())), request()).expect("encodes");
    assert_eq!(body["cachedContent"], "cachedContents/abc");
    for field in ["systemInstruction", "tools", "toolConfig"] {
        assert!(body.get(field).is_none(), "{field} is the cache's: {body}");
    }
    assert_eq!(
        body["contents"],
        json!([{"role": "user", "parts": [{"text": "where is A-1?"}]}])
    );
}

#[test]
fn a_cache_with_both_tool_kinds_turns_on_server_side_invocations() {
    let rendered = cache().render().expect("renders");
    assert_eq!(
        rendered.tool_config,
        Some(api::ToolConfig {
            include_server_side_tool_invocations: Some(true),
            ..Default::default()
        })
    );
}

#[test]
fn each_mismatch_is_one_line_naming_what_conflicts() {
    let cells: [(&str, NewCachedContent, CompletionRequest); 3] = [
        (
            "system instruction",
            cache(),
            CompletionRequest::new("hi")
                .preamble("another")
                .tool(lookup()),
        ),
        (
            "tools",
            cache(),
            CompletionRequest::new("hi").preamble(PREAMBLE),
        ),
        (
            "tool config",
            cache().tool_config(api::ToolConfigSettings {
                retrieval_config: Some(api::RetrievalConfig {
                    language_code: Some("en".to_owned()),
                    ..Default::default()
                }),
                ..Default::default()
            }),
            request(),
        ),
    ];
    for (what, new, request) in cells {
        let error = encode(&wire(prefix(new)), request).expect_err("a conflict");
        let message = std::error::Error::source(&error)
            .map_or_else(|| error.to_string(), ToString::to_string);
        assert_eq!(
            message,
            format!("cached content `cachedContents/abc` conflicts with the request's {what}")
        );
        assert!(!message.contains('\n'));
    }
}

#[test]
fn a_tool_choice_the_cache_does_not_hold_conflicts() {
    let error = encode(
        &wire(prefix(cache())),
        request().tool_choice(crate::message::ToolChoice::Required),
    )
    .expect_err("the cache owns the tool config");
    assert!(error.to_string().contains("tool config"), "{error}");
}

#[test]
fn a_prefix_round_trips_through_a_checkpoint() {
    let prefix = prefix(cache().expiry(CacheExpiry::ttl(Duration::from_secs(3600))));
    let json = serde_json::to_string(&prefix).expect("serialize");
    let restored: CachedPrefix = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(restored, prefix);
    assert_eq!(restored.name(), "cachedContents/abc");
    assert_eq!(restored.request.ttl.as_deref(), Some("3600.000000000s"));
}

#[test]
fn the_model_is_qualified_once() {
    for model in ["gemini-3.8-flash", "models/gemini-3.8-flash"] {
        let rendered = NewCachedContent::new(model)
            .content("corpus")
            .render()
            .expect("renders");
        assert_eq!(rendered.model.as_deref(), Some("models/gemini-3.8-flash"));
    }
}

#[tokio::test]
async fn ensure_recreates_an_expired_cache_from_its_body() {
    let gone = r#"{"error":{"code":404,"message":"CachedContent not found"}}"#;
    let fresh = r#"{"name":"cachedContents/def","model":"models/gemini-3.8-flash"}"#;
    let http = SequencedHttpClient::new([
        MockHttpResponse::error(http::StatusCode::NOT_FOUND, gone),
        MockHttpResponse::success(fresh),
    ]);
    let caches = crate::driver::Model::new(
        GeminiConfig::new("test-key").cached_contents(),
        http.clone(),
    );
    let expired = prefix(cache().expiry(CacheExpiry::expire_time("2020-01-01T00:00:00Z")));

    let renewed = caches.ensure(&expired).await.expect("recreated");
    assert_eq!(renewed.name(), "cachedContents/def");
    assert_eq!(
        renewed.request.expire_time, None,
        "a past expiry is dropped"
    );
    assert_eq!(renewed.request.tools, expired.request.tools);
    let requests = http.requests();
    assert_eq!(requests.len(), 2, "a read, then a create");
    let created: api::CachedContent =
        serde_json::from_slice(&requests[1].body).expect("the create body");
    assert_eq!(
        created.system_instruction,
        expired.request.system_instruction
    );
}

#[tokio::test]
async fn ensure_keeps_a_live_cache() {
    let live = r#"{"name":"cachedContents/abc","model":"models/gemini-3.8-flash","expireTime":"2030-01-01T00:00:00Z"}"#;
    let http = SequencedHttpClient::new([MockHttpResponse::success(live)]);
    let caches = crate::driver::Model::new(
        GeminiConfig::new("test-key").cached_contents(),
        http.clone(),
    );
    let prefix = prefix(cache());
    let kept = caches.ensure(&prefix).await.expect("alive");
    assert_eq!(kept.name(), "cachedContents/abc");
    assert_eq!(kept.request, prefix.request);
    assert_eq!(http.requests().len(), 1);
}
