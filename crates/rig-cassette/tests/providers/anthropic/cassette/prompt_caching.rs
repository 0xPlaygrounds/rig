//! Anthropic prompt caching cassette tests.

use futures::StreamExt;
use rig::completion::{
    AssistantContent, CacheRetention, CompletionResponse as RigCompletionResponse,
    GenerationOptions, ToolDefinition, Usage,
};
use rig::driver::Model;
use rig::message::ToolChoice;
use rig::providers::anthropic;
use rig::providers::anthropic::completion::CacheTtl;
use rig::providers::anthropic::wire::AnthropicConfig;
use rig::providers::anthropic::wire::Messages;
use rig::streaming::Item;
use rig::streaming::StreamEvent;
use rig_test_support::cassette_models::AnthropicModels;
use serde_json::json;

use super::super::support::with_anthropic_cassette;
use rig::completion::CompletionRequest;

const CACHE_PROBE_RESPONSE: &str = "cache probe ready";
const CACHE_PROBE_PROMPT: &str =
    "Do not call any tools. Reply with exactly these three words: cache probe ready";
const STREAMING_CACHE_PROBE_RESPONSE: &str = "stream cache probe ready";
const STREAMING_CACHE_PROBE_PROMPT: &str =
    "Do not call any tools. Reply with exactly these four words: stream cache probe ready";
const CACHE_PADDING_REPETITIONS: usize = 180;
const CACHE_PADDING_SENTENCE: &str = "\
This cache fixture paragraph is stable provider test padding about request routing, \
tool schemas, system instructions, and deterministic replay behavior.";

/// Which caching constructors the scenario enables.
#[derive(Clone, Copy, PartialEq)]
enum CachingMode {
    Automatic,
    Automatic1h,
    ManualAutomatic1h,
}

impl CachingMode {
    fn manual(self) -> bool {
        matches!(self, Self::ManualAutomatic1h)
    }

    fn top_level_1h(self) -> bool {
        matches!(self, Self::Automatic1h | Self::ManualAutomatic1h)
    }
}

/// A matrix configuration: the wire's placement knobs, and the cache
/// retention every request asks for.
struct Matrix {
    model: Model<Messages>,
    cache: CacheRetention,
}

impl Matrix {
    fn request(&self, request: CompletionRequest) -> CompletionRequest {
        request.options(GenerationOptions::default().cache(self.cache))
    }

    async fn call(
        &self,
        request: CompletionRequest,
    ) -> Result<RigCompletionResponse, rig::error::ProviderError> {
        self.model.call(self.request(request)).await
    }

    fn stream(
        &self,
        request: CompletionRequest,
    ) -> Result<rig::streaming::Streamed<rig::operation::Completion>, rig::error::ProviderError>
    {
        self.model.stream(self.request(request))
    }
}

fn matrix_model(
    client: &AnthropicModels,
    mode: CachingMode,
    prefix_ttl: Option<CacheTtl>,
) -> Matrix {
    let mut model = client.completion(anthropic::completion::CLAUDE_SONNET_4_6);
    if mode.manual() {
        model = rig::Model::new(model.wire.with_prompt_caching(), model.transport);
    }
    if let Some(ttl) = prefix_ttl {
        model = rig::Model::new(
            model.wire.with_static_prefix_cache_ttl(ttl),
            model.transport,
        );
    }
    let cache = match mode {
        CachingMode::Automatic => CacheRetention::Short,
        CachingMode::Automatic1h | CachingMode::ManualAutomatic1h => CacheRetention::Long,
    };
    Matrix { model, cache }
}

/// Which cache-write buckets this configuration's markers can legally touch.
/// A bucket no marker requests must stay zero on every recorded turn — the
/// structural invariant that survives warm re-recording (where writes are
/// zero because the turn reads instead).
fn expected_buckets(mode: CachingMode, prefix_ttl: Option<&CacheTtl>) -> (bool, bool) {
    let static_markers = mode.manual() || prefix_ttl.is_some();
    let static_1h = static_markers
        && (prefix_ttl == Some(&CacheTtl::OneHour)
            || (prefix_ttl.is_none() && mode.top_level_1h()));
    // Every remaining mode caches automatically, so no tail marker writes 5m.
    let top_level_5m = !mode.top_level_1h();
    let can_write_1h = static_1h || mode.top_level_1h();
    let can_write_5m = (static_markers && !static_1h) || top_level_5m;
    (can_write_5m, can_write_1h)
}

/// A counter of Anthropic's `usage`, zero when absent.
fn count(usage: &serde_json::Value, pointer: &str) -> u64 {
    usage
        .pointer(pointer)
        .and_then(serde_json::Value::as_u64)
        .unwrap_or_default()
}

fn assert_cache_creation_split(
    usage: &serde_json::Value,
    mode: CachingMode,
    prefix_ttl: Option<&CacheTtl>,
    context: &str,
) {
    let (can_write_5m, can_write_1h) = expected_buckets(mode, prefix_ttl);
    assert!(
        usage["cache_creation"].is_object(),
        "{context}: Anthropic should report the per-TTL cache_creation split: {usage}"
    );
    let five = count(usage, "/cache_creation/ephemeral_5m_input_tokens");
    let one = count(usage, "/cache_creation/ephemeral_1h_input_tokens");
    assert_eq!(
        five + one,
        count(usage, "/cache_creation_input_tokens"),
        "{context}: per-TTL buckets should sum to the aggregate: {usage}"
    );
    if !can_write_5m {
        assert_eq!(
            five, 0,
            "{context}: no marker requests a 5m write in this configuration: {usage}"
        );
    }
    if !can_write_1h {
        assert_eq!(
            one, 0,
            "{context}: no marker requests a 1h write in this configuration: {usage}"
        );
    }
}

async fn run_matrix_body(
    client: AnthropicModels,
    name: &'static str,
    mode: CachingMode,
    prefix_ttl: Option<CacheTtl>,
    with_tools: bool,
    streaming: bool,
) {
    let model = matrix_model(&client, mode, prefix_ttl.clone());
    let tools = with_tools.then(|| cache_probe_tools_for(name));
    let preamble = cache_probe_preamble_for(name);

    if streaming {
        let first = send_matrix_streaming_probe(&model, preamble.clone(), tools.clone()).await;
        assert_text_contains_cache_probe(&first.text, STREAMING_CACHE_PROBE_RESPONSE);
        assert_cache_created_or_read(&first.usage, "first streamed matrix request");

        let second = send_matrix_streaming_probe(&model, preamble, tools).await;
        assert_text_contains_cache_probe(&second.text, STREAMING_CACHE_PROBE_RESPONSE);
        assert!(
            second.usage.cached_input_tokens.is_some_and(|n| n > 0),
            "warm streamed matrix request should read cached tokens, got usage: {:?}",
            second.usage
        );
    } else {
        let first = send_matrix_raw_probe(&model, preamble.clone(), tools.clone()).await;
        assert_matrix_raw_response(&first, mode, prefix_ttl.as_ref(), "first matrix request");
        let first_usage = &first["usage"];
        assert!(
            count(first_usage, "/cache_creation_input_tokens") > 0
                || count(first_usage, "/cache_read_input_tokens") > 0,
            "first matrix request should create or read cache tokens, got usage: {first_usage}"
        );

        let second = send_matrix_raw_probe(&model, preamble, tools).await;
        assert_matrix_raw_response(&second, mode, prefix_ttl.as_ref(), "warm matrix request");
        assert!(
            count(&second["usage"], "/cache_read_input_tokens") > 0,
            "warm matrix request should read cached tokens, got usage: {}",
            second["usage"]
        );
    }
}

/// A client whose requests would fail: the client-side error tests below must
/// error before any HTTP happens, so a reachable endpoint would mask a
/// regression that starts sending requests.
fn unreachable_anthropic_client() -> AnthropicModels {
    AnthropicModels::new(
        AnthropicConfig::new("client-side-error-test-key").with_base_url("http://127.0.0.1:9"),
        rig_test_support::cassettes::local_http(),
    )
}

async fn send_matrix_raw_probe(
    model: &Matrix,
    preamble: String,
    tools: Option<Vec<ToolDefinition>>,
) -> serde_json::Value {
    let mut builder = CompletionRequest::new(CACHE_PROBE_PROMPT)
        .preamble(preamble)
        .temperature(0.0)
        .max_tokens(16);
    if let Some(tools) = tools {
        builder = builder.tools(tools).tool_choice(ToolChoice::None);
    }
    let response = model
        .call(builder)
        .await
        .expect("matrix Anthropic request should succeed");
    response.raw
}

fn assert_matrix_raw_response(
    response: &serde_json::Value,
    mode: CachingMode,
    prefix_ttl: Option<&CacheTtl>,
    context: &str,
) {
    let text: String = response["content"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|block| block["type"] == "text")
        .filter_map(|block| block["text"].as_str())
        .collect();
    assert_text_contains_cache_probe(&text, CACHE_PROBE_RESPONSE);
    assert_cache_creation_split(&response["usage"], mode, prefix_ttl, context);
}

async fn send_matrix_streaming_probe(
    model: &Matrix,
    preamble: String,
    tools: Option<Vec<ToolDefinition>>,
) -> StreamingCacheProbeResponse {
    let mut builder = CompletionRequest::new(STREAMING_CACHE_PROBE_PROMPT)
        .preamble(preamble)
        .temperature(0.0)
        .max_tokens(16);
    if let Some(tools) = tools {
        builder = builder.tools(tools).additional_params(json!({
            "tool_choice": { "type": "none" }
        }));
    }
    let mut stream = model
        .stream(builder)
        .expect("streaming matrix Anthropic request should start");
    let mut text = String::new();
    while let Some(item) = stream.next().await {
        if let Item::Event(StreamEvent::Text { text: delta, .. }) =
            item.expect("streaming matrix Anthropic item should succeed")
        {
            text.push_str(&delta)
        }
    }
    let response = stream
        .finish()
        .await
        .expect("matrix stream should yield final token usage");

    StreamingCacheProbeResponse {
        text,
        usage: response.usage,
    }
}

const PREFIX_UNSET: Option<CacheTtl> = None;
const PREFIX_5M: Option<CacheTtl> = Some(CacheTtl::FiveMinutes);
const PREFIX_1H: Option<CacheTtl> = Some(CacheTtl::OneHour);

#[tokio::test]
async fn automatic_prefix_unset_tools_streaming() {
    with_anthropic_cassette(
        "prompt_caching/matrix_automatic_prefix_unset_tools_streaming",
        |client| async move {
            run_matrix_body(
                client,
                "automatic_prefix_unset_tools_streaming",
                CachingMode::Automatic,
                PREFIX_UNSET,
                true,
                true,
            )
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn automatic_prefix_unset_no_tools_nonstreaming() {
    with_anthropic_cassette(
        "prompt_caching/matrix_automatic_prefix_unset_no_tools_nonstreaming",
        |client| async move {
            run_matrix_body(
                client,
                "automatic_prefix_unset_no_tools_nonstreaming",
                CachingMode::Automatic,
                PREFIX_UNSET,
                false,
                false,
            )
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn automatic_prefix_unset_no_tools_streaming() {
    with_anthropic_cassette(
        "prompt_caching/matrix_automatic_prefix_unset_no_tools_streaming",
        |client| async move {
            run_matrix_body(
                client,
                "automatic_prefix_unset_no_tools_streaming",
                CachingMode::Automatic,
                PREFIX_UNSET,
                false,
                true,
            )
            .await;
        },
    )
    .await;
}

/// Not a cassette test: the illegal inversion must fail before any request is
/// sent, so there is no HTTP interaction to record. The unreachable base URL
/// is the no-request proof — a request would fail with a connection error,
/// not the knob-naming validation error asserted here.
#[tokio::test]
async fn static_prefix_5m_with_automatic_1h_errors_client_side() {
    let client = unreachable_anthropic_client();
    let model = matrix_model(&client, CachingMode::Automatic1h, PREFIX_5M);
    let error = model
        .call(
            CompletionRequest::new(CACHE_PROBE_PROMPT)
                .preamble(cache_probe_preamble_for("illegal inversion"))
                .max_tokens(16),
        )
        .await
        .expect_err("5m static prefix under a 1h top-level TTL must fail client-side");
    let message = error.to_string();
    assert!(
        message.contains("with_static_prefix_cache_ttl")
            && message.contains("CacheRetention::Long"),
        "error should name the knob and the option, got: {message}"
    );
}

/// Not a cassette test: streaming surfaces the same client-side error with no
/// HTTP interaction to record (see above).
#[tokio::test]
async fn static_prefix_5m_with_manual_automatic_1h_errors_client_side_streaming() {
    let client = unreachable_anthropic_client();
    let model = matrix_model(&client, CachingMode::ManualAutomatic1h, PREFIX_5M);
    let error = model
        .stream(
            CompletionRequest::new(STREAMING_CACHE_PROBE_PROMPT)
                .preamble(cache_probe_preamble_for("illegal inversion streaming"))
                .max_tokens(16),
        )
        .err()
        .expect("5m static prefix under a 1h top-level TTL must fail client-side");
    let message = error.to_string();
    assert!(
        message.contains("with_static_prefix_cache_ttl"),
        "error should name the knob, got: {message}"
    );
}

/// Two explicit provider-tool markers plus the knob's system marker plus the
/// automatic top-level breakpoint lands exactly on Anthropic's 4-marker limit.
/// (The knob's tool marker is not spent: the final tool already carries an
/// explicit marker, which Rig preserves rather than doubling up.)
#[tokio::test]
async fn static_prefix_with_explicit_tool_marker_at_marker_limit() {
    with_anthropic_cassette(
        "prompt_caching/static_prefix_with_explicit_tool_marker_at_marker_limit",
        |client| async move {
            let model = matrix_model(&client, CachingMode::Automatic, PREFIX_1H);
            let response = model
                .call(
                    CompletionRequest::new(CACHE_PROBE_PROMPT)
                        .preamble(cache_probe_preamble_for("marker budget at the limit"))
                        .tools(cache_probe_tools_for("marker budget at the limit"))
                        .tool_choice(ToolChoice::None)
                        .additional_params(json!({
                            "tools": [{
                                "name": "provider_cache_probe_alpha",
                                "description": "Provider-specific cache probe tool.",
                                "input_schema": {"type": "object", "properties": {}},
                                "cache_control": {"type": "ephemeral", "ttl": "1h"}
                            }, {
                                "name": "provider_cache_probe_beta",
                                "description": "Second provider-specific cache probe tool.",
                                "input_schema": {"type": "object", "properties": {}},
                                "cache_control": {"type": "ephemeral", "ttl": "1h"}
                            }]
                        }))
                        .temperature(0.0)
                        .max_tokens(16),
                )
                .await
                .expect("request at the 4-marker limit should succeed");
            let text = response_text(&response);
            assert_text_contains_cache_probe(&text, CACHE_PROBE_RESPONSE);
            assert_cache_created_or_read(&response.usage, "marker-budget-limit request");
            // The typed view splits the recorded cache writes by lifetime.
            let extras = response
                .extras::<rig::providers::anthropic::extension::AnthropicExt>()
                .expect("an Anthropic reply")
                .expect("the extras read the recorded reply");
            assert_eq!(
                extras.cache_creation.map(|cache| (
                    cache.ephemeral_5m_input_tokens,
                    cache.ephemeral_1h_input_tokens
                )),
                Some((336, 9441))
            );
        },
    )
    .await;
}

/// Not a cassette test: one explicit marker over the budget fails client-side
/// with no HTTP interaction to record (see above).
#[tokio::test]
async fn static_prefix_with_excess_explicit_tool_markers_errors_client_side() {
    let client = unreachable_anthropic_client();
    let model = matrix_model(&client, CachingMode::Automatic, PREFIX_1H);
    let provider_tools: Vec<serde_json::Value> = (0..4)
        .map(|idx| {
            json!({
                "name": format!("provider_cache_probe_{idx}"),
                "description": "Provider-specific cache probe tool.",
                "input_schema": {"type": "object", "properties": {}},
                "cache_control": {"type": "ephemeral", "ttl": "1h"}
            })
        })
        .collect();
    let error = model
        .call(
            CompletionRequest::new(CACHE_PROBE_PROMPT)
                .preamble(cache_probe_preamble_for("marker budget over limit"))
                .additional_params(json!({ "tools": provider_tools }))
                .max_tokens(16),
        )
        .await
        .expect_err("explicit markers beyond the budget must fail client-side");
    assert!(
        error.to_string().contains("cache_control"),
        "error should describe the marker budget, got: {error}"
    );
}

struct StreamingCacheProbeResponse {
    text: String,
    usage: Usage,
}

fn assert_text_contains_cache_probe(text: &str, expected: &str) {
    assert!(
        text.to_ascii_lowercase()
            .contains(&expected.to_ascii_lowercase()),
        "response should contain the requested cache probe text {expected:?}, got: {text:?}"
    );
}

fn assert_cache_created_or_read(usage: &Usage, context: &str) {
    assert!(
        usage.cache_creation_input_tokens.is_some_and(|n| n > 0)
            || usage.cached_input_tokens.is_some_and(|n| n > 0),
        "{context} should create or read cache tokens, got usage: {usage:?}"
    );
}

fn response_text(response: &RigCompletionResponse) -> String {
    response
        .choice
        .iter()
        .filter_map(|content| match content {
            AssistantContent::Text(text) => Some(text.text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn cache_probe_preamble_for(label: &str) -> String {
    format!(
        "You are a deterministic cassette test assistant for {label}. {}\n{}",
        "Never call tools for the cache probe prompt; answer only with the requested phrase.",
        cache_padding(CACHE_PADDING_REPETITIONS)
    )
}

fn cache_probe_tools_for(label: &str) -> Vec<ToolDefinition> {
    vec![
        ToolDefinition {
            name: rig_core::message::ToolName::new("lookup_cache_policy").expect("tool name"),
            description: format!(
                "Return {label} internal prompt cache policy notes. {}",
                cache_padding(CACHE_PADDING_REPETITIONS / 2)
            ),
            parameters: json!({
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "Policy topic to look up."
                    }
                },
                "required": ["topic"]
            }),
        },
        ToolDefinition {
            name: rig_core::message::ToolName::new("lookup_cache_fixture").expect("tool name"),
            description: format!(
                "Return prompt cache fixture notes. {}",
                cache_padding(CACHE_PADDING_REPETITIONS / 2)
            ),
            parameters: json!({
                "type": "object",
                "properties": {
                    "fixture": {
                        "type": "string",
                        "description": "Fixture identifier to look up."
                    }
                },
                "required": ["fixture"]
            }),
        },
    ]
}

fn cache_padding(repetitions: usize) -> String {
    std::iter::repeat_n(CACHE_PADDING_SENTENCE, repetitions)
        .collect::<Vec<_>>()
        .join(" ")
}

// ---------------------------------------------------------------------------
// Cross-provider cache conformance
// ---------------------------------------------------------------------------
//
// The cells above are Anthropic's own cache matrix: manual/automatic/TTL knob
// combinations, marker budgets, per-TTL write buckets. What they do not do —
// what nothing in the tree did before the shared harness — is ask *how much* of
// the prefix was actually served from cache. Every one of them asserts
// `cached_input_tokens > 0`, which passes just as happily when 200 of 40,000
// prefix tokens are cached as when 39,800 are.
//
// These cells add that question, through the same provider-generic harness the
// other eleven providers use, so Anthropic's numbers are read the same way as
// everyone else's and the denominator is written down rather than assumed.
//
// Anthropic's wire counts cache tokens beside `input_tokens`; rig's `Usage`
// counts them inside it, as on every provider. Anthropic is the one provider
// in the matrix that needs explicit `cache_control` breakpoints, which the
// descriptor states.

use crate::cache_conformance::{CacheProbe, CacheSupport};

/// Rig's `input_tokens` for Anthropic includes cache reads and writes, so turn
/// 1's billed prompt is `input_tokens`.
pub(super) const ANTHROPIC_CACHE_SUPPORT: CacheSupport = CacheSupport {
    provider: "anthropic",
    explicit_breakpoints: true,
    reports_writes: true,
    // Anthropic's documented minimum is 1,024 tokens for Sonnet- and Opus-class
    // models (2,048 for Haiku-class). This suite runs on Sonnet.
    min_cacheable_tokens: 1024,
    cache_key_field: None,
    hit_ratio_floor: 0.80,
};

pub(super) fn conformance_probe() -> CacheProbe {
    CacheProbe::new("anthropic cache conformance")
}
