//! Anthropic prompt caching cassette tests.

use futures::StreamExt;
use rig::completion::{
    AssistantContent, CompletionModel, CompletionResponse as RigCompletionResponse, ToolDefinition,
    Usage,
};
use rig::message::ToolChoice;
use rig::prelude::*;
use rig::providers::anthropic;
use rig::providers::anthropic::completion::CacheTtl;
use rig::streaming::StreamedAssistantContent;
use rig::telemetry::ProviderResponseExt;
use serde_json::json;

use super::super::support::with_anthropic_cassette;

const CACHE_PROBE_RESPONSE: &str = "cache probe ready";
const CACHE_PROBE_PROMPT: &str =
    "Do not call any tools. Reply with exactly these three words: cache probe ready";
const STREAMING_CACHE_PROBE_RESPONSE: &str = "stream cache probe ready";
const STREAMING_CACHE_PROBE_PROMPT: &str =
    "Do not call any tools. Reply with exactly these four words: stream cache probe ready";
const AUTOMATIC_CACHE_PROBE_RESPONSE: &str = "automatic cache probe ready";
const AUTOMATIC_CACHE_PROBE_PROMPT: &str =
    "Do not call any tools. Reply with exactly these four words: automatic cache probe ready";
const CACHE_PADDING_REPETITIONS: usize = 180;
const CACHE_PADDING_SENTENCE: &str = "\
This cache fixture paragraph is stable provider test padding about request routing, \
tool schemas, system instructions, and deterministic replay behavior.";

#[tokio::test]
async fn manual_prompt_caching_reuses_tool_cache() {
    with_anthropic_cassette(
        "prompt_caching/manual_prompt_caching_reuses_tool_cache",
        |client| async move {
            let model = client
                .completion_model(anthropic::completion::CLAUDE_SONNET_4_6)
                .with_prompt_caching();
            let tools = cache_probe_tools();

            let first = send_cache_probe(
                model.clone(),
                CACHE_PROBE_PROMPT,
                cache_probe_preamble(),
                tools.clone(),
            )
            .await;
            assert_response_contains_cache_probe(&first, CACHE_PROBE_RESPONSE);
            assert_cache_created_or_read(&first.usage, "first prompt-cached request");

            let second =
                send_cache_probe(model, CACHE_PROBE_PROMPT, cache_probe_preamble(), tools).await;
            assert_response_contains_cache_probe(&second, CACHE_PROBE_RESPONSE);
            assert!(
                second.usage.cached_input_tokens > 0,
                "second prompt-cached request should read cached tokens, got usage: {:?}",
                second.usage
            );
        },
    )
    .await;
}

#[tokio::test]
async fn streaming_prompt_caching_reuses_tool_cache() {
    with_anthropic_cassette(
        "prompt_caching/streaming_prompt_caching_reuses_tool_cache",
        |client| async move {
            let model = client
                .completion_model(anthropic::completion::CLAUDE_SONNET_4_6)
                .with_prompt_caching();
            let tools = cache_probe_tools_for("streaming prompt caching");

            let first = send_streaming_cache_probe(
                model.clone(),
                STREAMING_CACHE_PROBE_PROMPT,
                cache_probe_preamble_for("streaming prompt caching"),
                tools.clone(),
            )
            .await;
            assert_text_contains_cache_probe(&first.text, STREAMING_CACHE_PROBE_RESPONSE);
            assert_cache_created_or_read(&first.usage, "first streaming prompt-cached request");

            let second = send_streaming_cache_probe(
                model,
                STREAMING_CACHE_PROBE_PROMPT,
                cache_probe_preamble_for("streaming prompt caching"),
                tools,
            )
            .await;
            assert_text_contains_cache_probe(&second.text, STREAMING_CACHE_PROBE_RESPONSE);
            assert!(
                second.usage.cached_input_tokens > 0,
                "second streaming prompt-cached request should read cached tokens, got usage: {:?}",
                second.usage
            );
        },
    )
    .await;
}

#[tokio::test]
async fn prompt_and_automatic_caching_reuses_tool_cache() {
    with_anthropic_cassette(
        "prompt_caching/prompt_and_automatic_caching_reuses_tool_cache",
        |client| async move {
            let model = client
                .completion_model(anthropic::completion::CLAUDE_SONNET_4_6)
                .with_prompt_caching()
                .with_automatic_caching();
            let tools = cache_probe_tools_for("manual plus automatic prompt caching");

            let first = send_cache_probe(
                model.clone(),
                AUTOMATIC_CACHE_PROBE_PROMPT,
                cache_probe_preamble_for("manual plus automatic prompt caching"),
                tools.clone(),
            )
            .await;
            assert_response_contains_cache_probe(&first, AUTOMATIC_CACHE_PROBE_RESPONSE);
            assert_cache_created_or_read(&first.usage, "first prompt+automatic cached request");

            let second = send_cache_probe(
                model,
                AUTOMATIC_CACHE_PROBE_PROMPT,
                cache_probe_preamble_for("manual plus automatic prompt caching"),
                tools,
            )
            .await;
            assert_response_contains_cache_probe(&second, AUTOMATIC_CACHE_PROBE_RESPONSE);
            assert!(
                second.usage.cached_input_tokens > 0,
                "second prompt+automatic cached request should read cached tokens, got usage: {:?}",
                second.usage
            );
        },
    )
    .await;
}

/// Which caching constructors the scenario enables.
#[derive(Clone, Copy, PartialEq)]
enum CachingMode {
    Manual,
    Automatic,
    Automatic1h,
    ManualAutomatic,
    ManualAutomatic1h,
}

impl CachingMode {
    fn manual(self) -> bool {
        matches!(
            self,
            Self::Manual | Self::ManualAutomatic | Self::ManualAutomatic1h
        )
    }

    fn automatic(self) -> bool {
        !matches!(self, Self::Manual)
    }

    fn top_level_1h(self) -> bool {
        matches!(self, Self::Automatic1h | Self::ManualAutomatic1h)
    }
}

fn matrix_model(
    client: &anthropic::Client,
    mode: CachingMode,
    prefix_ttl: Option<CacheTtl>,
) -> anthropic::completion::CompletionModel {
    let mut model = client.completion_model(anthropic::completion::CLAUDE_SONNET_4_6);
    if mode.manual() {
        model = model.with_prompt_caching();
    }
    model = match mode {
        CachingMode::Automatic | CachingMode::ManualAutomatic => model.with_automatic_caching(),
        CachingMode::Automatic1h | CachingMode::ManualAutomatic1h => {
            model.with_automatic_caching_1h()
        }
        CachingMode::Manual => model,
    };
    if let Some(ttl) = prefix_ttl {
        model = model.with_static_prefix_cache_ttl(ttl);
    }
    model
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
    let tail_marker_5m = mode.manual() && !mode.automatic();
    let top_level_5m = mode.automatic() && !mode.top_level_1h();
    let can_write_1h = static_1h || mode.top_level_1h();
    let can_write_5m = (static_markers && !static_1h) || tail_marker_5m || top_level_5m;
    (can_write_5m, can_write_1h)
}

fn assert_cache_creation_split(
    usage: &anthropic::completion::Usage,
    mode: CachingMode,
    prefix_ttl: Option<&CacheTtl>,
    context: &str,
) {
    let (can_write_5m, can_write_1h) = expected_buckets(mode, prefix_ttl);
    let Some(split) = usage.cache_creation.as_ref() else {
        panic!("{context}: Anthropic should report the per-TTL cache_creation split: {usage:?}");
    };
    assert_eq!(
        split.ephemeral_5m_input_tokens + split.ephemeral_1h_input_tokens,
        usage.cache_creation_input_tokens.unwrap_or_default(),
        "{context}: per-TTL buckets should sum to the aggregate: {usage:?}"
    );
    if !can_write_5m {
        assert_eq!(
            split.ephemeral_5m_input_tokens, 0,
            "{context}: no marker requests a 5m write in this configuration: {usage:?}"
        );
    }
    if !can_write_1h {
        assert_eq!(
            split.ephemeral_1h_input_tokens, 0,
            "{context}: no marker requests a 1h write in this configuration: {usage:?}"
        );
    }
}

async fn run_matrix_body(
    client: anthropic::Client,
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
            second.usage.cached_input_tokens > 0,
            "warm streamed matrix request should read cached tokens, got usage: {:?}",
            second.usage
        );
    } else {
        let first = send_matrix_raw_probe(&model, preamble.clone(), tools.clone()).await;
        assert_matrix_raw_response(&first, mode, prefix_ttl.as_ref(), "first matrix request");
        let first_usage = &first.usage;
        assert!(
            first_usage.cache_creation_input_tokens.unwrap_or_default() > 0
                || first_usage.cache_read_input_tokens.unwrap_or_default() > 0,
            "first matrix request should create or read cache tokens, got usage: {first_usage:?}"
        );

        let second = send_matrix_raw_probe(&model, preamble, tools).await;
        assert_matrix_raw_response(&second, mode, prefix_ttl.as_ref(), "warm matrix request");
        assert!(
            second.usage.cache_read_input_tokens.unwrap_or_default() > 0,
            "warm matrix request should read cached tokens, got usage: {:?}",
            second.usage
        );
    }
}

/// A client whose requests would fail: the client-side error tests below must
/// error before any HTTP happens, so a reachable endpoint would mask a
/// regression that starts sending requests.
fn unreachable_anthropic_client() -> anthropic::Client {
    anthropic::Client::builder()
        .api_key("client-side-error-test-key")
        .base_url("http://127.0.0.1:9")
        .build()
        .expect("client should build")
}

async fn send_matrix_raw_probe(
    model: &anthropic::completion::CompletionModel,
    preamble: String,
    tools: Option<Vec<ToolDefinition>>,
) -> anthropic::completion::CompletionResponse {
    let mut builder = model
        .completion_request(CACHE_PROBE_PROMPT)
        .preamble(preamble)
        .temperature(0.0)
        .max_tokens(16);
    if let Some(tools) = tools {
        builder = builder.tools(tools).tool_choice(ToolChoice::None);
    }
    model
        .raw_completion(builder.build())
        .await
        .expect("matrix Anthropic request should succeed")
}

fn assert_matrix_raw_response(
    response: &anthropic::completion::CompletionResponse,
    mode: CachingMode,
    prefix_ttl: Option<&CacheTtl>,
    context: &str,
) {
    let text = response.get_text_response().unwrap_or_default();
    assert_text_contains_cache_probe(&text, CACHE_PROBE_RESPONSE);
    assert_cache_creation_split(&response.usage, mode, prefix_ttl, context);
}

async fn send_matrix_streaming_probe(
    model: &anthropic::completion::CompletionModel,
    preamble: String,
    tools: Option<Vec<ToolDefinition>>,
) -> StreamingCacheProbeResponse {
    let mut builder = model
        .completion_request(STREAMING_CACHE_PROBE_PROMPT)
        .preamble(preamble)
        .temperature(0.0)
        .max_tokens(16);
    if let Some(tools) = tools {
        builder = builder.tools(tools).additional_params(json!({
            "tool_choice": { "type": "none" }
        }));
    }
    let mut stream = builder
        .stream()
        .await
        .expect("streaming matrix Anthropic request should start");
    let mut text = String::new();
    let mut usage = None;

    while let Some(item) = stream.next().await {
        match item.expect("streaming matrix Anthropic item should succeed") {
            StreamedAssistantContent::Text(delta) => text.push_str(&delta.text),
            StreamedAssistantContent::Final(response) => {
                usage = Some(response.usage);
            }
            _ => {}
        }
    }

    StreamingCacheProbeResponse {
        text,
        usage: usage.expect("matrix stream should yield final token usage"),
    }
}

const PREFIX_UNSET: Option<CacheTtl> = None;
const PREFIX_5M: Option<CacheTtl> = Some(CacheTtl::FiveMinutes);
const PREFIX_1H: Option<CacheTtl> = Some(CacheTtl::OneHour);

/// One test per knob combination. Cells deliberately absent from this list:
///
/// - `automatic_1h`/`manual_automatic_1h` × `prefix_5m` (8 cells): the illegal
///   TTL inversion; it fails client-side with no request sent, covered by the
///   `*_errors_client_side` tests below.
/// - `manual_automatic_1h` × `prefix_1h` (4 cells): serializes byte-identically
///   to `manual_automatic_1h` × prefix-unset (the prefix inherits the 1h
///   top-level TTL either way), so a recording would duplicate those fixtures.
/// - `manual` × prefix-unset × tools × both surfaces and `manual_automatic` ×
///   prefix-unset × tools × non-streaming (3 cells): covered by the
///   pre-existing cassettes in this suite, whose byte-identical replay is the
///   knob-unset compatibility proof.
#[tokio::test]
async fn manual_prefix_unset_no_tools_nonstreaming() {
    with_anthropic_cassette(
        "prompt_caching/matrix_manual_prefix_unset_no_tools_nonstreaming",
        |client| async move {
            run_matrix_body(
                client,
                "manual_prefix_unset_no_tools_nonstreaming",
                CachingMode::Manual,
                PREFIX_UNSET,
                false,
                false,
            )
            .await
        },
    )
    .await;
}

#[tokio::test]
async fn manual_prefix_unset_no_tools_streaming() {
    with_anthropic_cassette(
        "prompt_caching/matrix_manual_prefix_unset_no_tools_streaming",
        |client| async move {
            run_matrix_body(
                client,
                "manual_prefix_unset_no_tools_streaming",
                CachingMode::Manual,
                PREFIX_UNSET,
                false,
                true,
            )
            .await
        },
    )
    .await;
}

#[tokio::test]
async fn manual_prefix_5m_tools_nonstreaming() {
    with_anthropic_cassette(
        "prompt_caching/matrix_manual_prefix_5m_tools_nonstreaming",
        |client| async move {
            run_matrix_body(
                client,
                "manual_prefix_5m_tools_nonstreaming",
                CachingMode::Manual,
                PREFIX_5M,
                true,
                false,
            )
            .await
        },
    )
    .await;
}

#[tokio::test]
async fn manual_prefix_5m_tools_streaming() {
    with_anthropic_cassette(
        "prompt_caching/matrix_manual_prefix_5m_tools_streaming",
        |client| async move {
            run_matrix_body(
                client,
                "manual_prefix_5m_tools_streaming",
                CachingMode::Manual,
                PREFIX_5M,
                true,
                true,
            )
            .await
        },
    )
    .await;
}

#[tokio::test]
async fn manual_prefix_5m_no_tools_nonstreaming() {
    with_anthropic_cassette(
        "prompt_caching/matrix_manual_prefix_5m_no_tools_nonstreaming",
        |client| async move {
            run_matrix_body(
                client,
                "manual_prefix_5m_no_tools_nonstreaming",
                CachingMode::Manual,
                PREFIX_5M,
                false,
                false,
            )
            .await
        },
    )
    .await;
}

#[tokio::test]
async fn manual_prefix_5m_no_tools_streaming() {
    with_anthropic_cassette(
        "prompt_caching/matrix_manual_prefix_5m_no_tools_streaming",
        |client| async move {
            run_matrix_body(
                client,
                "manual_prefix_5m_no_tools_streaming",
                CachingMode::Manual,
                PREFIX_5M,
                false,
                true,
            )
            .await
        },
    )
    .await;
}

#[tokio::test]
async fn manual_prefix_1h_tools_nonstreaming() {
    with_anthropic_cassette(
        "prompt_caching/matrix_manual_prefix_1h_tools_nonstreaming",
        |client| async move {
            run_matrix_body(
                client,
                "manual_prefix_1h_tools_nonstreaming",
                CachingMode::Manual,
                PREFIX_1H,
                true,
                false,
            )
            .await
        },
    )
    .await;
}

#[tokio::test]
async fn manual_prefix_1h_tools_streaming() {
    with_anthropic_cassette(
        "prompt_caching/matrix_manual_prefix_1h_tools_streaming",
        |client| async move {
            run_matrix_body(
                client,
                "manual_prefix_1h_tools_streaming",
                CachingMode::Manual,
                PREFIX_1H,
                true,
                true,
            )
            .await
        },
    )
    .await;
}

#[tokio::test]
async fn manual_prefix_1h_no_tools_nonstreaming() {
    with_anthropic_cassette(
        "prompt_caching/matrix_manual_prefix_1h_no_tools_nonstreaming",
        |client| async move {
            run_matrix_body(
                client,
                "manual_prefix_1h_no_tools_nonstreaming",
                CachingMode::Manual,
                PREFIX_1H,
                false,
                false,
            )
            .await
        },
    )
    .await;
}

#[tokio::test]
async fn manual_prefix_1h_no_tools_streaming() {
    with_anthropic_cassette(
        "prompt_caching/matrix_manual_prefix_1h_no_tools_streaming",
        |client| async move {
            run_matrix_body(
                client,
                "manual_prefix_1h_no_tools_streaming",
                CachingMode::Manual,
                PREFIX_1H,
                false,
                true,
            )
            .await
        },
    )
    .await;
}

#[tokio::test]
async fn automatic_prefix_unset_tools_nonstreaming() {
    with_anthropic_cassette(
        "prompt_caching/matrix_automatic_prefix_unset_tools_nonstreaming",
        |client| async move {
            run_matrix_body(
                client,
                "automatic_prefix_unset_tools_nonstreaming",
                CachingMode::Automatic,
                PREFIX_UNSET,
                true,
                false,
            )
            .await
        },
    )
    .await;
}

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
            .await
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
            .await
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
            .await
        },
    )
    .await;
}

#[tokio::test]
async fn automatic_prefix_5m_tools_nonstreaming() {
    with_anthropic_cassette(
        "prompt_caching/matrix_automatic_prefix_5m_tools_nonstreaming",
        |client| async move {
            run_matrix_body(
                client,
                "automatic_prefix_5m_tools_nonstreaming",
                CachingMode::Automatic,
                PREFIX_5M,
                true,
                false,
            )
            .await
        },
    )
    .await;
}

#[tokio::test]
async fn automatic_prefix_5m_tools_streaming() {
    with_anthropic_cassette(
        "prompt_caching/matrix_automatic_prefix_5m_tools_streaming",
        |client| async move {
            run_matrix_body(
                client,
                "automatic_prefix_5m_tools_streaming",
                CachingMode::Automatic,
                PREFIX_5M,
                true,
                true,
            )
            .await
        },
    )
    .await;
}

#[tokio::test]
async fn automatic_prefix_5m_no_tools_nonstreaming() {
    with_anthropic_cassette(
        "prompt_caching/matrix_automatic_prefix_5m_no_tools_nonstreaming",
        |client| async move {
            run_matrix_body(
                client,
                "automatic_prefix_5m_no_tools_nonstreaming",
                CachingMode::Automatic,
                PREFIX_5M,
                false,
                false,
            )
            .await
        },
    )
    .await;
}

#[tokio::test]
async fn automatic_prefix_5m_no_tools_streaming() {
    with_anthropic_cassette(
        "prompt_caching/matrix_automatic_prefix_5m_no_tools_streaming",
        |client| async move {
            run_matrix_body(
                client,
                "automatic_prefix_5m_no_tools_streaming",
                CachingMode::Automatic,
                PREFIX_5M,
                false,
                true,
            )
            .await
        },
    )
    .await;
}

#[tokio::test]
async fn automatic_prefix_1h_tools_nonstreaming() {
    with_anthropic_cassette(
        "prompt_caching/matrix_automatic_prefix_1h_tools_nonstreaming",
        |client| async move {
            run_matrix_body(
                client,
                "automatic_prefix_1h_tools_nonstreaming",
                CachingMode::Automatic,
                PREFIX_1H,
                true,
                false,
            )
            .await
        },
    )
    .await;
}

#[tokio::test]
async fn automatic_prefix_1h_tools_streaming() {
    with_anthropic_cassette(
        "prompt_caching/matrix_automatic_prefix_1h_tools_streaming",
        |client| async move {
            run_matrix_body(
                client,
                "automatic_prefix_1h_tools_streaming",
                CachingMode::Automatic,
                PREFIX_1H,
                true,
                true,
            )
            .await
        },
    )
    .await;
}

#[tokio::test]
async fn automatic_prefix_1h_no_tools_nonstreaming() {
    with_anthropic_cassette(
        "prompt_caching/matrix_automatic_prefix_1h_no_tools_nonstreaming",
        |client| async move {
            run_matrix_body(
                client,
                "automatic_prefix_1h_no_tools_nonstreaming",
                CachingMode::Automatic,
                PREFIX_1H,
                false,
                false,
            )
            .await
        },
    )
    .await;
}

#[tokio::test]
async fn automatic_prefix_1h_no_tools_streaming() {
    with_anthropic_cassette(
        "prompt_caching/matrix_automatic_prefix_1h_no_tools_streaming",
        |client| async move {
            run_matrix_body(
                client,
                "automatic_prefix_1h_no_tools_streaming",
                CachingMode::Automatic,
                PREFIX_1H,
                false,
                true,
            )
            .await
        },
    )
    .await;
}

#[tokio::test]
async fn automatic_1h_prefix_unset_tools_nonstreaming() {
    with_anthropic_cassette(
        "prompt_caching/matrix_automatic_1h_prefix_unset_tools_nonstreaming",
        |client| async move {
            run_matrix_body(
                client,
                "automatic_1h_prefix_unset_tools_nonstreaming",
                CachingMode::Automatic1h,
                PREFIX_UNSET,
                true,
                false,
            )
            .await
        },
    )
    .await;
}

#[tokio::test]
async fn automatic_1h_prefix_unset_tools_streaming() {
    with_anthropic_cassette(
        "prompt_caching/matrix_automatic_1h_prefix_unset_tools_streaming",
        |client| async move {
            run_matrix_body(
                client,
                "automatic_1h_prefix_unset_tools_streaming",
                CachingMode::Automatic1h,
                PREFIX_UNSET,
                true,
                true,
            )
            .await
        },
    )
    .await;
}

#[tokio::test]
async fn automatic_1h_prefix_unset_no_tools_nonstreaming() {
    with_anthropic_cassette(
        "prompt_caching/matrix_automatic_1h_prefix_unset_no_tools_nonstreaming",
        |client| async move {
            run_matrix_body(
                client,
                "automatic_1h_prefix_unset_no_tools_nonstreaming",
                CachingMode::Automatic1h,
                PREFIX_UNSET,
                false,
                false,
            )
            .await
        },
    )
    .await;
}

#[tokio::test]
async fn automatic_1h_prefix_unset_no_tools_streaming() {
    with_anthropic_cassette(
        "prompt_caching/matrix_automatic_1h_prefix_unset_no_tools_streaming",
        |client| async move {
            run_matrix_body(
                client,
                "automatic_1h_prefix_unset_no_tools_streaming",
                CachingMode::Automatic1h,
                PREFIX_UNSET,
                false,
                true,
            )
            .await
        },
    )
    .await;
}

#[tokio::test]
async fn automatic_1h_prefix_1h_tools_nonstreaming() {
    with_anthropic_cassette(
        "prompt_caching/matrix_automatic_1h_prefix_1h_tools_nonstreaming",
        |client| async move {
            run_matrix_body(
                client,
                "automatic_1h_prefix_1h_tools_nonstreaming",
                CachingMode::Automatic1h,
                PREFIX_1H,
                true,
                false,
            )
            .await
        },
    )
    .await;
}

#[tokio::test]
async fn automatic_1h_prefix_1h_tools_streaming() {
    with_anthropic_cassette(
        "prompt_caching/matrix_automatic_1h_prefix_1h_tools_streaming",
        |client| async move {
            run_matrix_body(
                client,
                "automatic_1h_prefix_1h_tools_streaming",
                CachingMode::Automatic1h,
                PREFIX_1H,
                true,
                true,
            )
            .await
        },
    )
    .await;
}

#[tokio::test]
async fn automatic_1h_prefix_1h_no_tools_nonstreaming() {
    with_anthropic_cassette(
        "prompt_caching/matrix_automatic_1h_prefix_1h_no_tools_nonstreaming",
        |client| async move {
            run_matrix_body(
                client,
                "automatic_1h_prefix_1h_no_tools_nonstreaming",
                CachingMode::Automatic1h,
                PREFIX_1H,
                false,
                false,
            )
            .await
        },
    )
    .await;
}

#[tokio::test]
async fn automatic_1h_prefix_1h_no_tools_streaming() {
    with_anthropic_cassette(
        "prompt_caching/matrix_automatic_1h_prefix_1h_no_tools_streaming",
        |client| async move {
            run_matrix_body(
                client,
                "automatic_1h_prefix_1h_no_tools_streaming",
                CachingMode::Automatic1h,
                PREFIX_1H,
                false,
                true,
            )
            .await
        },
    )
    .await;
}

#[tokio::test]
async fn manual_automatic_prefix_unset_tools_streaming() {
    with_anthropic_cassette(
        "prompt_caching/matrix_manual_automatic_prefix_unset_tools_streaming",
        |client| async move {
            run_matrix_body(
                client,
                "manual_automatic_prefix_unset_tools_streaming",
                CachingMode::ManualAutomatic,
                PREFIX_UNSET,
                true,
                true,
            )
            .await
        },
    )
    .await;
}

#[tokio::test]
async fn manual_automatic_prefix_unset_no_tools_nonstreaming() {
    with_anthropic_cassette(
        "prompt_caching/matrix_manual_automatic_prefix_unset_no_tools_nonstreaming",
        |client| async move {
            run_matrix_body(
                client,
                "manual_automatic_prefix_unset_no_tools_nonstreaming",
                CachingMode::ManualAutomatic,
                PREFIX_UNSET,
                false,
                false,
            )
            .await
        },
    )
    .await;
}

#[tokio::test]
async fn manual_automatic_prefix_unset_no_tools_streaming() {
    with_anthropic_cassette(
        "prompt_caching/matrix_manual_automatic_prefix_unset_no_tools_streaming",
        |client| async move {
            run_matrix_body(
                client,
                "manual_automatic_prefix_unset_no_tools_streaming",
                CachingMode::ManualAutomatic,
                PREFIX_UNSET,
                false,
                true,
            )
            .await
        },
    )
    .await;
}

#[tokio::test]
async fn manual_automatic_prefix_5m_tools_nonstreaming() {
    with_anthropic_cassette(
        "prompt_caching/matrix_manual_automatic_prefix_5m_tools_nonstreaming",
        |client| async move {
            run_matrix_body(
                client,
                "manual_automatic_prefix_5m_tools_nonstreaming",
                CachingMode::ManualAutomatic,
                PREFIX_5M,
                true,
                false,
            )
            .await
        },
    )
    .await;
}

#[tokio::test]
async fn manual_automatic_prefix_5m_tools_streaming() {
    with_anthropic_cassette(
        "prompt_caching/matrix_manual_automatic_prefix_5m_tools_streaming",
        |client| async move {
            run_matrix_body(
                client,
                "manual_automatic_prefix_5m_tools_streaming",
                CachingMode::ManualAutomatic,
                PREFIX_5M,
                true,
                true,
            )
            .await
        },
    )
    .await;
}

#[tokio::test]
async fn manual_automatic_prefix_5m_no_tools_nonstreaming() {
    with_anthropic_cassette(
        "prompt_caching/matrix_manual_automatic_prefix_5m_no_tools_nonstreaming",
        |client| async move {
            run_matrix_body(
                client,
                "manual_automatic_prefix_5m_no_tools_nonstreaming",
                CachingMode::ManualAutomatic,
                PREFIX_5M,
                false,
                false,
            )
            .await
        },
    )
    .await;
}

#[tokio::test]
async fn manual_automatic_prefix_5m_no_tools_streaming() {
    with_anthropic_cassette(
        "prompt_caching/matrix_manual_automatic_prefix_5m_no_tools_streaming",
        |client| async move {
            run_matrix_body(
                client,
                "manual_automatic_prefix_5m_no_tools_streaming",
                CachingMode::ManualAutomatic,
                PREFIX_5M,
                false,
                true,
            )
            .await
        },
    )
    .await;
}

#[tokio::test]
async fn manual_automatic_prefix_1h_tools_nonstreaming() {
    with_anthropic_cassette(
        "prompt_caching/matrix_manual_automatic_prefix_1h_tools_nonstreaming",
        |client| async move {
            run_matrix_body(
                client,
                "manual_automatic_prefix_1h_tools_nonstreaming",
                CachingMode::ManualAutomatic,
                PREFIX_1H,
                true,
                false,
            )
            .await
        },
    )
    .await;
}

#[tokio::test]
async fn manual_automatic_prefix_1h_tools_streaming() {
    with_anthropic_cassette(
        "prompt_caching/matrix_manual_automatic_prefix_1h_tools_streaming",
        |client| async move {
            run_matrix_body(
                client,
                "manual_automatic_prefix_1h_tools_streaming",
                CachingMode::ManualAutomatic,
                PREFIX_1H,
                true,
                true,
            )
            .await
        },
    )
    .await;
}

#[tokio::test]
async fn manual_automatic_prefix_1h_no_tools_nonstreaming() {
    with_anthropic_cassette(
        "prompt_caching/matrix_manual_automatic_prefix_1h_no_tools_nonstreaming",
        |client| async move {
            run_matrix_body(
                client,
                "manual_automatic_prefix_1h_no_tools_nonstreaming",
                CachingMode::ManualAutomatic,
                PREFIX_1H,
                false,
                false,
            )
            .await
        },
    )
    .await;
}

#[tokio::test]
async fn manual_automatic_prefix_1h_no_tools_streaming() {
    with_anthropic_cassette(
        "prompt_caching/matrix_manual_automatic_prefix_1h_no_tools_streaming",
        |client| async move {
            run_matrix_body(
                client,
                "manual_automatic_prefix_1h_no_tools_streaming",
                CachingMode::ManualAutomatic,
                PREFIX_1H,
                false,
                true,
            )
            .await
        },
    )
    .await;
}

#[tokio::test]
async fn manual_automatic_1h_prefix_unset_tools_nonstreaming() {
    with_anthropic_cassette(
        "prompt_caching/matrix_manual_automatic_1h_prefix_unset_tools_nonstreaming",
        |client| async move {
            run_matrix_body(
                client,
                "manual_automatic_1h_prefix_unset_tools_nonstreaming",
                CachingMode::ManualAutomatic1h,
                PREFIX_UNSET,
                true,
                false,
            )
            .await
        },
    )
    .await;
}

#[tokio::test]
async fn manual_automatic_1h_prefix_unset_tools_streaming() {
    with_anthropic_cassette(
        "prompt_caching/matrix_manual_automatic_1h_prefix_unset_tools_streaming",
        |client| async move {
            run_matrix_body(
                client,
                "manual_automatic_1h_prefix_unset_tools_streaming",
                CachingMode::ManualAutomatic1h,
                PREFIX_UNSET,
                true,
                true,
            )
            .await
        },
    )
    .await;
}

#[tokio::test]
async fn manual_automatic_1h_prefix_unset_no_tools_nonstreaming() {
    with_anthropic_cassette(
        "prompt_caching/matrix_manual_automatic_1h_prefix_unset_no_tools_nonstreaming",
        |client| async move {
            run_matrix_body(
                client,
                "manual_automatic_1h_prefix_unset_no_tools_nonstreaming",
                CachingMode::ManualAutomatic1h,
                PREFIX_UNSET,
                false,
                false,
            )
            .await
        },
    )
    .await;
}

#[tokio::test]
async fn manual_automatic_1h_prefix_unset_no_tools_streaming() {
    with_anthropic_cassette(
        "prompt_caching/matrix_manual_automatic_1h_prefix_unset_no_tools_streaming",
        |client| async move {
            run_matrix_body(
                client,
                "manual_automatic_1h_prefix_unset_no_tools_streaming",
                CachingMode::ManualAutomatic1h,
                PREFIX_UNSET,
                false,
                true,
            )
            .await
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
        .completion_request(CACHE_PROBE_PROMPT)
        .preamble(cache_probe_preamble_for("illegal inversion"))
        .max_tokens(16)
        .send()
        .await
        .expect_err("5m static prefix under a 1h top-level TTL must fail client-side");
    let message = error.to_string();
    assert!(
        message.contains("with_static_prefix_cache_ttl")
            && message.contains("with_automatic_caching_1h"),
        "error should name both knobs, got: {message}"
    );
}

/// Not a cassette test: streaming surfaces the same client-side error with no
/// HTTP interaction to record (see above).
#[tokio::test]
async fn static_prefix_5m_with_manual_automatic_1h_errors_client_side_streaming() {
    let client = unreachable_anthropic_client();
    let model = matrix_model(&client, CachingMode::ManualAutomatic1h, PREFIX_5M);
    let error = model
        .completion_request(STREAMING_CACHE_PROBE_PROMPT)
        .preamble(cache_probe_preamble_for("illegal inversion streaming"))
        .max_tokens(16)
        .stream()
        .await
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
                .completion_request(CACHE_PROBE_PROMPT)
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
                .max_tokens(16)
                .send()
                .await
                .expect("request at the 4-marker limit should succeed");
            let text = response_text(&response);
            assert_text_contains_cache_probe(&text, CACHE_PROBE_RESPONSE);
            assert_cache_created_or_read(&response.usage, "marker-budget-limit request");
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
        .completion_request(CACHE_PROBE_PROMPT)
        .preamble(cache_probe_preamble_for("marker budget over limit"))
        .additional_params(json!({ "tools": provider_tools }))
        .max_tokens(16)
        .send()
        .await
        .expect_err("explicit markers beyond the budget must fail client-side");
    assert!(
        error.to_string().contains("cache_control"),
        "error should describe the marker budget, got: {error}"
    );
}

async fn send_cache_probe(
    model: anthropic::completion::CompletionModel,
    prompt: &'static str,
    preamble: String,
    tools: Vec<ToolDefinition>,
) -> RigCompletionResponse {
    model
        .completion_request(prompt)
        .preamble(preamble)
        .tools(tools)
        .tool_choice(ToolChoice::None)
        .temperature(0.0)
        .max_tokens(16)
        .send()
        .await
        .expect("prompt-cached Anthropic request should succeed")
}

struct StreamingCacheProbeResponse {
    text: String,
    usage: Usage,
}

async fn send_streaming_cache_probe(
    model: anthropic::completion::CompletionModel,
    prompt: &'static str,
    preamble: String,
    tools: Vec<ToolDefinition>,
) -> StreamingCacheProbeResponse {
    let mut stream = model
        .completion_request(prompt)
        .preamble(preamble)
        .tools(tools)
        .additional_params(json!({
            "tool_choice": { "type": "none" }
        }))
        .temperature(0.0)
        .max_tokens(16)
        .stream()
        .await
        .expect("streaming prompt-cached Anthropic request should start");
    let mut text = String::new();
    let mut usage = None;

    while let Some(item) = stream.next().await {
        match item.expect("streaming prompt-cached Anthropic item should succeed") {
            StreamedAssistantContent::Text(delta) => text.push_str(&delta.text),
            StreamedAssistantContent::Final(response) => {
                usage = Some(response.usage);
            }
            _ => {}
        }
    }

    StreamingCacheProbeResponse {
        text,
        usage: usage.expect("stream should yield final token usage"),
    }
}

fn assert_response_contains_cache_probe(response: &RigCompletionResponse, expected: &str) {
    let text = response_text(response);
    assert_text_contains_cache_probe(&text, expected);
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
        usage.cache_creation_input_tokens > 0 || usage.cached_input_tokens > 0,
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

fn cache_probe_preamble() -> String {
    format!(
        "You are a deterministic cassette test assistant. {}\n{}",
        "Never call tools for the cache probe prompt; answer only with the requested phrase.",
        cache_padding(CACHE_PADDING_REPETITIONS)
    )
}

fn cache_probe_preamble_for(label: &str) -> String {
    format!(
        "You are a deterministic cassette test assistant for {label}. {}\n{}",
        "Never call tools for the cache probe prompt; answer only with the requested phrase.",
        cache_padding(CACHE_PADDING_REPETITIONS)
    )
}

fn cache_probe_tools() -> Vec<ToolDefinition> {
    vec![
        ToolDefinition {
            name: "lookup_cache_policy".to_string(),
            description: format!(
                "Return internal prompt cache policy notes. {}",
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
            name: "lookup_cache_fixture".to_string(),
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

fn cache_probe_tools_for(label: &str) -> Vec<ToolDefinition> {
    vec![
        ToolDefinition {
            name: "lookup_cache_policy".to_string(),
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
            name: "lookup_cache_fixture".to_string(),
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
