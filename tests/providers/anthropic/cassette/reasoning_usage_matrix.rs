//! Edge matrix for extended-thinking tokens reaching `Usage::reasoning_tokens`.
//!
//! # The bug
//!
//! Anthropic reports the tokens Claude spent thinking as a breakdown of
//! `output_tokens`:
//!
//! ```text
//! "usage":{"input_tokens":773,"output_tokens":1835,
//!          "output_tokens_details":{"thinking_tokens":1167}}
//! ```
//!
//! Neither `anthropic::completion::Usage` nor the streaming `PartialUsage`
//! modeled `output_tokens_details`, so serde dropped it and
//! `anthropic_usage_totals` left `completion::Usage::reasoning_tokens` at `0`
//! on every turn — blocking and streaming alike. That field's own rustdoc
//! names "Anthropic extended thinking" as something it counts, and Gemini
//! (`thoughts_token_count`) and DeepSeek
//! (`completion_tokens_details.reasoning_tokens`) both populate it, so a
//! caller comparing reasoning spend across providers saw Anthropic as free.
//!
//! The value is a *breakdown* of `output_tokens`, not a sibling of it, so it
//! must not enter `total_tokens`.
//!
//! # Matrix
//!
//! Every row is a recorded live cell unless marked otherwise. Each recorded
//! cell asserts the parsed `reasoning_tokens` **equals its own fixture's**
//! recorded `thinking_tokens` (read back after the wrapper returns, since
//! record mode writes the fixture on the way out), so a cell whose provider
//! turn stopped thinking fails instead of passing vacuously.
//!
//! | # | Cell | Dimension | expected | Status |
//! |---|------|-----------|----------|--------|
//! | 1 | `blocking_budget_thinking` | `thinking.enabled`, minimum budget | `> 0` | recorded |
//! | 2 | `blocking_adaptive_declines_to_think` | adaptive declines: bucket present, zero | `0` | recorded |
//! | 3 | `blocking_larger_budget` | budget well above the minimum | `> 0` | recorded |
//! | 4 | `blocking_with_tools` | tools advertised, thinking on | `> 0` | recorded |
//! | 5 | `blocking_tool_use_terminal` | terminal is `tool_use` | `> 0` | recorded |
//! | 6 | `blocking_with_preamble` | system prompt present | `> 0` | recorded |
//! | 7 | `blocking_with_prompt_caching` | cache breakpoint + thinking | `> 0` | recorded |
//! | 8 | `blocking_redacted_thinking` | `redacted_thinking` block | `> 0` | recorded |
//! | 9 | `blocking_max_tokens_truncation` | `stop_reason: max_tokens` | `> 0` | recorded |
//! | 10 | `blocking_opus_model` | opus 4.8, adaptive-only model | `> 0` | recorded |
//! | 11 | `blocking_long_reasoning` | large breakdown, far above noise | `> 0` | recorded |
//! | 12 | `blocking_thinking_disabled_control` | control: thinking off, bucket absent | `0` | recorded |
//! | 13 | `blocking_haiku_no_thinking_control` | control: second model, bucket absent | `0` | recorded |
//! | 14 | `streaming_budget_thinking` | streamed twin of #1 | `> 0` | recorded |
//! | 15 | `streaming_adaptive_declines_to_think` | streamed twin of #2 | `0` | recorded |
//! | 16 | `streaming_larger_budget` | streamed twin of #3 | `> 0` | recorded |
//! | 17 | `streaming_with_tools` | streamed twin of #4 | `> 0` | recorded |
//! | 18 | `streaming_tool_use_terminal` | streamed twin of #5 | `> 0` | recorded |
//! | 19 | `streaming_with_preamble` | streamed twin of #6 | `> 0` | recorded |
//! | 20 | `streaming_with_prompt_caching` | streamed twin of #7 | `> 0` | recorded |
//! | 21 | `streaming_redacted_thinking` | streamed twin of #8 | `> 0` | recorded |
//! | 22 | `streaming_max_tokens_truncation` | streamed twin of #9 | `> 0` | recorded |
//! | 23 | `streaming_opus_model` | streamed twin of #10 | `> 0` | recorded |
//! | 24 | `streaming_long_reasoning` | streamed twin of #11 | `> 0` | recorded |
//! | 25 | `streaming_thinking_disabled_control` | streamed twin of #12 | `0` | recorded |
//! | 26 | `normalized_stream_budget_thinking` | adjacent: normalized stream usage | `> 0` | recorded |
//! | 27 | `agent_blocking_thinking` | adjacent: `Agent` request builder | `> 0` | recorded |
//! | 28 | `unit_absent_details_reports_zero` | serde: no `output_tokens_details` | `0` | unit |
//! | 29 | `unit_unknown_detail_bucket_is_ignored` | serde: forward compatibility | `> 0` | unit |
//! | 30 | `unit_thinking_tokens_stay_out_of_the_total` | totals arithmetic | n/a | unit |
//! | 31 | `unit_streaming_and_blocking_share_the_mapping` | shared helper parity | n/a | unit |
//!
//! Cells 28–31 are unit tests because no live turn can vary what they assert:
//! whether serde tolerates an absent or unknown bucket, and whether the
//! breakdown is excluded from `total_tokens`, are properties of the types and
//! the shared totals helper, not of any provider response.
//!
//! Two constraints the live API imposed, discovered while recording:
//!
//! - **Adaptive thinking may decline to think at all.** Opus 4.7 answers even
//!   the multi-step prompt without thinking, reporting the bucket *present and
//!   zero* — a different wire shape from thinking-off, which omits the bucket
//!   entirely. Rather than force it, cells 2 and 15 pin that shape, and cells
//!   10 and 23 (Opus 4.8) cover adaptive thinking that does engage.
//! - **Opus 4.8 rejects `thinking.type.enabled`** (`"is not supported for
//!   this model. Use \"thinking.type.adaptive\" and \"output_config.effort\""`),
//!   so the second-model-family cells cover the adaptive path only.
//!
//! Two dimensions were considered and deliberately dropped, with reasons:
//!
//! - **A `message_start` carry-forward cell.** `cache_creation` needs one
//!   because Anthropic reports it on `message_start`; `output_tokens_details`
//!   arrives on the terminal `message_delta` instead, so there is nothing to
//!   carry and no live turn can produce the inverse split. The streamed cells
//!   are what pin it: each reads the breakdown off its own terminal frame, so
//!   a fallback that started reading `message_start` would have to invent the
//!   same number to stay green.
//! - **A gateway cell.** The Anthropic-compatible gateways do not implement
//!   extended thinking on the Messages endpoint, so a gateway recording would
//!   assert the absent-breakdown case that cells 12–13 already cover.
//!
//! Run cassette tests in replay mode by default, or set
//! `RIG_PROVIDER_TEST_MODE=record` to record against the real provider.

use std::sync::{Arc, Mutex};

use futures::StreamExt;
use rig::completion::{CompletionModel, CompletionRequest, ToolDefinition, Usage};
use rig::prelude::*;
use rig::providers::anthropic;
use rig::streaming::RawStreamingChoice;
use serde_json::json;

use super::super::support::{recorded_response_body, with_anthropic_reasoning_usage_cassette};

type AnthropicModel = anthropic::completion::CompletionModel;

/// Anthropic's documented test string that forces `redacted_thinking` blocks.
const REDACTED_THINKING_MAGIC_STRING: &str = "ANTHROPIC_MAGIC_STRING_TRIGGER_REDACTED_THINKING_46C9A13E193C177646C7398A98432ECCCE4C1253D5E2D82641AC0E52CC2876CB";

/// A prompt that costs a few thinking tokens without a long answer.
const THINKING_PROMPT: &str = "What is 17 * 23? Reply with just the number.";

/// A prompt whose *answer* is long enough to run into a `max_tokens` ceiling
/// set just above the thinking budget, so the turn is cut off after thinking
/// rather than before it. Truncating during the thinking block itself is not
/// usable here: Anthropic requires `max_tokens > thinking.budget_tokens`, and
/// a turn that never finishes thinking emits no breakdown to assert on.
const TRUNCATING_PROMPT: &str = "Think briefly, then list the integers from 1 to 2000, one per line, with no \
     other text.";

/// A prompt whose reasoning is long enough that the breakdown is unmistakably
/// a real measurement rather than a rounding artifact.
const LONG_REASONING_PROMPT: &str = "A farmer has 17 sheep. All but 9 run away. Then he buys twice as many as \
     he has left, and sells 5. Work through it step by step, then give the \
     final count on its own line.";

fn budget_thinking(budget: u64) -> serde_json::Value {
    json!({ "thinking": { "type": "enabled", "budget_tokens": budget } })
}

fn adaptive_thinking() -> serde_json::Value {
    json!({ "thinking": { "type": "adaptive" } })
}

fn multiply_tool() -> ToolDefinition {
    ToolDefinition {
        name: "multiply".to_string(),
        description: "Multiply two integers.".to_string(),
        parameters: json!({
            "type": "object",
            "properties": {
                "a": { "type": "integer" },
                "b": { "type": "integer" }
            },
            "required": ["a", "b"]
        }),
    }
}

/// Carries the usage a cell observed out of the cassette closure, so the
/// premise can be re-derived from the fixture *after* the wrapper returns —
/// in record mode the fixture is written on the way out, so an in-body read
/// would assert against the previous recording.
#[derive(Clone, Default)]
struct Observed(Arc<Mutex<Option<Usage>>>);

impl Observed {
    fn new() -> Self {
        Self::default()
    }

    fn record(&self, usage: Usage) {
        *self.0.lock().expect("observed usage lock") = Some(usage);
    }

    fn take(&self, scenario: &str) -> Usage {
        (*self.0.lock().expect("observed usage lock"))
            .unwrap_or_else(|| panic!("{scenario}: the cell body never recorded a usage"))
    }

    /// A thinking cell: the parsed `reasoning_tokens` must equal the fixture's
    /// own recorded breakdown, be non-zero, and stay inside `output_tokens`
    /// without inflating the total.
    fn assert_matches(&self, scenario: &str, recorded: Option<u64>) {
        let usage = self.take(scenario);
        let recorded = recorded.unwrap_or_else(|| {
            panic!("{scenario}: the recorded turn reports no output_tokens_details")
        });

        assert!(
            recorded > 0,
            "{scenario}: the recorded turn must actually have spent thinking tokens",
        );
        assert_eq!(
            usage.reasoning_tokens, recorded,
            "{scenario}: reasoning_tokens must equal the recorded thinking_tokens",
        );
        assert!(
            usage.reasoning_tokens <= usage.output_tokens,
            "{scenario}: thinking is a breakdown of output_tokens ({} > {})",
            usage.reasoning_tokens,
            usage.output_tokens,
        );
        assert_eq!(
            usage.total_tokens,
            usage.input_tokens
                + usage.cached_input_tokens
                + usage.cache_creation_input_tokens
                + usage.output_tokens,
            "{scenario}: reasoning tokens must not be added into the total",
        );
    }

    /// Control: thinking was never requested, so the provider omits the
    /// breakdown entirely and `reasoning_tokens` stays `0` — while the turn
    /// still produced output, proving the cell drove a real completion.
    ///
    /// Asserting *absent* rather than "absent or zero" is deliberate: the two
    /// zero shapes reach `reasoning_tokens` through different serde paths
    /// (`None` vs `Some(0)`), and this suite covers both separately.
    fn assert_breakdown_absent(&self, scenario: &str, recorded: Option<u64>) {
        let usage = self.take(scenario);
        assert_eq!(
            recorded, None,
            "{scenario}: a turn with thinking off reports no breakdown at all",
        );
        assert_eq!(
            usage.reasoning_tokens, 0,
            "{scenario}: an absent breakdown reports zero reasoning tokens",
        );
        assert!(
            usage.output_tokens > 0,
            "{scenario}: the control turn still produced output",
        );
    }

    /// Control: adaptive thinking was requested and the model chose not to
    /// think, so the breakdown is *present and zero* — the other zero shape,
    /// which reaches `reasoning_tokens` through `Some(0)` rather than `None`.
    fn assert_breakdown_present_and_zero(&self, scenario: &str, recorded: Option<u64>) {
        let usage = self.take(scenario);
        assert_eq!(
            recorded,
            Some(0),
            "{scenario}: adaptive thinking that declines still reports the bucket",
        );
        assert_eq!(
            usage.reasoning_tokens, 0,
            "{scenario}: a zero breakdown reports zero reasoning tokens",
        );
        assert!(
            usage.output_tokens > 0,
            "{scenario}: the turn still produced output",
        );
    }
}

/// The `thinking_tokens` a blocking cassette's recorded response body reports.
fn recorded_blocking_thinking_tokens(scenario: &str) -> Option<u64> {
    let body = recorded_response_body(scenario);
    match &body["usage"]["output_tokens_details"]["thinking_tokens"] {
        serde_json::Value::Null => None,
        value => Some(value.as_u64().unwrap_or_else(|| {
            panic!("{scenario}: thinking_tokens should be a number, got {value}")
        })),
    }
}

/// The `thinking_tokens` a streamed cassette's terminal `message_delta` frame
/// reports. Streaming fixtures record SSE frames, not a JSON body, so the
/// blocking reader cannot be reused.
fn recorded_streamed_thinking_tokens(scenario: &str) -> Option<u64> {
    let path = crate::cassettes::cassette_path("anthropic", scenario);
    let contents = std::fs::read_to_string(&path)
        .unwrap_or_else(|err| panic!("cassette {} should be readable: {err}", path.display()));

    // Exactly one terminal frame, asserted rather than assumed: a multi-turn
    // cell would otherwise silently compare turn 1's breakdown against turn
    // 2's parsed usage.
    let mut frames = contents
        .lines()
        .filter(|line| line.contains(r#""type":"message_delta""#));
    let frame = frames.next().unwrap_or_else(|| {
        panic!(
            "cassette {} should record a message_delta frame",
            path.display()
        )
    });
    assert!(
        frames.next().is_none(),
        "cassette {} records more than one message_delta; this reader assumes a \
         single-turn cell",
        path.display()
    );
    let (_, after) = frame.split_once(r#""thinking_tokens":"#)?;
    let digits: String = after.chars().take_while(char::is_ascii_digit).collect();
    Some(digits.parse().unwrap_or_else(|err| {
        panic!(
            "thinking_tokens in {} should be a number: {err}",
            path.display()
        )
    }))
}

/// The `stop_reason` a streamed cassette's terminal `message_delta` reports.
fn recorded_streamed_stop_reason(scenario: &str) -> Option<String> {
    let path = crate::cassettes::cassette_path("anthropic", scenario);
    let contents = std::fs::read_to_string(&path)
        .unwrap_or_else(|err| panic!("cassette {} should be readable: {err}", path.display()));
    let frame = contents
        .lines()
        .find(|line| line.contains(r#""type":"message_delta""#))?;
    let (_, after) = frame.split_once(r#""stop_reason":""#)?;
    after.split('"').next().map(str::to_string)
}

fn request(
    model: &AnthropicModel,
    prompt: &str,
    params: serde_json::Value,
    max_tokens: u64,
) -> CompletionRequest {
    model
        .completion_request(prompt)
        .max_tokens(max_tokens)
        .additional_params(params)
        .build()
}

async fn blocking_usage(model: &AnthropicModel, request: CompletionRequest) -> Usage {
    model
        .completion(request)
        .await
        .expect("completion should succeed")
        .usage
}

/// Drain a provider-native stream and return its terminal record's usage.
async fn streamed_usage(model: &AnthropicModel, request: CompletionRequest) -> Usage {
    let mut stream = model.raw_stream(request).await.expect("stream should open");
    let mut terminal = None;
    while let Some(item) = stream.next().await {
        if let RawStreamingChoice::FinalResponse(record) =
            item.expect("stream item should not error")
        {
            terminal = Some(record);
        }
    }
    Usage::from(
        &terminal
            .expect("stream should yield a terminal record")
            .usage,
    )
}

// ------------------------------------------------------------- blocking ---

#[tokio::test]
async fn blocking_budget_thinking() {
    let observed = Observed::new();
    let slot = observed.clone();
    with_anthropic_reasoning_usage_cassette(
        "reasoning_usage_matrix/blocking_budget_thinking",
        |client| async move {
            let model = client.completion_model(anthropic::completion::CLAUDE_SONNET_4_6);
            slot.record(
                blocking_usage(
                    &model,
                    request(&model, THINKING_PROMPT, budget_thinking(1024), 2048),
                )
                .await,
            );
        },
    )
    .await;

    let scenario = "reasoning_usage_matrix/blocking_budget_thinking";
    observed.assert_matches(scenario, recorded_blocking_thinking_tokens(scenario));
}

#[tokio::test]
async fn blocking_adaptive_declines_to_think() {
    let observed = Observed::new();
    let slot = observed.clone();
    with_anthropic_reasoning_usage_cassette(
        "reasoning_usage_matrix/blocking_adaptive_declines_to_think",
        |client| async move {
            let model = client.completion_model(anthropic::completion::CLAUDE_OPUS_4_7);
            slot.record(
                blocking_usage(
                    &model,
                    request(&model, LONG_REASONING_PROMPT, adaptive_thinking(), 4096),
                )
                .await,
            );
        },
    )
    .await;

    let scenario = "reasoning_usage_matrix/blocking_adaptive_declines_to_think";
    observed
        .assert_breakdown_present_and_zero(scenario, recorded_blocking_thinking_tokens(scenario));
}

#[tokio::test]
async fn blocking_larger_budget() {
    let observed = Observed::new();
    let slot = observed.clone();
    with_anthropic_reasoning_usage_cassette(
        "reasoning_usage_matrix/blocking_larger_budget",
        |client| async move {
            let model = client.completion_model(anthropic::completion::CLAUDE_SONNET_4_6);
            slot.record(
                blocking_usage(
                    &model,
                    request(&model, LONG_REASONING_PROMPT, budget_thinking(4096), 8192),
                )
                .await,
            );
        },
    )
    .await;

    let scenario = "reasoning_usage_matrix/blocking_larger_budget";
    observed.assert_matches(scenario, recorded_blocking_thinking_tokens(scenario));
}

#[tokio::test]
async fn blocking_with_tools() {
    let observed = Observed::new();
    let slot = observed.clone();
    with_anthropic_reasoning_usage_cassette(
        "reasoning_usage_matrix/blocking_with_tools",
        |client| async move {
            let model = client.completion_model(anthropic::completion::CLAUDE_SONNET_4_6);
            let request = model
                .completion_request("Say the single word READY.")
                .max_tokens(2048)
                .additional_params(budget_thinking(1024))
                .tool(multiply_tool())
                .build();
            slot.record(blocking_usage(&model, request).await);
        },
    )
    .await;

    let scenario = "reasoning_usage_matrix/blocking_with_tools";
    observed.assert_matches(scenario, recorded_blocking_thinking_tokens(scenario));
}

#[tokio::test]
async fn blocking_tool_use_terminal() {
    let observed = Observed::new();
    let slot = observed.clone();
    with_anthropic_reasoning_usage_cassette(
        "reasoning_usage_matrix/blocking_tool_use_terminal",
        |client| async move {
            let model = client.completion_model(anthropic::completion::CLAUDE_SONNET_4_6);
            let request = model
                .completion_request("Use the multiply tool to compute 17 times 23.")
                .max_tokens(2048)
                .additional_params(budget_thinking(1024))
                .tool(multiply_tool())
                .build();
            slot.record(blocking_usage(&model, request).await);
        },
    )
    .await;

    let scenario = "reasoning_usage_matrix/blocking_tool_use_terminal";
    observed.assert_matches(scenario, recorded_blocking_thinking_tokens(scenario));
}

#[tokio::test]
async fn blocking_with_preamble() {
    let observed = Observed::new();
    let slot = observed.clone();
    with_anthropic_reasoning_usage_cassette(
        "reasoning_usage_matrix/blocking_with_preamble",
        |client| async move {
            let model = client.completion_model(anthropic::completion::CLAUDE_SONNET_4_6);
            let request = model
                .completion_request(THINKING_PROMPT)
                .preamble("You are a careful arithmetic assistant.".to_string())
                .max_tokens(2048)
                .additional_params(budget_thinking(1024))
                .build();
            slot.record(blocking_usage(&model, request).await);
        },
    )
    .await;

    let scenario = "reasoning_usage_matrix/blocking_with_preamble";
    observed.assert_matches(scenario, recorded_blocking_thinking_tokens(scenario));
}

#[tokio::test]
async fn blocking_with_prompt_caching() {
    let observed = Observed::new();
    let slot = observed.clone();
    with_anthropic_reasoning_usage_cassette(
        "reasoning_usage_matrix/blocking_with_prompt_caching",
        |client| async move {
            let model = client
                .completion_model(anthropic::completion::CLAUDE_SONNET_4_6)
                .with_static_prefix_cache_ttl(anthropic::completion::CacheTtl::OneHour);
            let request = model
                .completion_request(THINKING_PROMPT)
                .preamble("You are a careful arithmetic assistant. ".repeat(200))
                .max_tokens(2048)
                .additional_params(budget_thinking(1024))
                .build();
            slot.record(blocking_usage(&model, request).await);
        },
    )
    .await;

    let scenario = "reasoning_usage_matrix/blocking_with_prompt_caching";
    observed.assert_matches(scenario, recorded_blocking_thinking_tokens(scenario));
}

#[tokio::test]
async fn blocking_redacted_thinking() {
    let observed = Observed::new();
    let slot = observed.clone();
    with_anthropic_reasoning_usage_cassette(
        "reasoning_usage_matrix/blocking_redacted_thinking",
        |client| async move {
            let model = client.completion_model(anthropic::completion::CLAUDE_SONNET_4_6);
            let prompt = format!("{REDACTED_THINKING_MAGIC_STRING} Reply with the single word OK.");
            slot.record(
                blocking_usage(
                    &model,
                    request(&model, &prompt, budget_thinking(1024), 2048),
                )
                .await,
            );
        },
    )
    .await;

    let scenario = "reasoning_usage_matrix/blocking_redacted_thinking";
    observed.assert_matches(scenario, recorded_blocking_thinking_tokens(scenario));
}

/// A turn cut off by `max_tokens` still bills the thinking it did, so the
/// breakdown must survive a truncated turn. The cell asserts the truncation
/// itself from the fixture — otherwise a turn that happened to finish early
/// would leave it covering the same shape as the plain budget cells.
#[tokio::test]
async fn blocking_max_tokens_truncation() {
    let observed = Observed::new();
    let slot = observed.clone();
    with_anthropic_reasoning_usage_cassette(
        "reasoning_usage_matrix/blocking_max_tokens_truncation",
        |client| async move {
            let model = client.completion_model(anthropic::completion::CLAUDE_SONNET_4_6);
            slot.record(
                blocking_usage(
                    &model,
                    request(&model, TRUNCATING_PROMPT, budget_thinking(1024), 1025),
                )
                .await,
            );
        },
    )
    .await;

    let scenario = "reasoning_usage_matrix/blocking_max_tokens_truncation";
    observed.assert_matches(scenario, recorded_blocking_thinking_tokens(scenario));
    assert_eq!(
        recorded_response_body(scenario)["stop_reason"],
        "max_tokens",
        "this cell exists to cover a truncated turn",
    );
}

#[tokio::test]
async fn blocking_opus_model() {
    let observed = Observed::new();
    let slot = observed.clone();
    with_anthropic_reasoning_usage_cassette(
        "reasoning_usage_matrix/blocking_opus_model",
        |client| async move {
            // Opus 4.8 rejects `thinking.type.enabled` outright — it takes
            // `adaptive` plus `output_config.effort` instead — so the second
            // model family can only be covered on the adaptive path.
            let model = client.completion_model(anthropic::completion::CLAUDE_OPUS_4_8);
            slot.record(
                blocking_usage(
                    &model,
                    request(&model, LONG_REASONING_PROMPT, adaptive_thinking(), 4096),
                )
                .await,
            );
        },
    )
    .await;

    let scenario = "reasoning_usage_matrix/blocking_opus_model";
    observed.assert_matches(scenario, recorded_blocking_thinking_tokens(scenario));
}

#[tokio::test]
async fn blocking_long_reasoning() {
    let observed = Observed::new();
    let slot = observed.clone();
    with_anthropic_reasoning_usage_cassette(
        "reasoning_usage_matrix/blocking_long_reasoning",
        |client| async move {
            let model = client.completion_model(anthropic::completion::CLAUDE_SONNET_4_6);
            slot.record(
                blocking_usage(
                    &model,
                    request(&model, LONG_REASONING_PROMPT, budget_thinking(8192), 12288),
                )
                .await,
            );
        },
    )
    .await;

    let scenario = "reasoning_usage_matrix/blocking_long_reasoning";
    let recorded = recorded_blocking_thinking_tokens(scenario);
    observed.assert_matches(scenario, recorded);
    assert!(
        recorded.is_some_and(|tokens| tokens > 100),
        "this cell exists to cover a large breakdown, got {recorded:?}",
    );
}

#[tokio::test]
async fn blocking_thinking_disabled_control() {
    let observed = Observed::new();
    let slot = observed.clone();
    with_anthropic_reasoning_usage_cassette(
        "reasoning_usage_matrix/blocking_thinking_disabled_control",
        |client| async move {
            let model = client.completion_model(anthropic::completion::CLAUDE_SONNET_4_6);
            let request = model
                .completion_request(THINKING_PROMPT)
                .max_tokens(64)
                .build();
            slot.record(blocking_usage(&model, request).await);
        },
    )
    .await;

    let scenario = "reasoning_usage_matrix/blocking_thinking_disabled_control";
    observed.assert_breakdown_absent(scenario, recorded_blocking_thinking_tokens(scenario));
}

#[tokio::test]
async fn blocking_haiku_no_thinking_control() {
    let observed = Observed::new();
    let slot = observed.clone();
    with_anthropic_reasoning_usage_cassette(
        "reasoning_usage_matrix/blocking_haiku_no_thinking_control",
        |client| async move {
            let model = client.completion_model(anthropic::completion::CLAUDE_HAIKU_4_5);
            let request = model
                .completion_request(THINKING_PROMPT)
                .max_tokens(64)
                .build();
            slot.record(blocking_usage(&model, request).await);
        },
    )
    .await;

    let scenario = "reasoning_usage_matrix/blocking_haiku_no_thinking_control";
    observed.assert_breakdown_absent(scenario, recorded_blocking_thinking_tokens(scenario));
}

// ------------------------------------------------------------ streaming ---

#[tokio::test]
async fn streaming_budget_thinking() {
    let observed = Observed::new();
    let slot = observed.clone();
    with_anthropic_reasoning_usage_cassette(
        "reasoning_usage_matrix/streaming_budget_thinking",
        |client| async move {
            let model = client.completion_model(anthropic::completion::CLAUDE_SONNET_4_6);
            slot.record(
                streamed_usage(
                    &model,
                    request(&model, THINKING_PROMPT, budget_thinking(1024), 2048),
                )
                .await,
            );
        },
    )
    .await;

    let scenario = "reasoning_usage_matrix/streaming_budget_thinking";
    observed.assert_matches(scenario, recorded_streamed_thinking_tokens(scenario));
}

#[tokio::test]
async fn streaming_adaptive_declines_to_think() {
    let observed = Observed::new();
    let slot = observed.clone();
    with_anthropic_reasoning_usage_cassette(
        "reasoning_usage_matrix/streaming_adaptive_declines_to_think",
        |client| async move {
            let model = client.completion_model(anthropic::completion::CLAUDE_OPUS_4_7);
            slot.record(
                streamed_usage(
                    &model,
                    request(&model, LONG_REASONING_PROMPT, adaptive_thinking(), 4096),
                )
                .await,
            );
        },
    )
    .await;

    let scenario = "reasoning_usage_matrix/streaming_adaptive_declines_to_think";
    observed
        .assert_breakdown_present_and_zero(scenario, recorded_streamed_thinking_tokens(scenario));
}

#[tokio::test]
async fn streaming_larger_budget() {
    let observed = Observed::new();
    let slot = observed.clone();
    with_anthropic_reasoning_usage_cassette(
        "reasoning_usage_matrix/streaming_larger_budget",
        |client| async move {
            let model = client.completion_model(anthropic::completion::CLAUDE_SONNET_4_6);
            slot.record(
                streamed_usage(
                    &model,
                    request(&model, LONG_REASONING_PROMPT, budget_thinking(4096), 8192),
                )
                .await,
            );
        },
    )
    .await;

    let scenario = "reasoning_usage_matrix/streaming_larger_budget";
    observed.assert_matches(scenario, recorded_streamed_thinking_tokens(scenario));
}

#[tokio::test]
async fn streaming_with_tools() {
    let observed = Observed::new();
    let slot = observed.clone();
    with_anthropic_reasoning_usage_cassette(
        "reasoning_usage_matrix/streaming_with_tools",
        |client| async move {
            let model = client.completion_model(anthropic::completion::CLAUDE_SONNET_4_6);
            let request = model
                .completion_request("Say the single word READY.")
                .max_tokens(2048)
                .additional_params(budget_thinking(1024))
                .tool(multiply_tool())
                .build();
            slot.record(streamed_usage(&model, request).await);
        },
    )
    .await;

    let scenario = "reasoning_usage_matrix/streaming_with_tools";
    observed.assert_matches(scenario, recorded_streamed_thinking_tokens(scenario));
}

#[tokio::test]
async fn streaming_tool_use_terminal() {
    let observed = Observed::new();
    let slot = observed.clone();
    with_anthropic_reasoning_usage_cassette(
        "reasoning_usage_matrix/streaming_tool_use_terminal",
        |client| async move {
            let model = client.completion_model(anthropic::completion::CLAUDE_SONNET_4_6);
            let request = model
                .completion_request("Use the multiply tool to compute 17 times 23.")
                .max_tokens(2048)
                .additional_params(budget_thinking(1024))
                .tool(multiply_tool())
                .build();
            slot.record(streamed_usage(&model, request).await);
        },
    )
    .await;

    let scenario = "reasoning_usage_matrix/streaming_tool_use_terminal";
    observed.assert_matches(scenario, recorded_streamed_thinking_tokens(scenario));
}

#[tokio::test]
async fn streaming_with_preamble() {
    let observed = Observed::new();
    let slot = observed.clone();
    with_anthropic_reasoning_usage_cassette(
        "reasoning_usage_matrix/streaming_with_preamble",
        |client| async move {
            let model = client.completion_model(anthropic::completion::CLAUDE_SONNET_4_6);
            let request = model
                .completion_request(THINKING_PROMPT)
                .preamble("You are a careful arithmetic assistant.".to_string())
                .max_tokens(2048)
                .additional_params(budget_thinking(1024))
                .build();
            slot.record(streamed_usage(&model, request).await);
        },
    )
    .await;

    let scenario = "reasoning_usage_matrix/streaming_with_preamble";
    observed.assert_matches(scenario, recorded_streamed_thinking_tokens(scenario));
}

#[tokio::test]
async fn streaming_with_prompt_caching() {
    let observed = Observed::new();
    let slot = observed.clone();
    with_anthropic_reasoning_usage_cassette(
        "reasoning_usage_matrix/streaming_with_prompt_caching",
        |client| async move {
            let model = client
                .completion_model(anthropic::completion::CLAUDE_SONNET_4_6)
                .with_static_prefix_cache_ttl(anthropic::completion::CacheTtl::OneHour);
            let request = model
                .completion_request(THINKING_PROMPT)
                .preamble("You are a careful arithmetic assistant. ".repeat(200))
                .max_tokens(2048)
                .additional_params(budget_thinking(1024))
                .build();
            slot.record(streamed_usage(&model, request).await);
        },
    )
    .await;

    let scenario = "reasoning_usage_matrix/streaming_with_prompt_caching";
    observed.assert_matches(scenario, recorded_streamed_thinking_tokens(scenario));
}

#[tokio::test]
async fn streaming_redacted_thinking() {
    let observed = Observed::new();
    let slot = observed.clone();
    with_anthropic_reasoning_usage_cassette(
        "reasoning_usage_matrix/streaming_redacted_thinking",
        |client| async move {
            let model = client.completion_model(anthropic::completion::CLAUDE_SONNET_4_6);
            let prompt = format!("{REDACTED_THINKING_MAGIC_STRING} Reply with the single word OK.");
            slot.record(
                streamed_usage(
                    &model,
                    request(&model, &prompt, budget_thinking(1024), 2048),
                )
                .await,
            );
        },
    )
    .await;

    let scenario = "reasoning_usage_matrix/streaming_redacted_thinking";
    observed.assert_matches(scenario, recorded_streamed_thinking_tokens(scenario));
}

#[tokio::test]
async fn streaming_max_tokens_truncation() {
    let observed = Observed::new();
    let slot = observed.clone();
    with_anthropic_reasoning_usage_cassette(
        "reasoning_usage_matrix/streaming_max_tokens_truncation",
        |client| async move {
            let model = client.completion_model(anthropic::completion::CLAUDE_SONNET_4_6);
            slot.record(
                streamed_usage(
                    &model,
                    request(&model, TRUNCATING_PROMPT, budget_thinking(1024), 1025),
                )
                .await,
            );
        },
    )
    .await;

    let scenario = "reasoning_usage_matrix/streaming_max_tokens_truncation";
    observed.assert_matches(scenario, recorded_streamed_thinking_tokens(scenario));
    assert!(
        recorded_streamed_stop_reason(scenario).as_deref() == Some("max_tokens"),
        "this cell exists to cover a truncated turn, got {:?}",
        recorded_streamed_stop_reason(scenario),
    );
}

#[tokio::test]
async fn streaming_opus_model() {
    let observed = Observed::new();
    let slot = observed.clone();
    with_anthropic_reasoning_usage_cassette(
        "reasoning_usage_matrix/streaming_opus_model",
        |client| async move {
            // Opus 4.8 rejects `thinking.type.enabled`; see the blocking twin.
            let model = client.completion_model(anthropic::completion::CLAUDE_OPUS_4_8);
            slot.record(
                streamed_usage(
                    &model,
                    request(&model, LONG_REASONING_PROMPT, adaptive_thinking(), 4096),
                )
                .await,
            );
        },
    )
    .await;

    let scenario = "reasoning_usage_matrix/streaming_opus_model";
    observed.assert_matches(scenario, recorded_streamed_thinking_tokens(scenario));
}

#[tokio::test]
async fn streaming_long_reasoning() {
    let observed = Observed::new();
    let slot = observed.clone();
    with_anthropic_reasoning_usage_cassette(
        "reasoning_usage_matrix/streaming_long_reasoning",
        |client| async move {
            let model = client.completion_model(anthropic::completion::CLAUDE_SONNET_4_6);
            slot.record(
                streamed_usage(
                    &model,
                    request(&model, LONG_REASONING_PROMPT, budget_thinking(8192), 12288),
                )
                .await,
            );
        },
    )
    .await;

    let scenario = "reasoning_usage_matrix/streaming_long_reasoning";
    let recorded = recorded_streamed_thinking_tokens(scenario);
    observed.assert_matches(scenario, recorded);
    assert!(
        recorded.is_some_and(|tokens| tokens > 100),
        "this cell exists to cover a large breakdown, got {recorded:?}",
    );
}

#[tokio::test]
async fn streaming_thinking_disabled_control() {
    let observed = Observed::new();
    let slot = observed.clone();
    with_anthropic_reasoning_usage_cassette(
        "reasoning_usage_matrix/streaming_thinking_disabled_control",
        |client| async move {
            let model = client.completion_model(anthropic::completion::CLAUDE_SONNET_4_6);
            let request = model
                .completion_request(THINKING_PROMPT)
                .max_tokens(64)
                .build();
            slot.record(streamed_usage(&model, request).await);
        },
    )
    .await;

    let scenario = "reasoning_usage_matrix/streaming_thinking_disabled_control";
    observed.assert_breakdown_absent(scenario, recorded_streamed_thinking_tokens(scenario));
}

// ----------------------------------------------------- adjacent surfaces ---

/// The normalized stream shares the terminal record the raw stream exposes, so
/// `StreamingCompletionResponse::usage()` must report the same breakdown.
#[tokio::test]
async fn normalized_stream_budget_thinking() {
    let observed = Observed::new();
    let slot = observed.clone();
    with_anthropic_reasoning_usage_cassette(
        "reasoning_usage_matrix/normalized_stream_budget_thinking",
        |client| async move {
            let model = client.completion_model(anthropic::completion::CLAUDE_SONNET_4_6);
            let mut stream = model
                .stream(request(
                    &model,
                    THINKING_PROMPT,
                    budget_thinking(1024),
                    2048,
                ))
                .await
                .expect("stream should open");
            while stream.next().await.is_some() {}
            slot.record(stream.usage());
        },
    )
    .await;

    let scenario = "reasoning_usage_matrix/normalized_stream_budget_thinking";
    observed.assert_matches(scenario, recorded_streamed_thinking_tokens(scenario));
}

/// The agent surface reaches the same mapping by a different route: usage
/// arrives through `PromptRequest`'s per-call record
/// (`completion_calls[].usage`) rather than off a `CompletionResponse`
/// directly, so a regression in that plumbing would not show up in any raw
/// model cell above. The preamble differs from `blocking_with_preamble` only
/// so the two fixtures stay distinguishable — the wire shapes are equivalent
/// by design, which is part of what this cell asserts.
#[tokio::test]
async fn agent_blocking_thinking() {
    let observed = Observed::new();
    let slot = observed.clone();
    with_anthropic_reasoning_usage_cassette(
        "reasoning_usage_matrix/agent_blocking_thinking",
        |client| async move {
            let agent = client
                .agent(anthropic::completion::CLAUDE_SONNET_4_6)
                .preamble("You are a meticulous arithmetic assistant.")
                .max_tokens(2048)
                .additional_params(budget_thinking(1024))
                .build();
            let response = agent
                .prompt(THINKING_PROMPT)
                .extended_details()
                .await
                .expect("agent run should succeed");
            let call = response
                .completion_calls
                .first()
                .expect("the run makes at least one completion call");
            slot.record(call.usage);
        },
    )
    .await;

    let scenario = "reasoning_usage_matrix/agent_blocking_thinking";
    observed.assert_matches(scenario, recorded_blocking_thinking_tokens(scenario));
}

// ----------------------------------------------------------------- unit ---

/// Serde: a turn that reports no breakdown yields zero, not a parse failure.
#[test]
fn unit_absent_details_reports_zero() {
    let usage: anthropic::completion::Usage = serde_json::from_value(json!({
        "input_tokens": 10,
        "output_tokens": 20,
        "cache_read_input_tokens": null,
        "cache_creation_input_tokens": null
    }))
    .expect("usage without a breakdown should parse");

    assert!(usage.output_tokens_details.is_none());
    assert_eq!(Usage::from(&usage).reasoning_tokens, 0);
}

/// Serde: a bucket Anthropic adds later must not break the known one.
#[test]
fn unit_unknown_detail_bucket_is_ignored() {
    let usage: anthropic::completion::Usage = serde_json::from_value(json!({
        "input_tokens": 10,
        "output_tokens": 20,
        "cache_read_input_tokens": null,
        "cache_creation_input_tokens": null,
        "output_tokens_details": { "thinking_tokens": 7, "future_bucket_tokens": 3 }
    }))
    .expect("an unknown detail bucket should be ignored");

    assert_eq!(Usage::from(&usage).reasoning_tokens, 7);
}

/// The breakdown is inside `output_tokens`, so the total must not grow.
#[test]
fn unit_thinking_tokens_stay_out_of_the_total() {
    let with_thinking: anthropic::completion::Usage = serde_json::from_value(json!({
        "input_tokens": 10,
        "output_tokens": 20,
        "cache_read_input_tokens": 3,
        "cache_creation_input_tokens": 4,
        "output_tokens_details": { "thinking_tokens": 15 }
    }))
    .expect("usage should parse");
    let without: anthropic::completion::Usage = serde_json::from_value(json!({
        "input_tokens": 10,
        "output_tokens": 20,
        "cache_read_input_tokens": 3,
        "cache_creation_input_tokens": 4
    }))
    .expect("usage should parse");

    let with_thinking = Usage::from(&with_thinking);
    let without = Usage::from(&without);

    assert_eq!(with_thinking.reasoning_tokens, 15);
    assert_eq!(without.reasoning_tokens, 0);
    assert_eq!(
        with_thinking.total_tokens, without.total_tokens,
        "the breakdown is already counted in output_tokens",
    );
    assert_eq!(with_thinking.total_tokens, 10 + 3 + 4 + 20);
}

/// Blocking and streaming go through the same totals helper, so the same
/// counters must normalize identically — including the breakdown that arrives
/// on the streaming path's terminal `message_delta` rather than its
/// `message_start`.
#[test]
fn unit_streaming_and_blocking_share_the_mapping() {
    let blocking: anthropic::completion::Usage = serde_json::from_value(json!({
        "input_tokens": 100,
        "output_tokens": 200,
        "cache_read_input_tokens": 5,
        "cache_creation_input_tokens": 6,
        "output_tokens_details": { "thinking_tokens": 42 }
    }))
    .expect("usage should parse");
    let streamed: anthropic::streaming::PartialUsage = serde_json::from_value(json!({
        "input_tokens": 100,
        "output_tokens": 200,
        "cache_read_input_tokens": 5,
        "cache_creation_input_tokens": 6,
        "output_tokens_details": { "thinking_tokens": 42 }
    }))
    .expect("partial usage should parse");

    assert_eq!(Usage::from(&blocking), Usage::from(&streamed));
    assert_eq!(Usage::from(&streamed).reasoning_tokens, 42);
}
