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
//! | 5 | `blocking_tool_use_terminal` | terminal is `tool_use` | `> 0` | recorded |
//! | 9 | `blocking_max_tokens_truncation` | `stop_reason: max_tokens` | `> 0` | recorded |
//! | 15 | `streaming_adaptive_declines_to_think` | streamed twin of #2 | `0` | recorded |
//! | 18 | `streaming_tool_use_terminal` | streamed twin of #5 | `> 0` | recorded |
//! | 22 | `streaming_max_tokens_truncation` | streamed twin of #9 | `> 0` | recorded |
//! | 28 | `unit_absent_details_reports_none` | decoder: no `output_tokens_details` | none | unit |
//! | 29 | `unit_unknown_detail_bucket_is_ignored` | decoder: forward compatibility | `> 0` | unit |
//! | 30 | `unit_thinking_tokens_stay_out_of_the_total` | totals arithmetic | n/a | unit |
//! | 31 | `unit_streaming_and_blocking_share_the_mapping` | one usage reader for both modes | n/a | unit |
//!
//! Cells 28–31 are unit tests because no live turn can vary what they assert:
//! whether the decoder tolerates an absent or unknown bucket, and whether the
//! breakdown is excluded from `total_tokens`, are properties of its one usage
//! reader, not of any provider response.
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
use rig::completion::{CompletionRequest, ToolDefinition, Usage};
use rig::driver::Model;
use rig::providers::anthropic;
use rig::providers::anthropic::wire::Messages;
use serde_json::json;

use super::super::support::{recorded_response_body, with_anthropic_reasoning_usage_cassette};

type AnthropicModel = Model<Messages>;

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

pub(super) fn budget_thinking(budget: u64) -> serde_json::Value {
    json!({ "thinking": { "type": "enabled", "budget_tokens": budget } })
}

fn adaptive_thinking() -> serde_json::Value {
    json!({ "thinking": { "type": "adaptive" } })
}

fn multiply_tool() -> ToolDefinition {
    ToolDefinition {
        name: rig_core::message::ToolName::new("multiply").expect("tool name"),
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
pub(super) struct Observed(Arc<Mutex<Option<Usage>>>);

impl Observed {
    pub(super) fn new() -> Self {
        Self::default()
    }

    pub(super) fn record(&self, usage: Usage) {
        *self.0.lock().expect("observed usage lock") = Some(usage);
    }

    fn take(&self, scenario: &str) -> Usage {
        (*self.0.lock().expect("observed usage lock"))
            .unwrap_or_else(|| panic!("{scenario}: the cell body never recorded a usage"))
    }

    /// A thinking cell: the parsed `reasoning_tokens` must equal the fixture's
    /// own recorded breakdown, be non-zero, and stay inside `output_tokens`
    /// without inflating the total.
    pub(super) fn assert_matches(&self, scenario: &str, recorded: Option<u64>) {
        let usage = self.take(scenario);
        let recorded = recorded.unwrap_or_else(|| {
            panic!("{scenario}: the recorded turn reports no output_tokens_details")
        });

        assert!(
            recorded > 0,
            "{scenario}: the recorded turn must actually have spent thinking tokens",
        );
        assert_eq!(
            usage.reasoning_tokens,
            Some(recorded),
            "{scenario}: reasoning_tokens must equal the recorded thinking_tokens",
        );
        let reasoning = usage.reasoning_tokens.unwrap_or(0);
        let output = usage.output_tokens.unwrap_or(0);
        assert!(
            reasoning <= output,
            "{scenario}: thinking is a breakdown of output_tokens ({reasoning} > {output})",
        );
        let input = usage.input_tokens.unwrap_or(0);
        assert!(
            usage.cached_input_tokens.unwrap_or(0) + usage.cache_creation_input_tokens.unwrap_or(0)
                <= input,
            "{scenario}: input includes the cache reads and writes: {usage:?}",
        );
        assert_eq!(
            usage.total_tokens,
            Some(input + output),
            "{scenario}: the total is input plus output, reasoning not added again",
        );
    }

    /// Control: adaptive thinking was requested and the model chose not to
    /// think, so the breakdown is *present and zero* — the other zero shape,
    /// which reaches `reasoning_tokens` as `Some(0)` rather than `None`.
    fn assert_breakdown_present_and_zero(&self, scenario: &str, recorded: Option<u64>) {
        let usage = self.take(scenario);
        assert_eq!(
            recorded,
            Some(0),
            "{scenario}: adaptive thinking that declines still reports the bucket",
        );
        assert_eq!(
            usage.reasoning_tokens,
            Some(0),
            "{scenario}: a zero breakdown reports a zero reasoning counter",
        );
        assert!(
            usage.output_tokens.is_some_and(|n| n > 0),
            "{scenario}: the turn still produced output",
        );
    }
}

/// The `thinking_tokens` a blocking cassette's recorded response body reports.
pub(super) fn recorded_blocking_thinking_tokens(scenario: &str) -> Option<u64> {
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
    // Single terminal frame, asserted like its sibling reader above rather
    // than assumed, so the two do not disagree about their own premise.
    let mut frames = contents
        .lines()
        .filter(|line| line.contains(r#""type":"message_delta""#));
    let frame = frames.next()?;
    assert!(
        frames.next().is_none(),
        "cassette {} records more than one message_delta; this reader assumes a \
         single-turn cell",
        path.display()
    );
    let (_, after) = frame.split_once(r#""stop_reason":""#)?;
    after.split('"').next().map(str::to_string)
}

fn request(prompt: &str, params: serde_json::Value, max_tokens: u64) -> CompletionRequest {
    CompletionRequest::new(prompt)
        .max_tokens(max_tokens)
        .additional_params(params)
}

async fn blocking_usage(model: &AnthropicModel, request: CompletionRequest) -> Usage {
    model
        .call(request)
        .await
        .expect("completion should succeed")
        .usage
}

/// Drain a provider-native stream and return its terminal record's usage.
async fn streamed_usage(model: &AnthropicModel, request: CompletionRequest) -> Usage {
    let mut stream = model.stream(request).expect("stream should open");
    while let Some(item) = stream.next().await {
        item.expect("stream item should not error");
    }
    stream
        .finish()
        .await
        .expect("stream should yield a terminal record")
        .usage
}

// ------------------------------------------------------------- blocking ---

#[tokio::test]
async fn blocking_tool_use_terminal() {
    let observed = Observed::new();
    let slot = observed.clone();
    with_anthropic_reasoning_usage_cassette(
        "reasoning_usage_matrix/blocking_tool_use_terminal",
        |client| async move {
            let model = client.completion(anthropic::completion::CLAUDE_SONNET_4_6);
            let request = CompletionRequest::new("Use the multiply tool to compute 17 times 23.")
                .max_tokens(2048)
                .additional_params(budget_thinking(1024))
                .tool(multiply_tool());
            slot.record(blocking_usage(&model, request).await);
        },
    )
    .await;

    let scenario = "reasoning_usage_matrix/blocking_tool_use_terminal";
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
            let model = client.completion(anthropic::completion::CLAUDE_SONNET_4_6);
            slot.record(
                blocking_usage(
                    &model,
                    request(TRUNCATING_PROMPT, budget_thinking(1024), 1025),
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

// ------------------------------------------------------------ streaming ---

#[tokio::test]
async fn streaming_adaptive_declines_to_think() {
    let observed = Observed::new();
    let slot = observed.clone();
    with_anthropic_reasoning_usage_cassette(
        "reasoning_usage_matrix/streaming_adaptive_declines_to_think",
        |client| async move {
            let model = client.completion(anthropic::completion::CLAUDE_OPUS_4_7);
            slot.record(
                streamed_usage(
                    &model,
                    request(LONG_REASONING_PROMPT, adaptive_thinking(), 4096),
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
async fn streaming_tool_use_terminal() {
    let observed = Observed::new();
    let slot = observed.clone();
    with_anthropic_reasoning_usage_cassette(
        "reasoning_usage_matrix/streaming_tool_use_terminal",
        |client| async move {
            let model = client.completion(anthropic::completion::CLAUDE_SONNET_4_6);
            let request = CompletionRequest::new("Use the multiply tool to compute 17 times 23.")
                .max_tokens(2048)
                .additional_params(budget_thinking(1024))
                .tool(multiply_tool());
            slot.record(streamed_usage(&model, request).await);
        },
    )
    .await;

    let scenario = "reasoning_usage_matrix/streaming_tool_use_terminal";
    observed.assert_matches(scenario, recorded_streamed_thinking_tokens(scenario));
}

#[tokio::test]
async fn streaming_max_tokens_truncation() {
    let observed = Observed::new();
    let slot = observed.clone();
    with_anthropic_reasoning_usage_cassette(
        "reasoning_usage_matrix/streaming_max_tokens_truncation",
        |client| async move {
            let model = client.completion(anthropic::completion::CLAUDE_SONNET_4_6);
            slot.record(
                streamed_usage(
                    &model,
                    request(TRUNCATING_PROMPT, budget_thinking(1024), 1025),
                )
                .await,
            );
        },
    )
    .await;

    let scenario = "reasoning_usage_matrix/streaming_max_tokens_truncation";
    observed.assert_matches(scenario, recorded_streamed_thinking_tokens(scenario));
    assert_eq!(
        recorded_streamed_stop_reason(scenario).as_deref(),
        Some("max_tokens"),
        "this cell exists to cover a truncated turn",
    );
}

// ----------------------------------------------------- adjacent surfaces ---

// ----------------------------------------------------------------- unit ---

/// `usage` as the Messages decoder reads it, off a whole reply or off a
/// stream whose terminal `message_delta` carries it.
fn decoded_usage(usage: serde_json::Value, streamed: bool) -> Usage {
    use rig::wire::{Mode, WireFrame};
    let wire = anthropic::Anthropic::new("test-key")
        .completion(anthropic::completion::CLAUDE_SONNET_4_6)
        .wire;
    let message = |usage: serde_json::Value, stop_reason: serde_json::Value| {
        json!({"type": "message", "id": "msg_1", "role": "assistant",
            "model": anthropic::completion::CLAUDE_SONNET_4_6,
            "content": [{"type": "text", "text": "done"}],
            "stop_reason": stop_reason, "usage": usage})
    };
    let (mode, frames) = if streamed {
        let start = json!({"type": "message_start",
            "message": message(json!({"input_tokens": 1, "output_tokens": 1}), json!(null))});
        let delta = json!({"type": "message_delta", "delta": {"stop_reason": "end_turn"},
            "usage": usage});
        (Mode::Streaming, vec![start, delta])
    } else {
        (Mode::Unary, vec![message(usage, json!("end_turn"))])
    };
    rig_core::test_utils::history_conformance::decode(
        &wire,
        &CompletionRequest::new("hello"),
        mode,
        frames
            .into_iter()
            .map(|frame| WireFrame::Text(frame.to_string())),
    )
    .expect("the reply decodes")
    .usage
}

/// A turn that reports no breakdown yields no counter, not a failure.
#[test]
fn unit_absent_details_reports_none() {
    let usage = decoded_usage(
        json!({
            "input_tokens": 10,
            "output_tokens": 20,
            "cache_read_input_tokens": null,
            "cache_creation_input_tokens": null
        }),
        false,
    );
    assert_eq!(usage.reasoning_tokens, None);
}

/// A bucket Anthropic adds later must not break the known one.
#[test]
fn unit_unknown_detail_bucket_is_ignored() {
    let usage = decoded_usage(
        json!({
            "input_tokens": 10,
            "output_tokens": 20,
            "output_tokens_details": { "thinking_tokens": 7, "future_bucket_tokens": 3 }
        }),
        false,
    );
    assert_eq!(usage.reasoning_tokens, Some(7));
}

/// The breakdown is inside `output_tokens`, so the total must not grow.
#[test]
fn unit_thinking_tokens_stay_out_of_the_total() {
    let counts = json!({
        "input_tokens": 10,
        "output_tokens": 20,
        "cache_read_input_tokens": 3,
        "cache_creation_input_tokens": 4
    });
    let mut thinking = counts.clone();
    thinking["output_tokens_details"] = json!({ "thinking_tokens": 15 });
    let with_thinking = decoded_usage(thinking, false);
    let without = decoded_usage(counts, false);

    assert_eq!(with_thinking.reasoning_tokens, Some(15));
    assert_eq!(without.reasoning_tokens, None);
    assert_eq!(
        with_thinking.total_tokens, without.total_tokens,
        "the breakdown is already counted in output_tokens",
    );
    assert_eq!(with_thinking.total_tokens, Some(10 + 3 + 4 + 20));
}

/// Blocking and streaming read usage through the same reader, so the same
/// counters must normalize identically, including the breakdown that
/// arrives on the streaming path's terminal `message_delta` rather than its
/// `message_start`.
#[test]
fn unit_streaming_and_blocking_share_the_mapping() {
    let counts = json!({
        "input_tokens": 100,
        "output_tokens": 200,
        "cache_read_input_tokens": 5,
        "cache_creation_input_tokens": 6,
        "output_tokens_details": { "thinking_tokens": 42 }
    });
    let streamed = decoded_usage(counts.clone(), true);
    assert_eq!(decoded_usage(counts, false), streamed);
    assert_eq!(streamed.reasoning_tokens, Some(42));
}
