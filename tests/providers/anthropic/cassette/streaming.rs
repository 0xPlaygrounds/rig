//! Anthropic streaming smoke test.

use rig::prelude::*;
use rig::providers::anthropic;
use rig::streaming::StreamingPrompt;

use super::super::support::{with_anthropic_cassette, with_anthropic_gateway_cassette};
use crate::support::{
    STREAMING_PREAMBLE, STREAMING_PROMPT, assert_nonempty_response,
    collect_stream_final_response_and_provider_final,
};

#[tokio::test]
async fn streaming_smoke() {
    with_anthropic_cassette("streaming/streaming_smoke", |client| async move {
        let agent = client
            .agent(anthropic::completion::CLAUDE_SONNET_4_6)
            .preamble(STREAMING_PREAMBLE)
            .build();

        let mut stream = agent.stream_prompt(STREAMING_PROMPT).await;
        let (response, provider_final): (_, rig::streaming::StreamFinal) =
            collect_stream_final_response_and_provider_final(&mut stream)
                .await
                .expect("streaming prompt should succeed");

        assert_nonempty_response(&response);
        assert_eq!(provider_final.provider, "anthropic");
        assert!(provider_final.usage.total_tokens > 0);
    })
    .await;
}

/// Regression: the streamed terminal must carry the metadata the provider puts
/// on `message_delta`.
///
/// Two signals used to be dropped. `stop_reason` never reached the consumer, so
/// a `max_tokens` truncation was indistinguishable from a natural stop. And
/// `input_tokens` was read only from `message_start`, which Anthropic-compatible
/// gateways may report as `0` while sending the real prompt size on
/// `message_delta`, silently yielding `Usage { input_tokens: 0 }`.
///
/// Recorded against OpenRouter's Anthropic Messages endpoint rather than
/// `api.anthropic.com`, because the *disagreement* is what Anthropic proper does
/// not produce: it reports the count on both frames and they always match (see
/// every streaming cassette under `tests/cassettes/anthropic/`), so a recording
/// from Anthropic passes whether or not the bug is present and cannot witness
/// it. `max_tokens` is capped low so one recording carries both signals.
///
/// **Re-recording this cassette needs care.** The split is a property of the
/// upstream OpenRouter routed to, not of OpenRouter itself: this fixture's
/// `message_start` says `"provider":"Amazon Bedrock"`, and the request pins no
/// routing preference. Routed to Anthropic directly, OpenRouter would relay
/// Anthropic's agreeing frames and the recording would stop witnessing the bug.
/// The `message_start` assertion after the wrapper exists to make that fail
/// loudly rather than pass vacuously — if it trips after a re-record, the new
/// recording did not reproduce the divergence and pinning `provider` in the
/// request (or keeping the old fixture) is the fix, not relaxing the assertion.
#[tokio::test]
async fn gateway_reports_input_tokens_on_message_delta() {
    // Spelled out rather than shared with the constant below: the cassette-safety
    // scan reads scenarios off these call sites syntactically and rejects
    // anything that is not a string literal.
    with_anthropic_gateway_cassette(
        "streaming/gateway_message_delta_metadata",
        |client| async move {
            let agent = client
                .agent("anthropic/claude-haiku-4.5")
                .preamble(STREAMING_PREAMBLE)
                .max_tokens(16)
                .build();

            let mut stream = agent.stream_prompt(STREAMING_PROMPT).await;
            let (_response, provider_final): (_, rig::streaming::StreamFinal) =
                collect_stream_final_response_and_provider_final(&mut stream)
                    .await
                    .expect("streaming prompt should succeed");

            assert_eq!(provider_final.provider, "anthropic");
            assert_eq!(
                provider_final.finish_reason,
                Some(rig::completion::FinishReason::Length),
                "a max_tokens truncation must be distinguishable from a natural stop"
            );
            assert_eq!(
                provider_final.usage.input_tokens, 32,
                "the prompt size the gateway reported on message_delta must reach the consumer"
            );
        },
    )
    .await;

    // Anchors the count above to the frame it has to have come from: the fixture
    // is only a regression test while the two frames disagree.
    //
    // Read out here rather than inside the body because in record mode the
    // cassette is written by `finish_after_test` only once the body returns — an
    // in-body read would assert against the *previous* recording (or panic
    // before anything is written at all, on a checkout that has no fixture yet).
    assert_eq!(
        recorded_message_start_input_tokens(GATEWAY_METADATA_SCENARIO),
        0,
        "this fixture must keep reporting 0 input_tokens on message_start — without \
         the disagreement the assertion above passes with or without the fix (see \
         this test's doc comment before re-recording)"
    );
}

/// Regression: the premise the `input_tokens` preference rests on — Anthropic
/// proper reports the prompt size on **both** `message_start` and the terminal
/// `message_delta`, and the two agree.
///
/// `providers::anthropic::streaming` states this as the reason the preference is
/// safe (preferring the delta cannot change what Anthropic-proper turns report,
/// because the frames carry the same number). Nothing pinned it: the existing
/// cassettes satisfy it incidentally, so a future wire change that split the two
/// values on `api.anthropic.com` would silently alter every streamed turn's
/// reported prompt size with no test objecting.
///
/// Recorded against `api.anthropic.com`, unlike the gateway fixture above, because
/// the claim under test is specifically about Anthropic proper.
#[tokio::test]
async fn anthropic_proper_agrees_on_input_tokens_across_both_frames() {
    with_anthropic_cassette(
        "streaming/input_tokens_agree_across_frames",
        |client| async move {
            let agent = client
                .agent(anthropic::completion::CLAUDE_SONNET_4_6)
                .preamble(STREAMING_PREAMBLE)
                .max_tokens(64)
                .build();

            let mut stream = agent.stream_prompt(STREAMING_PROMPT).await;
            let (_response, provider_final): (_, rig::streaming::StreamFinal) =
                collect_stream_final_response_and_provider_final(&mut stream)
                    .await
                    .expect("streaming prompt should succeed");

            assert!(
                provider_final.usage.input_tokens > 0,
                "Anthropic proper reports a real prompt size"
            );
        },
    )
    .await;

    // The premise itself, read off the recorded frames: both carry the count and
    // the values match. Read after the wrapper returns so a re-record is what
    // gets asserted (record mode writes the cassette only once the body ends).
    let (start, delta) = recorded_input_tokens_on_both_frames(AGREEMENT_SCENARIO);
    assert_eq!(
        start, delta,
        "api.anthropic.com must keep reporting the same input_tokens on \
         message_start and message_delta — the streaming adapter's preference for \
         the delta is documented as safe *because* they agree. If this trips, \
         Anthropic's wire changed and the comment in anthropic/streaming.rs is \
         now wrong."
    );
}

const AGREEMENT_SCENARIO: &str = "streaming/input_tokens_agree_across_frames";

/// The `input_tokens` a cassette reports on `message_start` and on the terminal
/// `message_delta`. Panics if either frame omits the field — for this fixture,
/// an omission is itself the regression.
fn recorded_input_tokens_on_both_frames(scenario: &str) -> (u64, u64) {
    let path = crate::cassettes::cassette_path("anthropic", scenario);
    let contents = std::fs::read_to_string(&path)
        .unwrap_or_else(|err| panic!("cassette {} should be readable: {err}", path.display()));

    let read = |marker: &str| -> u64 {
        let frame = contents
            .lines()
            .find(|line| line.contains(marker))
            .unwrap_or_else(|| {
                panic!("cassette {} should record a {marker} frame", path.display())
            });
        // The leading quote keeps this off the `cache_*_input_tokens` siblings.
        let (_, after) = frame.split_once(r#""input_tokens":"#).unwrap_or_else(|| {
            panic!(
                "{marker} in cassette {} should report input_tokens",
                path.display()
            )
        });
        after
            .chars()
            .take_while(char::is_ascii_digit)
            .collect::<String>()
            .parse()
            .unwrap_or_else(|err| {
                panic!(
                    "{marker} input_tokens in cassette {} should be a number: {err}",
                    path.display()
                )
            })
    };

    (
        read(r#""type":"message_start""#),
        read(r#""type":"message_delta""#),
    )
}

/// Kept in sync by hand with the literal at the call site above; a mismatch
/// fails loudly on the read below rather than silently skipping the guard.
const GATEWAY_METADATA_SCENARIO: &str = "streaming/gateway_message_delta_metadata";

/// The `input_tokens` a cassette's `message_start` frame records.
///
/// Read from the fixture on disk rather than hard-coded, so the guard tracks
/// whatever a re-record actually captured.
fn recorded_message_start_input_tokens(scenario: &str) -> u64 {
    let path = crate::cassettes::cassette_path("anthropic", scenario);
    let contents = std::fs::read_to_string(&path)
        .unwrap_or_else(|err| panic!("cassette {} should be readable: {err}", path.display()));

    let frame = contents
        .lines()
        .find(|line| line.contains(r#""type":"message_start""#))
        .unwrap_or_else(|| {
            panic!(
                "cassette {} should record a message_start frame",
                path.display()
            )
        });

    // The leading quote keeps this off `cache_creation_input_tokens` and
    // `cache_read_input_tokens`, which share the suffix.
    let (_, after) = frame.split_once(r#""input_tokens":"#).unwrap_or_else(|| {
        panic!(
            "message_start in cassette {} should report input_tokens",
            path.display()
        )
    });
    let digits: String = after.chars().take_while(char::is_ascii_digit).collect();

    digits.parse().unwrap_or_else(|err| {
        panic!(
            "message_start input_tokens in cassette {} should be a number: {err}",
            path.display()
        )
    })
}
