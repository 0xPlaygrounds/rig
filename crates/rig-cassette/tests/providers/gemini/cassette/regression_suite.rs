//! Live regression cassettes for shipped Gemini fixes whose premise was
//! previously unrecorded.
//!
//! See `many_rigs/rig-regression-cassette-suite-proposal.md` for the catalogue.

use rig::agent::OutputMode;
use rig::providers::gemini;
use rig_agent::test_utils::decode_structured_output;

use super::super::support::assert_recorded_sampling_fields;
use crate::support::{
    STREAMING_PREAMBLE, STREAMING_PROMPT, STRUCTURED_OUTPUT_PROMPT, SmokeStructuredOutput,
    assert_smoke_structured_output, collect_stream_final_response_and_provider_final,
};

/// rig#2322 — Regression: a native structured-output turn that sets **no**
/// `max_tokens` must not acquire one.
///
/// This is the reported bug's primary path. `create_request_body`'s
/// `output_schema` arm seeded its config with `GenerationConfig::default()`,
/// which hardcoded `temperature: Some(1.0)` and `max_output_tokens: Some(4096)`.
/// Any structured-output call was therefore capped at 4096 output tokens and
/// pinned to temperature 1.0 regardless of the caller's budget — silently, since
/// neither field appears anywhere in the caller's code.
///
/// The #2283 fix that introduced the sibling arm below it deliberately avoided
/// `Default::default()` for exactly this reason, but did not fix this arm; the
/// hazard is now removed at the root by making the `Default` all-`None`.
///
/// **The recorded request body is the assertion** — an injected
/// `maxOutputTokens` reappears in the cassette and fails the check below.
#[tokio::test]
async fn structured_output_without_max_tokens_sends_no_sampling_fields() {
    super::super::support::with_gemini_cassette(
        "regression/structured_output_without_max_tokens",
        |client| async move {
            let agent = rig::AgentBuilder::new(
                client.completion(gemini::completion::GEMINI_3_FLASH_PREVIEW),
            )
            .output_schema::<SmokeStructuredOutput>()
            .output_mode(OutputMode::Native)
            // Deliberately no `.max_tokens(...)` and no `.temperature(...)`:
            // the caller is relying on the model's own output limit.
            .build();

            let response = agent
                .prompt(STRUCTURED_OUTPUT_PROMPT)
                .await
                .expect("structured output prompt should succeed");
            let structured: SmokeStructuredOutput = decode_structured_output(
                "gemini_regression_structured_output_without_max_tokens",
                &response.output(),
            )
            .expect("structured output should deserialize");

            assert_smoke_structured_output(&structured);
        },
    )
    .await;

    assert_recorded_sampling_fields("regression/structured_output_without_max_tokens", &[]);
}

/// rig#2322 — Regression: a native structured-output turn that *does* set
/// `max_tokens` sends the caller's value and still acquires no `temperature`.
///
/// The complement of the test above: the previous code overwrote the injected
/// 4096 with the caller's value at the `max_tokens` arm, so an explicit budget
/// masked the defect. Pinning this direction keeps a future fix from
/// "resolving" the bug by dropping the caller's value instead.
#[tokio::test]
async fn structured_output_with_max_tokens_sends_only_the_caller_value() {
    super::super::support::with_gemini_cassette(
        "regression/structured_output_with_max_tokens",
        |client| async move {
            let agent = rig::AgentBuilder::new(
                client.completion(gemini::completion::GEMINI_3_FLASH_PREVIEW),
            )
            .output_schema::<SmokeStructuredOutput>()
            .output_mode(OutputMode::Native)
            .max_tokens(16_384)
            .build();

            let response = agent
                .prompt(STRUCTURED_OUTPUT_PROMPT)
                .await
                .expect("structured output prompt should succeed");
            let structured: SmokeStructuredOutput = decode_structured_output(
                "gemini_regression_structured_output_with_max_tokens",
                &response.output(),
            )
            .expect("structured output should deserialize");

            assert_smoke_structured_output(&structured);
        },
    )
    .await;

    assert_recorded_sampling_fields(
        "regression/structured_output_with_max_tokens",
        &[("maxOutputTokens", serde_json::json!(16_384))],
    );
}

/// rig#2322 — Regression: a caller who supplies an `additional_params`
/// `generationConfig` for `thinkingConfig` gets *that* on the wire and nothing
/// else.
///
/// This is the usage pattern that hid the original #2283 defect: every
/// pre-existing Gemini cassette passed a `GenerationConfig` for thinking, which
/// made the `Option` `Some` and masked the dropped `max_tokens`. It is also the
/// pattern most exposed to a reintroduced non-`None` `Default`, because callers
/// build these configs with `..Default::default()` — so a value restored to the
/// `Default` would silently ride along with every thinking request.
#[tokio::test]
async fn thinking_config_without_max_tokens_sends_no_sampling_fields() {
    super::super::support::with_gemini_cassette(
        "regression/thinking_config_without_max_tokens",
        |client| async move {
            let params = serde_json::json!({
                "generationConfig": {
                    "thinkingConfig": {
                        "thinkingLevel": "low",
                        "includeThoughts": true
                    }
                }
            });

            let agent = rig::AgentBuilder::new(
                client.completion(gemini::completion::GEMINI_3_FLASH_PREVIEW),
            )
            .preamble(STREAMING_PREAMBLE)
            .additional_params(params)
            // Again no `.max_tokens(...)`: the thinking budget is the only
            // generation setting this caller asked for.
            .build();

            agent
                .prompt(STREAMING_PROMPT)
                .await
                .expect("thinking-config prompt should succeed");
        },
    )
    .await;

    assert_recorded_sampling_fields("regression/thinking_config_without_max_tokens", &[]);

    // The point of the scenario: thinkingConfig survived, so the assertion
    // above is proving absence in a request that *did* carry a generationConfig
    // — not one that omitted the object entirely.
    let recorded = std::fs::read_to_string(crate::cassettes::cassette_path(
        "gemini",
        "regression/thinking_config_without_max_tokens",
    ))
    .expect("cassette should be readable");
    assert!(
        recorded.contains("thinkingConfig"),
        "the caller's thinkingConfig must still reach Gemini"
    );
}

/// rig#2322 — Regression: the streaming surface gets the same request-boundary
/// guarantee as the blocking one for native structured output.
///
/// `create_request_body` is shared, so this cannot diverge by construction
/// today — but the streaming path is the one that truncated silently (a
/// content-less `MAX_TOKENS` turn used to finalize as a successful empty
/// answer), so it is pinned explicitly rather than left implied.
#[tokio::test]
async fn streaming_structured_output_without_max_tokens_sends_no_sampling_fields() {
    super::super::support::with_gemini_cassette(
        "regression/streaming_structured_output_without_max_tokens",
        |client| async move {
            let agent = rig::AgentBuilder::new(
                client.completion(gemini::completion::GEMINI_3_FLASH_PREVIEW),
            )
            .output_schema::<SmokeStructuredOutput>()
            .output_mode(OutputMode::Native)
            .build();

            let mut stream = agent.prompt(STRUCTURED_OUTPUT_PROMPT).stream();
            let (_response, provider_final) =
                collect_stream_final_response_and_provider_final(&mut stream)
                    .await
                    .expect("streaming structured output should succeed");

            // The turn completed on its own rather than being cut short — the
            // condition that, when violated with no content, must now error.
            assert_ne!(
                provider_final.finish_reason,
                Some(rig::completion::FinishReason::Length),
                "an unbudgeted structured-output turn should not be hitting the \
                 output-token limit"
            );
        },
    )
    .await;

    assert_recorded_sampling_fields(
        "regression/streaming_structured_output_without_max_tokens",
        &[],
    );
}
