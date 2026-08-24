//! Live regression cassettes for shipped Gemini fixes whose premise was
//! previously unrecorded.
//!
//! See `many_rigs/rig-regression-cassette-suite-proposal.md` for the catalogue.

use rig::agent::OutputMode;
use rig::completion::Prompt;
use rig::prelude::*;
use rig::providers::gemini;
use rig::providers::gemini::completion::gemini_api_types::{
    AdditionalParameters, GenerationConfig, ThinkingConfig, ThinkingLevel,
};
use rig::streaming::StreamingPrompt;
use rig_agent::test_utils::decode_structured_output;

use super::super::support::assert_recorded_sampling_fields;
use crate::support::{
    STREAMING_PREAMBLE, STREAMING_PROMPT, STRUCTURED_OUTPUT_PROMPT, SmokeStructuredOutput,
    assert_smoke_structured_output, collect_stream_final_response_and_provider_final,
};

/// C15 — Regression: agent-level `max_tokens` reaches Gemini's
/// `generationConfig.maxOutputTokens` when the caller supplies **no**
/// `additional_params`.
///
/// Found by recording this suite. `create_request_body` applied `temperature`
/// and `max_tokens` through `generation_config.map(...)`, and `Option::map` is a
/// no-op on `None` — so a request that set either field without *also* passing an
/// `additional_params.generationConfig` dropped both silently. `.max_tokens(8)`
/// on a Gemini agent never reached the wire and the model ran to its own limit.
///
/// It survived because every other Gemini cassette in the tree passes an
/// `additional_params` `GenerationConfig` (for `thinkingConfig`), which makes the
/// Option `Some` and the `map` fire. The whole corpus shared one usage pattern
/// that dodges the defect; nothing exercised the plain builder path.
///
/// **The recorded request body is the assertion.** Per `tests/README.md`'s
/// "assert on the request boundary too": the cassette matches the outbound body,
/// so if the plumbing regresses, `maxOutputTokens` disappears from the request,
/// the mock misses, and this test fails — no in-test assertion needed for it.
/// The `finish_reason` check below is secondary.
#[tokio::test]
async fn agent_max_tokens_reaches_generation_config_without_additional_params() {
    super::super::support::with_gemini_cassette(
        "regression/agent_max_tokens_without_additional_params",
        |client| async move {
            let agent = client
                .agent(gemini::completion::GEMINI_3_FLASH_PREVIEW)
                .preamble(STREAMING_PREAMBLE)
                // Deliberately no `.additional_params(...)` — that is the path
                // the bug lived on and the one no other cassette covers.
                .max_tokens(512)
                .build();

            let mut stream = agent.stream_prompt(STREAMING_PROMPT).await;
            let (_response, provider_final): (_, rig::streaming::StreamFinal) =
                collect_stream_final_response_and_provider_final(&mut stream)
                    .await
                    .expect("streaming prompt should succeed");

            assert_eq!(provider_final.provider, "gcp.gemini");
        },
    )
    .await;

    // Read the recorded request back and assert the field is really there, so
    // the guarantee is stated in the test and not only implied by mock matching
    // (a future harness change that relaxed body matching would otherwise make
    // this test silently stop covering the bug).
    //
    // The `temperature` half of this assertion is the rig#2322 direction: an
    // unset field must stay off the wire, so setting one sampling field does not
    // silently acquire the other.
    assert_recorded_sampling_fields(
        "regression/agent_max_tokens_without_additional_params",
        &[("maxOutputTokens", serde_json::json!(512))],
    );
}

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
            let agent = client
                .agent(gemini::completion::GEMINI_3_FLASH_PREVIEW)
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
                &response,
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
            let agent = client
                .agent(gemini::completion::GEMINI_3_FLASH_PREVIEW)
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
                &response,
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

/// rig#2322 — Regression: the mirror of C15. Setting `temperature` alone must
/// not acquire a `maxOutputTokens`.
///
/// C15 pins `max_tokens` without `temperature`; this pins `temperature` without
/// `max_tokens`. Together they cover both halves of the shared seed, so a
/// reintroduced non-`None` default is caught whichever field it lands on — the
/// `maxOutputTokens` half being the one that truncated real workloads.
#[tokio::test]
async fn temperature_without_max_tokens_sends_no_max_output_tokens() {
    super::super::support::with_gemini_cassette(
        "regression/temperature_without_max_tokens",
        |client| async move {
            let agent = client
                .agent(gemini::completion::GEMINI_3_FLASH_PREVIEW)
                .preamble(STREAMING_PREAMBLE)
                .temperature(0.0)
                .build();

            agent
                .prompt(STREAMING_PROMPT)
                .await
                .expect("prompt should succeed");
        },
    )
    .await;

    assert_recorded_sampling_fields(
        "regression/temperature_without_max_tokens",
        &[("temperature", serde_json::json!(0.0))],
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
            let config = GenerationConfig {
                thinking_config: Some(ThinkingConfig {
                    thinking_budget: None,
                    thinking_level: Some(ThinkingLevel::Low),
                    include_thoughts: Some(true),
                }),
                ..Default::default()
            };
            let params = AdditionalParameters::default().with_config(config);

            let agent = client
                .agent(gemini::completion::GEMINI_3_FLASH_PREVIEW)
                .preamble(STREAMING_PREAMBLE)
                .additional_params(serde_json::to_value(params).expect("params should serialize"))
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
            let agent = client
                .agent(gemini::completion::GEMINI_3_FLASH_PREVIEW)
                .output_schema::<SmokeStructuredOutput>()
                .output_mode(OutputMode::Native)
                .build();

            let mut stream = agent.stream_prompt(STRUCTURED_OUTPUT_PROMPT).await;
            let (_response, provider_final): (_, rig::streaming::StreamFinal) =
                collect_stream_final_response_and_provider_final(&mut stream)
                    .await
                    .expect("streaming structured output should succeed");

            assert_eq!(provider_final.provider, "gcp.gemini");
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
