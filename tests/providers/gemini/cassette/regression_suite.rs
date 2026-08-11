//! Live regression cassettes for shipped Gemini fixes whose premise was
//! previously unrecorded.
//!
//! See `many_rigs/rig-regression-cassette-suite-proposal.md` for the catalogue.

use rig::prelude::*;
use rig::providers::gemini;
use rig::streaming::StreamingPrompt;

use crate::support::{
    STREAMING_PREAMBLE, STREAMING_PROMPT, collect_stream_final_response_and_provider_final,
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
    let recorded = std::fs::read_to_string(crate::cassettes::cassette_path(
        "gemini",
        "regression/agent_max_tokens_without_additional_params",
    ))
    .expect("cassette should be readable");

    assert!(
        recorded.contains(r#""maxOutputTokens":512"#),
        "the recorded request must carry the agent's max_tokens as \
         generationConfig.maxOutputTokens — without it the value was dropped \
         before it reached Gemini"
    );
    assert!(
        !recorded.contains(r#""temperature""#),
        "an unset temperature must stay off the wire: the config is seeded with \
         both fields cleared, not with GenerationConfig::default() (which is \
         temperature 1.0 / maxOutputTokens 4096), so setting one field does not \
         silently send the other"
    );
}
