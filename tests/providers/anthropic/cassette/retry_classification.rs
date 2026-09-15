//! Retry classification for errors that arrive with no HTTP status.
//!
//! A provider error delivered *inside* a stream carries no status, so
//! `ProviderResponseError::is_retryable` used to fall through to
//! `transient.unwrap_or(false)` and report every one of them as permanent. The
//! runtime in `rig-ecs` retries only when `ErrorReport::retryable` is set, so a
//! mid-stream `overloaded_error` ended the run without spending any of the
//! retry budget the host configured. The classification now reads the
//! provider's own machine code.
//!
//! **What this cassette can and cannot prove.** An overload frame cannot be
//! induced on demand: a provider emits one when it is busy, and asking it to be
//! busy is not something a recording session can arrange. Neither key available
//! at recording time could produce one — the Anthropic workspace is over its
//! usage limit and answers HTTP 400 before any stream opens, and OpenRouter
//! relayed a healthy stream for every request shaped to provoke an upstream
//! failure (`max_tokens` far beyond the model's ceiling included). No committed
//! fixture in `tests/cassettes/` carries one either.
//!
//! So this recording pins the neighbouring real case — a streamed turn that
//! completes normally, proving the new classification leaves a healthy stream
//! untouched — and the error frame itself is covered by a hand-written body in
//! `provider_response::tests::a_body_borne_transient_code_is_retryable_without_a_status`
//! and end to end by
//! `rig-ecs/tests/run_provider_retry.rs::an_overload_reported_without_a_status_is_reissued`.
//! Those two fail without the fix; this one passes either way and is here to
//! catch a regression in the opposite direction.

use rig::prelude::*;

use super::super::support::with_anthropic_gateway_cassette;
use crate::support::{
    STREAMING_PROMPT, assert_nonempty_response, collect_stream_final_response_and_provider_final,
};

#[tokio::test]
async fn a_healthy_stream_is_unaffected_by_the_transient_code_classification() {
    with_anthropic_gateway_cassette(
        "retry_classification/healthy_stream_still_completes",
        |client| async move {
            let agent = client.agent("anthropic/claude-opus-5").build();

            let mut stream = agent.prompt(STREAMING_PROMPT).stream();
            let (response, provider_final): (_, rig::streaming::StreamFinal) =
                collect_stream_final_response_and_provider_final(&mut stream)
                    .await
                    .expect("a healthy stream should complete");

            assert_nonempty_response(&response);
            assert!(provider_final.usage.total_tokens > 0);
        },
    )
    .await;
}
