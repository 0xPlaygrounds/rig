//! Mistral reasoning roundtrip tests.
//!
//! The cross-provider reasoning contract (`crate::reasoning`), which every
//! reasoning-capable provider in the tree runs: a reasoning turn's trace must
//! reach the caller, and the turn must be replayable as history without the
//! provider rejecting it. Mistral had neither file, because it emitted no
//! reasoning at all — its trace arrives inside `content` as a `thinking` chunk
//! rather than beside it on `reasoning_content`, and rig joined only the text
//! parts of that array.

use rig::prelude::*;
use rig::providers::mistral;

use super::support::with_mistral_cassette;
use crate::reasoning::{self, ReasoningRoundtripAgent};

/// `mistral-small-latest` is the reasoning model — the live catalog lists
/// `magistral-small-latest` as one of its own aliases — and `reasoning_effort`
/// is how Mistral turns the trace on. Without it the same model answers with a
/// plain string and no trace at all.
fn reasoning_params() -> serde_json::Value {
    serde_json::json!({ "reasoning_effort": "high" })
}

#[tokio::test]
async fn streaming() {
    with_mistral_cassette("reasoning_roundtrip/streaming", |client| async move {
        reasoning::run_reasoning_roundtrip_streaming(ReasoningRoundtripAgent::new(
            client.completion_model(mistral::MISTRAL_SMALL),
            Some(reasoning_params()),
        ))
        .await;
    })
    .await;
}

#[tokio::test]
async fn nonstreaming() {
    with_mistral_cassette("reasoning_roundtrip/nonstreaming", |client| async move {
        reasoning::run_reasoning_roundtrip_nonstreaming(ReasoningRoundtripAgent::new(
            client.completion_model(mistral::MISTRAL_SMALL),
            Some(reasoning_params()),
        ))
        .await;
    })
    .await;
}
