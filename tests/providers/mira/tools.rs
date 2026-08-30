//! Mira tools smoke test.

use rig::completion::Prompt;
use rig::prelude::*;
use rig::providers::mira;

use crate::support::{
    Adder, Subtract, TOOLS_PREAMBLE, TOOLS_PROMPT, assert_mentions_expected_number,
};

/// Mira routes this id to Anthropic's model. Spelled out rather than borrowed
/// from `anthropic::completion::CLAUDE_SONNET_4_6` so this target does not
/// need the `anthropic` feature; `model_id_tracks_anthropic` below pins the
/// two together wherever that feature is on, so a model bump cannot leave
/// this test quietly requesting a retired id.
const MODEL: &str = "claude-sonnet-4-6";

#[cfg(feature = "anthropic")]
#[test]
fn model_id_tracks_anthropic() {
    assert_eq!(
        MODEL,
        rig::providers::anthropic::completion::CLAUDE_SONNET_4_6
    );
}

#[tokio::test]
#[ignore = "requires MIRA_API_KEY"]
async fn tools_smoke() {
    let client = mira::Client::from_env().expect("client should build");
    let agent = client
        .agent(MODEL)
        .preamble(TOOLS_PREAMBLE)
        .tool(Adder)
        .tool(Subtract)
        .build();

    let response = agent
        .prompt(TOOLS_PROMPT)
        .await
        .expect("tool prompt should succeed");

    assert_mentions_expected_number(&response, -3);
}
