//! Migrated from `examples/openai_agent_completions_api.rs`.

use rig::client::{CompletionClient, ProviderClient};
use rig::completion::Prompt;
use rig::providers::openai;
use rig::streaming::StreamingPrompt;

use crate::support::{
    ALPHA_SIGNAL_OUTPUT, AlphaSignal, BETA_SIGNAL_OUTPUT, BetaSignal, TWO_TOOL_STREAM_PREAMBLE,
    TWO_TOOL_STREAM_PROMPT, assert_nonempty_response, assert_two_tool_roundtrip_contract,
    collect_stream_observation,
};

#[tokio::test]
#[ignore = "requires OPENAI_API_KEY"]
async fn completions_api_agent_prompt() {
    let agent = openai::Client::from_env()
        .completion_model(openai::GPT_4O)
        .completions_api()
        .into_agent_builder()
        .preamble("You are a helpful assistant.")
        .build();

    let response = agent
        .prompt("Hello world!")
        .await
        .expect("completions api prompt should succeed");

    assert_nonempty_response(&response);
}

#[tokio::test]
#[ignore = "requires OPENAI_API_KEY"]
async fn completions_api_streams_two_tool_calls_before_final_answer() {
    let client = openai::Client::from_env().completions_api();
    let agent = client
        .agent(openai::GPT_4O)
        .preamble(TWO_TOOL_STREAM_PREAMBLE)
        .tool(AlphaSignal)
        .tool(BetaSignal)
        .build();

    let mut stream = agent
        .stream_prompt(TWO_TOOL_STREAM_PROMPT)
        .multi_turn(8)
        .await;
    let observation = collect_stream_observation(&mut stream).await;

    assert_two_tool_roundtrip_contract(
        &observation,
        &["lookup_harbor_label", "lookup_orchard_label"],
        &[ALPHA_SIGNAL_OUTPUT, BETA_SIGNAL_OUTPUT],
    );
}
