//! OpenAI streaming tools coverage, including the migrated example path.

use rig::client::{CompletionClient, ProviderClient};
use rig::providers::openai;
use rig::streaming::StreamingPrompt;

use crate::support::{
    ALPHA_SIGNAL_OUTPUT, Adder, AlphaSignal, ORDERED_TOOL_STREAM_PREAMBLE,
    ORDERED_TOOL_STREAM_PROMPT, STREAMING_TOOLS_PREAMBLE, STREAMING_TOOLS_PROMPT, Subtract,
    assert_mentions_expected_number, assert_tool_call_precedes_later_text,
    collect_stream_final_response, collect_stream_observation,
};

#[tokio::test]
#[ignore = "requires OPENAI_API_KEY"]
async fn streaming_tools_smoke() {
    let client = openai::Client::from_env();
    let agent = client
        .agent(openai::GPT_4O)
        .preamble(STREAMING_TOOLS_PREAMBLE)
        .tool(Adder)
        .tool(Subtract)
        .build();

    let mut stream = agent.stream_prompt(STREAMING_TOOLS_PROMPT).await;
    let response = collect_stream_final_response(&mut stream)
        .await
        .expect("streaming tool prompt should succeed");

    assert_mentions_expected_number(&response, -3);
}

#[tokio::test]
#[ignore = "requires OPENAI_API_KEY"]
async fn example_streaming_with_tools() {
    let client = openai::Client::from_env();
    let agent = client
        .agent(openai::GPT_4O)
        .preamble(
            "You are a calculator here to help the user perform arithmetic operations. \
             Use the tools provided to answer the user's question and answer in a full sentence.",
        )
        .max_tokens(1024)
        .tool(Adder)
        .tool(Subtract)
        .build();

    let mut stream = agent.stream_prompt("Calculate 2 - 5").await;
    let response = collect_stream_final_response(&mut stream)
        .await
        .expect("streaming tools prompt should succeed");

    assert_mentions_expected_number(&response, -3);
}

#[tokio::test]
#[ignore = "requires OPENAI_API_KEY"]
async fn responses_stream_preserves_tool_result_flow() {
    let client = openai::Client::from_env();
    let agent = client
        .agent(openai::GPT_4O)
        .preamble(ORDERED_TOOL_STREAM_PREAMBLE)
        .tool(AlphaSignal)
        .build();

    let mut stream = agent
        .stream_prompt(ORDERED_TOOL_STREAM_PROMPT)
        .multi_turn(5)
        .await;
    let observation = collect_stream_observation(&mut stream).await;

    assert_tool_call_precedes_later_text(
        &observation,
        "lookup_harbor_label",
        &[ALPHA_SIGNAL_OUTPUT],
    );
}
