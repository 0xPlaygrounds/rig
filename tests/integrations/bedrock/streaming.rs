//! AWS Bedrock streaming smoke tests inspired by the OpenAI and Anthropic provider tests.

use rig::client::CompletionClient;
use rig::completion::CompletionModel as _;
use rig::streaming::StreamingPrompt;
use rig::tool::Tool;

use super::{
    BEDROCK_COMPLETION_MODEL, client,
    support::{
        Adder, AlphaSignal, ORDERED_TOOL_STREAM_PREAMBLE, ORDERED_TOOL_STREAM_PROMPT,
        STREAMING_PREAMBLE, STREAMING_PROMPT, STREAMING_TOOLS_PREAMBLE, STREAMING_TOOLS_PROMPT,
        Subtract, assert_mentions_expected_number, assert_nonempty_response,
        assert_raw_stream_tool_call_precedes_text, collect_raw_stream_observation,
        collect_stream_final_response,
    },
};

#[tokio::test]
#[ignore = "requires AWS credentials and Bedrock model access"]
async fn streaming_smoke() {
    let agent = client()
        .agent(BEDROCK_COMPLETION_MODEL)
        .preamble(STREAMING_PREAMBLE)
        .build();

    let mut stream = agent.stream_prompt(STREAMING_PROMPT).await;
    let response = collect_stream_final_response(&mut stream)
        .await
        .expect("streaming prompt should succeed");

    assert_nonempty_response(&response);
}

#[tokio::test]
#[ignore = "requires AWS credentials and Bedrock model access"]
async fn streaming_tools_smoke() {
    let agent = client()
        .agent(BEDROCK_COMPLETION_MODEL)
        .preamble(STREAMING_TOOLS_PREAMBLE)
        .max_tokens(1024)
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
#[ignore = "requires AWS credentials and Bedrock model access"]
async fn raw_streaming_tool_call_smoke() {
    let model = client().completion_model(BEDROCK_COMPLETION_MODEL);
    let request = model
        .completion_request(ORDERED_TOOL_STREAM_PROMPT)
        .preamble(ORDERED_TOOL_STREAM_PREAMBLE.to_string())
        .tool(AlphaSignal.definition(String::new()).await)
        .build();

    let observation = collect_raw_stream_observation(
        model
            .stream(request)
            .await
            .expect("raw Bedrock stream should start"),
    )
    .await;

    assert_raw_stream_tool_call_precedes_text(&observation, "lookup_harbor_label");
}
