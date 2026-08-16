//! Tool-call round trips on Z.AI's general endpoint, blocking and streaming.
//!
//! The pair is a parity check as much as a smoke test. `ZAiExt` leaves
//! `EMITS_COMPLETE_SINGLE_CHUNK_TOOL_CALLS` at its `false` default, so the
//! shared accumulator holds a streamed tool call until the stream ends — while
//! Z.AI documents `tool_stream` as defaulting to *false*, i.e. tool arguments
//! are buffered and validated before they are sent. If the recording shows one
//! SSE frame carrying id, name and complete arguments together, the default is
//! wrong for Z.AI and a streamed tool call arrives strictly later than its
//! blocking twin. The docs never print the frames, so only the fixture decides.

use rig::completion::Prompt;
use rig::prelude::*;
use rig::streaming::StreamingPrompt;

use super::super::CHEAP_GENERAL_MODEL;
use super::super::support::{recorded_response_text, with_zai_general_cassette};
use crate::support::{
    Adder, STREAMING_TOOLS_PREAMBLE, STREAMING_TOOLS_PROMPT, Subtract, TOOLS_PREAMBLE,
    TOOLS_PROMPT, assert_mentions_expected_number, collect_stream_final_response,
};

#[tokio::test]
#[ignore = "unrecorded: requires ZAI_API_KEY to record the cassette"]
async fn general_tool_call_roundtrip_blocking() {
    with_zai_general_cassette(
        "general/tool_call_roundtrip_blocking",
        |client| async move {
            let agent = client
                .agent(CHEAP_GENERAL_MODEL)
                .preamble(TOOLS_PREAMBLE)
                .tool(Adder)
                .tool(Subtract)
                .default_max_turns(2)
                .build();

            let response = agent
                .prompt(TOOLS_PROMPT)
                .await
                .expect("Z.AI tool prompt should succeed");

            assert_mentions_expected_number(&response, -3);
        },
    )
    .await;
}

#[tokio::test]
#[ignore = "unrecorded: requires ZAI_API_KEY to record the cassette"]
async fn general_tool_call_roundtrip_streaming() {
    with_zai_general_cassette(
        "general/tool_call_roundtrip_streaming",
        |client| async move {
            let agent = client
                .agent(CHEAP_GENERAL_MODEL)
                .preamble(STREAMING_TOOLS_PREAMBLE)
                .tool(Adder)
                .tool(Subtract)
                .default_max_turns(2)
                .build();

            let mut stream = agent.stream_prompt(STREAMING_TOOLS_PROMPT).await;
            let response = collect_stream_final_response(&mut stream)
                .await
                .expect("Z.AI streaming tool prompt should succeed");

            assert_mentions_expected_number(&response, -3);
        },
    )
    .await;

    // The cell's premise: it only covers tool-call *streaming* if Z.AI actually
    // streamed a tool call. Assert that from turn 1's recorded frames rather
    // than inferring it from the answer, which a model could reach without
    // calling anything. (`recorded_response_text` stops at the document
    // separator, so turn 2's request body — which also carries `tool_calls` —
    // is outside the window.)
    let frames = recorded_response_text("general/tool_call_roundtrip_streaming");
    assert!(
        frames.contains("tool_calls"),
        "the streaming tool cell must have recorded tool-call deltas"
    );
}
