//! Streaming coverage for Z.AI's general endpoint.
//!
//! Two things are unproven for Z.AI here and only a recording settles them.
//! First, rig appends `stream_options: {include_usage: true}` to every
//! streaming request (`STREAM_INCLUDE_USAGE` defaults to true) while Z.AI
//! documents no `stream_options` parameter at all and already puts `usage` on
//! the same terminal frame as `finish_reason`; whether Z.AI tolerates the
//! unknown key or rejects it is a wire fact. Second, GLM is a CJK-heavy model,
//! so a multi-byte character split across SSE frames is a realistic input, not
//! a contrived one.

use rig::prelude::*;
use rig::streaming::StreamingPrompt;

use super::super::support::{
    recorded_request_body, recorded_response_text, with_zai_general_cassette,
};
use super::super::{CHEAP_GENERAL_MODEL, THINKING_MODEL};
use crate::support::{
    STREAMING_PREAMBLE, STREAMING_PROMPT, assert_nonempty_response,
    collect_stream_final_response_and_provider_final,
};

/// A prompt whose answer is CJK plus an emoji, so the merge that assembles the
/// streamed text has to carry multi-byte characters through.
///
/// Scope note: this cell cannot pin the *byte-split* hazard. The replay server
/// fragments an SSE body only at `char` boundaries
/// (`fragmented_sse_body_chunks` in `tests/common/cassettes.rs`), and the
/// fixture stores the body as one string rather than the frames Z.AI actually
/// emitted, so a cassette cannot reproduce a character split across chunks in
/// either mode. What it does pin is that non-ASCII content survives the
/// provider mapping and the multi-chunk merge intact.
const CJK_PROMPT: &str = "用一句中文回答:什么是内存安全?最后加一个 🦀 表情。";

#[tokio::test]
#[ignore = "unrecorded: requires ZAI_API_KEY to record the cassette"]
async fn general_completion_streaming_smoke() {
    with_zai_general_cassette("general/completion_streaming_smoke", |client| async move {
        let agent = client
            .agent(CHEAP_GENERAL_MODEL)
            .preamble(STREAMING_PREAMBLE)
            .build();

        let mut stream = agent.stream_prompt(STREAMING_PROMPT).await;
        let (response, provider_final) =
            collect_stream_final_response_and_provider_final(&mut stream)
                .await
                .expect("Z.AI streaming completion should succeed");

        assert_nonempty_response(&response);
        assert!(
            provider_final.usage.output_tokens > 0,
            "Z.AI reports usage on the terminal streaming frame; got {:?}",
            provider_final.usage
        );
        assert!(
            provider_final.finish_reason.is_some(),
            "the terminal record must carry the finish reason Z.AI sent"
        );
    })
    .await;

    // The premise this cell exists to expose: rig sends a parameter Z.AI does
    // not document. If the recording shows Z.AI accepting it *and* reporting
    // usage on the terminal frame regardless, the parameter is merely
    // unnecessary; if the recorded response is a 400, every streaming call on
    // Z.AI is broken and this assertion is where that shows up.
    let request = recorded_request_body("general/completion_streaming_smoke");
    assert_eq!(
        request["stream"], true,
        "the cell must have recorded a streaming request"
    );
    assert_eq!(
        request["stream_options"]["include_usage"], true,
        "rig still appends the undocumented stream_options block; request was {request}"
    );
}

#[tokio::test]
#[ignore = "unrecorded: requires ZAI_API_KEY to record the cassette"]
async fn general_thinking_streaming_and_cjk() {
    with_zai_general_cassette("general/thinking_streaming_and_cjk", |client| async move {
        let agent = client
            .agent(THINKING_MODEL)
            .preamble("你是一个简洁的助手。")
            .build();

        let mut stream = agent.stream_prompt(CJK_PROMPT).await;
        let (response, _) = collect_stream_final_response_and_provider_final(&mut stream)
            .await
            .expect("Z.AI CJK streaming completion should succeed");

        assert_nonempty_response(&response);
        // A mishandled multi-byte character surfaces as a replacement
        // character, never as an error.
        assert!(
            !response.contains('\u{fffd}'),
            "multi-byte characters must survive the frame boundaries: {response:?}"
        );
        assert!(
            response
                .chars()
                .any(|ch| ('\u{4e00}'..='\u{9fff}').contains(&ch)),
            "the CJK cell must actually have produced CJK output: {response:?}"
        );
    })
    .await;

    // Assert the cell's own premise from the fixture: a single-frame answer
    // would pass the assertions above without exercising the merge at all.
    let frames = recorded_response_text("general/thinking_streaming_and_cjk");
    assert!(
        frames.matches("data:").count() > 2,
        "the CJK cell needs a genuinely multi-frame stream to be about anything"
    );
}
