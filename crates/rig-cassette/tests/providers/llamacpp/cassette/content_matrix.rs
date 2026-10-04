//! Message-content shapes, and what the chat template does with each.
//!
//! **Server**: the default configuration — `unsloth/Qwen3-1.7B-GGUF` Q4_K_M,
//! `--jinja --seed 42 --temp 0 -c 4096`, `llama-server` b10964-b29c606e2.
//! Every claim here is about the request rig builds and the template's
//! tolerance for it, not about whether the model is clever, so the smoke tier
//! is the right tier for all of them.
//!
//! | Cell | Shape | Pinned |
//! | --- | --- | --- |
//! | [`an_answer_fully_consumed_by_a_stop_sequence_is_an_empty_turn`] | empty content | 200 on the wire, an empty successful turn |
//! | [`consecutive_user_messages_become_one`] | user, user | the history adapter makes adjacent user messages one |
//! | [`unicode_split_across_stream_chunks_reassembles`] | emoji + CJK, streaming | multi-byte characters survive SSE chunk boundaries |
//! | [`a_very_long_tool_output_survives_the_round_trip`] | 8 KiB tool result | the payload reaches the model intact |
//! | [`a_system_message_plus_history_keeps_its_order`] | system + 3 turns | the system message stays first and the turns keep their order |
//!
//! # Why the empty-answer cell exists
//!
//! A `stop` sequence that matches the model's very first token leaves
//! `content: ""` with `finish_reason: "stop"` and a perfectly healthy 200.
//! Rig keeps that as an empty turn, as pi does: the core fold judges
//! emptiness once for every wire, and the fixture holds the 200 beside it.

use rig::message::{AssistantContent, CallId, Message, ToolResult, ToolResultContent, UserContent};
use serde_json::{Value, json};

use crate::cassettes::{
    recorded_json_request, recorded_sse_json_frames, recorded_statuses_and_bodies,
};
use crate::support::{assistant_text_response, collect_stream_final_response};

use super::super::cassette_support::*;
use rig::completion::CompletionRequest;

const NO_THINK: &str = "/no_think ";

/// A turn whose whole answer is eaten by a stop sequence is an empty turn:
/// llama.cpp answers `200` with `finish_reason: "stop"` and `content: ""`,
/// and rig keeps it as pi does, a success with no blocks, decided once by
/// the core fold for whole and streamed replies alike.
#[tokio::test]
async fn an_answer_fully_consumed_by_a_stop_sequence_is_an_empty_turn() {
    with_llamacpp_cassette(
        "content_matrix/empty_answer_with_stop",
        |client| async move {
            let model = client.completion(CASSETTE_MODEL);
            let response = model
                .call(
                    CompletionRequest::new("Reply with exactly this and nothing else: STOPWORD")
                        .max_tokens(64)
                        // Qwen3 opens every turn with a `<think>` block, so
                        // this matches the model's very first emitted token
                        // and the whole answer is consumed before a character
                        // of it exists.
                        .additional_params(json!({ "stop": ["<think>"] })),
                )
                .await
                .expect("an empty turn that ended cleanly decodes");
            assert!(response.choice.is_empty(), "{:?}", response.choice);
            assert!(!response.stop().is_failure(), "{:?}", response.stop());
        },
    )
    .await;

    let recorded =
        recorded_statuses_and_bodies("llamacpp", "content_matrix/empty_answer_with_stop");
    let (status, body) = &recorded[0];
    assert_eq!(
        *status, 200,
        "the server did not fail; the emptiness is the whole story"
    );
    let response: Value = serde_json::from_str(body).expect("response should be JSON");
    let content = response["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or_default();
    assert!(
        content.is_empty(),
        "the recorded content must itself be empty for this cell to mean anything: {content:?}"
    );
    assert_eq!(
        response["choices"][0]["finish_reason"],
        json!("stop"),
        "a stop sequence, not a length cut"
    );
}

/// Two consecutive `user` messages and the prompt go out as one message.
///
/// Some providers reject alternation violations, so the history adapter
/// makes adjacent user messages one on every wire. The cell reads the
/// recorded request so the merge stays visible.
#[tokio::test]
async fn consecutive_user_messages_become_one() {
    with_llamacpp_cassette(
        "content_matrix/consecutive_same_role",
        |client| async move {
            let model = client.completion(CASSETTE_MODEL);
            let response = model
                .call(
                    CompletionRequest::new(format!("{NO_THINK}What was the second word I said?"))
                        .messages(vec![
                            Message::User {
                                content: vec![UserContent::text("First word: heliotrope.")],
                            },
                            Message::User {
                                content: vec![UserContent::text("Second word: quicksilver.")],
                            },
                        ])
                        .max_tokens(256),
                )
                .await
                .expect("consecutive same-role messages are accepted");
            assert!(!response.choice.is_empty());
        },
    )
    .await;

    let request = recorded_json_request("llamacpp", "content_matrix/consecutive_same_role");
    let roles = request["messages"]
        .as_array()
        .expect("messages")
        .iter()
        .map(|message| message["role"].as_str().unwrap_or_default().to_string())
        .collect::<Vec<_>>();
    assert_eq!(
        roles,
        vec!["user".to_string()],
        "the two history turns and the prompt arrive as one user message: {roles:?}"
    );
}

/// Multi-byte characters survive SSE chunk boundaries.
///
/// llama.cpp streams per token and a UTF-8 sequence can straddle two of them,
/// so a naive per-chunk `String::from_utf8` would either panic or emit
/// replacement characters. The prompt asks for an emoji and CJK text
/// specifically because both are multi-byte in ways ASCII is not.
#[tokio::test]
async fn unicode_split_across_stream_chunks_reassembles() {
    with_llamacpp_cassette(
        "content_matrix/unicode_across_chunks",
        |client| async move {
            let agent = rig::AgentBuilder::new(client.completion(CASSETTE_MODEL))
                .preamble(
                    "Reply with exactly the text you are asked for and nothing else. \
                 No explanation, no quotes.",
                )
                .max_tokens(256)
                .build();

            let mut stream = agent
                .prompt(format!(
                    "{NO_THINK}Write exactly this line and nothing else: \
                 🌍こんにちは世界🎉안녕하세요🚀Здравствуйте🌸"
                ))
                .stream();
            let answer = collect_stream_final_response(&mut stream)
                .await
                .expect("a unicode stream should complete");

            assert!(
                answer.contains('🌍') && answer.contains('🎉'),
                "the astral-plane emoji must survive chunking: {answer:?}"
            );
            assert!(
                answer.contains("こんにちは"),
                "the CJK run must survive chunking: {answer:?}"
            );
            assert!(
                !answer.contains('\u{FFFD}'),
                "no replacement characters: {answer:?}"
            );
        },
    )
    .await;

    // The premise: the recorded frames really did split the text, so this is
    // not a single-chunk answer that would have passed trivially.
    let frames = recorded_sse_json_frames("llamacpp", "content_matrix/unicode_across_chunks");
    let pieces = frames
        .iter()
        .filter_map(|frame| frame["choices"][0]["delta"]["content"].as_str())
        .filter(|piece| !piece.is_empty())
        .count();
    assert!(
        pieces >= 8,
        "the recorded stream must be genuinely chunked for this cell to test \
         anything, saw {pieces} content deltas"
    );
    // And the deltas really are carrying the multi-byte runs, so the cell
    // would not pass on an ASCII answer.
    //
    // What this cannot check, and does not claim to: that a UTF-8 *sequence*
    // was split across two frames. llama.cpp buffers until a character is
    // whole before emitting a delta, and every recorded delta is therefore
    // already valid UTF-8 — so the split case is not reachable through this
    // wire at all. The reassembly claim above is about rig concatenating
    // correct fragments in order, which is what is actually at risk.
    let multibyte_deltas = frames
        .iter()
        .filter_map(|frame| frame["choices"][0]["delta"]["content"].as_str())
        .filter(|piece| piece.chars().any(|ch| ch.len_utf8() > 1))
        .count();
    assert!(
        multibyte_deltas >= 3,
        "only {multibyte_deltas} deltas carried a multi-byte character; the cell \
         would be passing on mostly-ASCII output"
    );
}

/// An 8 KiB tool result reaches the model intact.
#[tokio::test]
async fn a_very_long_tool_output_survives_the_round_trip() {
    // Long, and with a distinctive needle at the very end so a truncation
    // anywhere in the payload is detectable from the answer alone.
    const NEEDLE: &str = "ZANZIBAR-9317";
    let long_output = format!(
        "{}END OF DUMP. The code is {NEEDLE}.",
        "lorem ipsum dolor sit amet. ".repeat(290)
    );

    with_llamacpp_cassette(
        "content_matrix/long_tool_output",
        move |client| async move {
            let model = client.completion(CASSETTE_MODEL);
            let response = model
                .call(
                    CompletionRequest::new(Message::User {
                        content: vec![UserContent::ToolResult(ToolResult {
                            is_error: false,
                            call: CallId::from_wire("call_long"),
                            name: rig_core::message::ToolName::new("dump".to_string())
                                .expect("tool name"),
                            content: vec![ToolResultContent::text(long_output)],
                        })],
                    })
                    .preamble(
                        "The tool result ends with a code. Reply with only that code.".to_string(),
                    )
                    .messages(vec![
                        Message::User {
                            content: vec![UserContent::text("What code does the dump end with?")],
                        },
                        Message::Assistant(rig_core::message::AssistantMessage::new(vec![
                            AssistantContent::tool_call(
                                "call_long",
                                rig_core::message::ToolName::new("dump").expect("tool name"),
                                json!({}),
                            ),
                        ])),
                    ])
                    .max_tokens(256),
                )
                .await
                .expect("a long tool result should be accepted");

            let text = assistant_text_response(&response.choice).unwrap_or_default();
            assert!(
                text.contains(NEEDLE),
                "the needle at the end of an 8 KiB payload must reach the model: {text:?}"
            );
        },
    )
    .await;

    let request = recorded_json_request("llamacpp", "content_matrix/long_tool_output");
    let serialized = request["messages"].to_string();
    assert!(
        serialized.len() > 8_000,
        "the payload must really be large for this cell to test anything, was {}",
        serialized.len()
    );
    assert!(
        serialized.contains(NEEDLE),
        "the tail of the payload must reach the wire uncut"
    );
}

/// A system message plus a multi-turn history keeps its order.
#[tokio::test]
async fn a_system_message_plus_history_keeps_its_order() {
    with_llamacpp_cassette("content_matrix/system_plus_history", |client| async move {
        let model = client.completion(CASSETTE_MODEL);
        let response = model
            .call(
                CompletionRequest::new(format!("{NO_THINK}And what was the first one?"))
                    .preamble(
                        "You are a ledger. Answer with the requested codeword only.".to_string(),
                    )
                    .messages(vec![
                        Message::User {
                            content: vec![UserContent::text("Codeword one is heliotrope.")],
                        },
                        Message::Assistant(rig_core::message::AssistantMessage::new(vec![
                            AssistantContent::text("Noted."),
                        ])),
                        Message::User {
                            content: vec![UserContent::text("Codeword two is quicksilver.")],
                        },
                    ])
                    .max_tokens(256),
            )
            .await
            .expect("a system message plus history should be accepted");

        let text = assistant_text_response(&response.choice).unwrap_or_default();
        assert!(
            text.to_ascii_lowercase().contains("heliotrope"),
            "the model must be able to reach the first turn: {text:?}"
        );
    })
    .await;

    let request = recorded_json_request("llamacpp", "content_matrix/system_plus_history");
    let roles = request["messages"]
        .as_array()
        .expect("messages")
        .iter()
        .map(|message| message["role"].as_str().unwrap_or_default().to_string())
        .collect::<Vec<_>>();
    assert_eq!(
        roles,
        vec![
            "system".to_string(),
            "user".to_string(),
            "assistant".to_string(),
            "user".to_string(),
        ],
        "the preamble leads, the history keeps its order, and its last user turn \
         and the prompt are one message: {roles:?}"
    );
}
