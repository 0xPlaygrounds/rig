//! What a plain OpenAI configuration pointed at a local server does
//! differently.
//!
//! **This suite is deliberately small and must stay small.** Rig's llama.cpp
//! coverage used to exist twice — once through `providers::llamafile` and once
//! through an unconfigured OpenAI client aimed at the same server — and 19 of
//! 61 fixtures were the same scenario recorded down two code paths.
//! Re-recording the generation matrix here is how that happens again.
//!
//! Since the wire unification the two paths are the *same type*: an
//! [`OpenAI`](rig::providers::openai::wire::OpenAI) configuration bound to a
//! socket. What differs is the
//! [`Dialect`](rig::providers::openai::wire::Dialect) it carries — `OPENAI`
//! here, `LLAMACPP` everywhere else in this suite — and that difference is
//! exactly what is still worth covering, because it is not generation:
//!
//! | Difference | Cell |
//! | --- | --- |
//! | base-URL composition — the caller supplies `/v1`, the dialect's default carries it | [`caller_supplies_the_v1_prefix_the_provider_would_add`] |
//! | the `Authorization` header — `Auth::Bearer` always sends one | [`bare_openai_client_always_sends_an_authorization_header`] |
//! | the absence of this dialect's quirk flags — a fragmented tool-call stream still reassembles | [`a_fragmented_tool_call_stream_reassembles_without_the_provider_consts`] |
//! | the Responses/Completions split — the `OPENAI` dialect defaults to `/responses`, so a local server is reached by routing the configuration to Chat once | [`agent_prompt_through_completions_api`] |
//! | `raw` under the `openai` descriptor name, not `llamacpp` | [`raw_response_text_matches_normalized_choice_text`] |
//!
//! Recorded against the default server (`--jinja --seed 42 --temp 0 -c 4096`,
//! `unsloth/Qwen3-1.7B-GGUF` Q4_K_M, `llama-server` b10964-b29c606e2).

use rig::completion::CompletionModel;
use rig::prelude::*;
use rig::providers::openai::wire::{LLAMACPP, OpenAI};

use crate::support::{
    Adder, RAW_TEXT_RESPONSE_PREAMBLE, RAW_TEXT_RESPONSE_PROMPT, STREAMING_TOOLS_PREAMBLE,
    STREAMING_TOOLS_PROMPT, Subtract, assert_contains_all_case_insensitive,
    assert_mentions_expected_number, assert_nonempty_response, assistant_text_response,
    collect_stream_final_response,
};

use super::super::cassette_support::*;

/// The caller carries the `/v1` the `LLAMACPP` dialect's default base URL
/// carries for them.
///
/// The recorded path is identical on both sides of the boundary; what differs
/// is who put the prefix there. This cell fails the moment the OpenAI wire
/// starts composing a base URL differently, which would silently 404 every
/// local-server user who followed rig's own OpenAI-compatible instructions.
#[tokio::test]
async fn caller_supplies_the_v1_prefix_the_provider_would_add() {
    // The scenario string is repeated as a literal at every call site on
    // purpose: the cassette-safety scan reads them off the AST, and a `const`
    // registers nothing.
    with_llamacpp_bare_openai_cassette(
        "bare_openai_client/caller_supplies_the_v1_prefix",
        |client| async move {
            let agent = client
                .agent(CASSETTE_MODEL)
                .preamble("You are a concise assistant.")
                .max_tokens(256)
                .build();

            let response = agent
                .prompt("Say the single word: ok")
                .await
                .expect("a plain OpenAI configuration should reach the local server");
            assert_nonempty_response(&response.output);
        },
    )
    .await;

    // The premise, read back off the cassette's own bytes: the request landed
    // on `/v1/chat/completions` and nothing doubled the prefix.
    let recorded = crate::cassettes::recorded_request_paths(
        "llamacpp",
        "bare_openai_client/caller_supplies_the_v1_prefix",
    );
    assert_eq!(
        recorded,
        vec!["/v1/chat/completions".to_string()],
        "a bare openai client with a `/v1` base URL must produce exactly one `/v1`"
    );
}

/// The `OPENAI` dialect authenticates with `Auth::Bearer`, so it always sends
/// `Authorization` — and llama.cpp accepts any bearer token when it was not
/// started with `--api-key`.
///
/// The `LLAMACPP` dialect's `Auth::OptionalBearer` sends no header at all for
/// an empty key in the same situation. That asymmetry is the whole reason the
/// dialect names its own `Auth`, and it is why a server started *with*
/// `--api-key` was unreachable before this PR.
///
/// The header cannot be read back from a fixture — `authorization` is
/// sensitive and is scrubbed out of every recording — so the cell proves it
/// two ways instead. In process, both configurations are bound to the same
/// recording socket and their headers compared directly; on the wire, the
/// recorded turn shows the local server accepting the request the header rode
/// on.
#[tokio::test]
async fn bare_openai_client_always_sends_an_authorization_header() {
    // The in-process half: two configurations, one socket, one comparison.
    {
        use rig::embeddings::EmbeddingModel as _;
        use rig::test_utils::RecordingHttpClient;

        let recorder = RecordingHttpClient::new(
            r#"{"object":"list","model":"m","usage":{"prompt_tokens":1,"total_tokens":1},
                "data":[{"object":"embedding","index":0,"embedding":[0.1]}]}"#,
        );
        let bare = OpenAI::new("llamacpp-local").bind(recorder.clone());
        let _ = bare
            .embedding("m", Some(1))
            .embed_texts(["probe".to_string()])
            .await;
        let sent = &recorder.requests()[0];
        assert_eq!(
            sent.headers
                .get("authorization")
                .map(|value| value.to_str().unwrap_or_default()),
            Some("Bearer llamacpp-local"),
            "the `OPENAI` dialect has no way *not* to send one"
        );

        let recorder = RecordingHttpClient::new(
            r#"{"object":"list","model":"m","usage":{"prompt_tokens":1,"total_tokens":1},
                "data":[{"object":"embedding","index":0,"embedding":[0.1]}]}"#,
        );
        let provider = OpenAI::with_key(&LLAMACPP, "").bind(recorder.clone());
        let _ = provider
            .embedding("m", Some(1))
            .embed_texts(["probe".to_string()])
            .await;
        assert!(
            recorder.requests()[0]
                .headers
                .get("authorization")
                .is_none(),
            "and `Auth::OptionalBearer` with an empty key has a way not to, \
             which is the asymmetry this cell exists for"
        );
    }

    // The on-the-wire half: the local server accepts it.
    with_llamacpp_bare_openai_cassette(
        "bare_openai_client/authorization_header_is_always_sent",
        |client| async move {
            let model = client.chat(CASSETTE_MODEL);
            let response = model
                .completion(
                    model
                        .completion_request("Reply with the single word: ok")
                        .max_tokens(256)
                        .build(),
                )
                .await
                .expect("an unauthenticated local server accepts any bearer token");
            assert!(!response.choice.is_empty());
        },
    )
    .await;

    // And the recorded turn carries no `authorization` at all — which is the
    // stated reason the fixture cannot be the proof, turned into an assertion
    // about the recorder rather than a restatement of the cell's own input.
    let headers = crate::cassettes::recorded_request_header_pairs(
        "llamacpp",
        "bare_openai_client/authorization_header_is_always_sent",
    );
    assert!(!headers.is_empty(), "the scenario recorded an interaction");
    for interaction in &headers {
        assert!(
            !interaction.iter().any(|(name, _)| name == "authorization"),
            "`authorization` is not on RECORDED_REQUEST_HEADERS, so no fixture in \
             this repository may ever contain one: {interaction:?}"
        );
        assert!(
            interaction.iter().any(|(name, _)| name == "content-type"),
            "the recording does keep the headers it is meant to, or the check \
             above is vacuous: {interaction:?}"
        );
    }
}

/// The same tool-call stream, decoded under a dialect that is **not**
/// `LLAMACPP` — and reassembled identically.
///
/// This cell used to be framed around
/// `EMITS_COMPLETE_SINGLE_CHUNK_TOOL_CALLS` differing between the two paths.
/// It no longer does: this PR measured llama.cpp's streaming and left
/// `emits_complete_single_chunk_tool_calls` at the OpenAI default of `false`
/// for both dialects, so both take the same branch. Keeping the old framing
/// would have left a cell whose doc described a difference that does not
/// exist.
///
/// What it is worth instead is the *reassembly* claim, which no other cell in
/// this file makes: llama.cpp streams tool-call arguments one token at a time,
/// and a caller who reaches it through a plain `OPENAI` configuration — with
/// none of the `LLAMACPP` quirk flags — must still get one complete call with
/// parseable arguments. The premise is re-derived from the fixture: the
/// recorded stream must genuinely be fragmented, or the cell tests nothing.
#[tokio::test]
async fn a_fragmented_tool_call_stream_reassembles_without_the_provider_consts() {
    with_llamacpp_bare_openai_cassette(
        "bare_openai_client/tool_call_stream_without_the_single_chunk_const",
        |client| async move {
            let agent = client
                .agent(CASSETTE_MODEL)
                .preamble(STREAMING_TOOLS_PREAMBLE)
                .tool(Adder)
                .tool(Subtract)
                .build();

            let mut stream = agent.prompt(STREAMING_TOOLS_PROMPT).max_turns(4).stream();
            let response = collect_stream_final_response(&mut stream)
                .await
                .expect("streaming tool prompt should succeed");

            assert_mentions_expected_number(&response, -3);
        },
    )
    .await;

    // The premise: the recorded stream really did split the call's arguments
    // across fragments, and the first of them is not parseable on its own.
    let frames = crate::cassettes::recorded_sse_json_frames(
        "llamacpp",
        "bare_openai_client/tool_call_stream_without_the_single_chunk_const",
    );
    let fragments: Vec<String> = frames
        .iter()
        .flat_map(|frame| {
            frame["choices"][0]["delta"]["tool_calls"]
                .as_array()
                .cloned()
                .unwrap_or_default()
        })
        .filter_map(|call| call["function"]["arguments"].as_str().map(str::to_string))
        .collect();
    assert!(
        fragments.len() > 1,
        "the recorded stream must be fragmented for this cell to be about \
         reassembly at all: {fragments:?}"
    );
    assert!(
        serde_json::from_str::<serde_json::Value>(&fragments[0]).is_err(),
        "and the opening fragment must not parse on its own: {:?}",
        fragments[0]
    );
    let assembled: String = fragments.concat();
    let parsed: serde_json::Value = serde_json::from_str(&assembled)
        .unwrap_or_else(|error| panic!("the concatenation must parse: {error}: {assembled:?}"));
    assert!(parsed.is_object(), "{parsed}");
}

/// The Responses/Completions split is configuration rather than one client's
/// two surfaces: the `OPENAI` dialect's flagship is `/responses`, and llama.cpp
/// is reachable by routing the configuration to Chat once, after which the
/// agent sugar follows.
#[tokio::test]
async fn agent_prompt_through_completions_api() {
    with_llamacpp_bare_openai_cassette(
        "bare_openai_client/agent_prompt_through_completions_api",
        |client| async move {
            let agent = client
                .agent(CASSETTE_MODEL)
                .preamble("You are a helpful assistant.")
                .build();

            let response = agent
                .prompt("Hello world!")
                .await
                .expect("completions api prompt should succeed");

            assert_nonempty_response(&response.output);
        },
    )
    .await;
}

/// `raw` on this path is the server's verbatim chat-completions payload, and
/// it rides under the `openai` descriptor name rather than `llamacpp`.
#[tokio::test]
async fn raw_response_text_matches_normalized_choice_text() {
    with_llamacpp_bare_openai_cassette(
        "bare_openai_client/raw_response_text_matches_normalized_choice_text",
        |client| async move {
            let model = client.chat(CASSETTE_MODEL);
            let request = model
                .completion_request(RAW_TEXT_RESPONSE_PROMPT)
                .preamble(RAW_TEXT_RESPONSE_PREAMBLE.to_string())
                .build();
            // One request, two views of the one reply: `raw` holds the
            // server's own chat-completions JSON verbatim, and `choice` holds
            // what the decoder folded it into. The assistant text must be the
            // same text either way.
            let response = model
                .completion(request)
                .await
                .expect("completions api request should succeed");
            assert_eq!(
                response.provider, "openai",
                "the `OPENAI` dialect names itself, whatever server answered"
            );
            let raw_text = response.raw["choices"][0]["message"]["content"]
                .as_str()
                .expect("raw response should carry the assistant text");

            let normalized_text = assistant_text_response(&response.choice)
                .expect("normalized completions api response should contain assistant text");

            assert_nonempty_response(&normalized_text);
            assert_nonempty_response(raw_text);
            assert_contains_all_case_insensitive(raw_text, &["cedar", "maple"]);
            assert_eq!(raw_text.trim(), normalized_text.trim());
        },
    )
    .await;
}
