//! What a bare `openai::Client` pointed at a local server does differently.
//!
//! **This suite is deliberately small and must stay small.** Rig's llama.cpp
//! coverage used to exist twice — once through `providers::llamafile` and once
//! through a bare `openai::Client` aimed at the same server — and 19 of 61
//! fixtures were the same scenario recorded down two code paths. Re-recording
//! the generation matrix here is how that happens again.
//!
//! What this path is still worth covering is the part that genuinely differs,
//! and that is not generation:
//!
//! | Difference | Cell |
//! | --- | --- |
//! | base-URL composition — the caller supplies `/v1`, the provider does not | [`caller_supplies_the_v1_prefix_the_provider_would_add`] |
//! | the `Authorization` header — `openai::Client` always sends one | [`bare_openai_client_always_sends_an_authorization_header`] |
//! | the absence of this provider's consts — a fragmented tool-call stream still reassembles | [`a_fragmented_tool_call_stream_reassembles_without_the_provider_consts`] |
//! | the Responses/Completions split, which `llamacpp::Client` does not have | [`agent_prompt_through_completions_api`] |
//! | `raw_completion` normalization under the `openai` descriptor name | [`raw_response_text_matches_normalized_choice_text`] |
//!
//! Recorded against the default server (`--jinja --seed 42 --temp 0 -c 4096`,
//! `unsloth/Qwen3-1.7B-GGUF` Q4_K_M, `llama-server` b10499-6d05498).

use rig::completion::CompletionModel;
use rig::completion::NormalizeCompletionResponse;
use rig::prelude::*;
use rig::providers::{llamacpp, openai};
use rig::telemetry::ProviderResponseExt;

use crate::support::{
    Adder, RAW_TEXT_RESPONSE_PREAMBLE, RAW_TEXT_RESPONSE_PROMPT, STREAMING_TOOLS_PREAMBLE,
    STREAMING_TOOLS_PROMPT, Subtract, assert_contains_all_case_insensitive,
    assert_mentions_expected_number, assert_nonempty_response, assistant_text_response,
    collect_stream_final_response,
};

use super::super::cassette_support::*;

/// The caller carries the `/v1` that `llamacpp::Client` adds for them.
///
/// The recorded path is identical on both sides of the boundary; what differs
/// is who put the prefix there. This cell fails the moment `openai::Client`
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
                .completions_api()
                .agent(CASSETTE_MODEL)
                .preamble("You are a concise assistant.")
                .max_tokens(256)
                .build();

            let response = agent
                .prompt("Say the single word: ok")
                .await
                .expect("a bare openai client should reach the local server");
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

/// `openai::Client` has no optional-key form, so it always sends
/// `Authorization` — and llama.cpp accepts any bearer token when it was not
/// started with `--api-key`.
///
/// `llamacpp::Client` sends no header at all in the same situation. That
/// asymmetry is the whole reason the provider needed its own key type, and it
/// is why a server started *with* `--api-key` was unreachable before this PR.
///
/// The header cannot be read back from a fixture — `authorization` is
/// sensitive and is scrubbed out of every recording — so the cell proves it
/// two ways instead. In process, both clients are driven through the same
/// recording HTTP backend and their headers compared directly; on the wire,
/// the recorded turn shows the local server accepting the request the header
/// rode on.
#[tokio::test]
async fn bare_openai_client_always_sends_an_authorization_header() {
    // The in-process half: two clients, one backend, one comparison.
    {
        use rig::embeddings::EmbeddingModel as _;
        use rig::test_utils::RecordingHttpClient;

        let recorder = RecordingHttpClient::new(
            r#"{"object":"list","model":"m","usage":{"prompt_tokens":1,"total_tokens":1},
                "data":[{"object":"embedding","index":0,"embedding":[0.1]}]}"#,
        );
        let bare = openai::Client::builder()
            .api_key("llamacpp-local")
            .http_client(recorder.clone())
            .build()
            .expect("client should build");
        let _ = bare
            .embedding_model_with_ndims("m", 1)
            .embed_texts(["probe".to_string()])
            .await;
        let sent = &recorder.requests()[0];
        assert_eq!(
            sent.headers
                .get("authorization")
                .map(|value| value.to_str().unwrap_or_default()),
            Some("Bearer llamacpp-local"),
            "a bare openai::Client has no way *not* to send one"
        );

        let recorder = RecordingHttpClient::new(
            r#"{"object":"list","model":"m","usage":{"prompt_tokens":1,"total_tokens":1},
                "data":[{"object":"embedding","index":0,"embedding":[0.1]}]}"#,
        );
        let provider = llamacpp::Client::builder()
            .api_key(llamacpp::LlamacppApiKey::default())
            .http_client(recorder.clone())
            .build()
            .expect("client should build");
        let _ = provider
            .embedding_model_with_ndims("m", 1)
            .embed_texts(["probe".to_string()])
            .await;
        assert!(
            recorder.requests()[0]
                .headers
                .get("authorization")
                .is_none(),
            "and the provider has a way not to, which is the asymmetry this cell \
             exists for"
        );
    }

    // The on-the-wire half: the local server accepts it.
    with_llamacpp_bare_openai_cassette(
        "bare_openai_client/authorization_header_is_always_sent",
        |client| async move {
            let model = client.completions_api().completion_model(CASSETTE_MODEL);
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

/// The same tool-call stream, decoded by a provider that is **not**
/// `llamacpp` — and reassembled identically.
///
/// This cell used to be framed around
/// `EMITS_COMPLETE_SINGLE_CHUNK_TOOL_CALLS` differing between the two paths.
/// It no longer does: this PR measured llama.cpp's streaming and set the
/// llamacpp const to `false`, which is also `openai`'s trait default, so both
/// paths now take the same branch. Keeping the old framing would have left a
/// cell whose doc described a difference that does not exist.
///
/// What it is worth instead is the *reassembly* claim, which no other cell in
/// this file makes: llama.cpp streams tool-call arguments one token at a time,
/// and a caller who reaches it through a bare `openai::Client` — with none of
/// this provider's associated consts — must still get one complete call with
/// parseable arguments. The premise is re-derived from the fixture: the
/// recorded stream must genuinely be fragmented, or the cell tests nothing.
#[tokio::test]
async fn a_fragmented_tool_call_stream_reassembles_without_the_provider_consts() {
    with_llamacpp_bare_openai_cassette(
        "bare_openai_client/tool_call_stream_without_the_single_chunk_const",
        |client| async move {
            let agent = client
                .completions_api()
                .agent(CASSETTE_MODEL)
                .preamble(STREAMING_TOOLS_PREAMBLE)
                .tool(Adder)
                .tool(Subtract)
                .build();

            let mut stream = agent
                .stream_prompt(STREAMING_TOOLS_PROMPT)
                .max_turns(4)
                .stream()
                .await;
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

/// `openai::Client` exposes a Responses/Completions split that
/// `llamacpp::Client` does not have.
#[tokio::test]
async fn agent_prompt_through_completions_api() {
    with_llamacpp_bare_openai_cassette(
        "bare_openai_client/agent_prompt_through_completions_api",
        |client| async move {
            let agent = client
                .clone()
                .completion_model(CASSETTE_MODEL)
                .completions_api()
                .into_agent_builder()
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

/// `raw_completion` on this path normalizes under the `openai` descriptor
/// name, not `llamacpp`.
#[tokio::test]
async fn raw_response_text_matches_normalized_choice_text() {
    with_llamacpp_bare_openai_cassette(
        "bare_openai_client/raw_response_text_matches_normalized_choice_text",
        |client| async move {
            let client = client.completions_api();
            let model = client.completion_model(CASSETTE_MODEL);
            let request = model
                .completion_request(RAW_TEXT_RESPONSE_PROMPT)
                .preamble(RAW_TEXT_RESPONSE_PREAMBLE.to_string())
                .build();
            // One request, two views: `raw_completion` returns llama.cpp's own
            // wire response and the provider-local conversion produces exactly
            // what `CompletionModel::completion` would have returned for it.
            let raw = model
                .raw_completion(request)
                .await
                .expect("raw completions api request should succeed");
            let raw_text = raw
                .text_response()
                .expect("raw completions api response should contain assistant text");
            let response: rig::completion::CompletionResponse = raw
                .normalize("openai")
                .expect("raw completions api response should normalize");

            let normalized_text = assistant_text_response(&response.choice)
                .expect("normalized completions api response should contain assistant text");

            assert_nonempty_response(&normalized_text);
            assert_nonempty_response(&raw_text);
            assert_contains_all_case_insensitive(&raw_text, &["cedar", "maple"]);
            assert_eq!(raw_text.trim(), normalized_text.trim());
        },
    )
    .await;
}
