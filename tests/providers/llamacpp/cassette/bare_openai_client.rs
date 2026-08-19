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
//! | the absence of this provider's consts — `openai`'s `EMITS_COMPLETE_SINGLE_CHUNK_TOOL_CALLS` is `false` | [`tool_call_streams_decode_without_the_single_chunk_const`] |
//! | the Responses/Completions split, which `llamacpp::Client` does not have | [`agent_prompt_through_completions_api`] |
//! | `raw_completion` normalization under the `openai` descriptor name | [`raw_response_text_matches_normalized_choice_text`] |
//!
//! Recorded against the default server (`--jinja --seed 42 --temp 0 -c 4096`,
//! `unsloth/Qwen3-1.7B-GGUF` Q4_K_M, `llama-server` b10499-6d05498).

use rig::completion::CompletionModel;
use rig::completion::NormalizeCompletionResponse;
use rig::completion::Prompt;
use rig::prelude::*;
use rig::streaming::StreamingPrompt;
use rig::telemetry::ProviderResponseExt;

use crate::cassettes::recorded_json_request;
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
    const SCENARIO: &str = "bare_openai_client/caller_supplies_the_v1_prefix";

    with_llamacpp_bare_openai_cassette(SCENARIO, |client| async move {
        let agent = client
            .completions_api()
            .agent(CASSETTE_MODEL)
            .preamble("You are a concise assistant.")
            .max_tokens(24)
            .build();

        let response = agent
            .prompt("Say the single word: ok")
            .await
            .expect("a bare openai client should reach the local server");
        assert_nonempty_response(&response);
    })
    .await;

    // The premise, read back off the cassette's own bytes: the request landed
    // on `/v1/chat/completions` and nothing doubled the prefix.
    let recorded = crate::cassettes::recorded_request_paths("llamacpp", SCENARIO);
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
/// `llamacpp::Client` sends no header at all in the same situation (pinned
/// definitionally in `providers::llamacpp::client`'s unit tests). That
/// asymmetry is the whole reason the provider needed its own key type, and it
/// is why a server started *with* `--api-key` was unreachable before this PR.
#[tokio::test]
async fn bare_openai_client_always_sends_an_authorization_header() {
    const SCENARIO: &str = "bare_openai_client/authorization_header_is_always_sent";

    with_llamacpp_bare_openai_cassette(SCENARIO, |client| async move {
        let model = client.completions_api().completion_model(CASSETTE_MODEL);
        let response = model
            .completion(
                model
                    .completion_request("Reply with the single word: ok")
                    .max_tokens(16)
                    .build(),
            )
            .await
            .expect("an unauthenticated local server accepts any bearer token");
        assert!(!response.choice.is_empty());
    })
    .await;

    // `authorization` is a sensitive header and is scrubbed out of every
    // fixture, so the *presence* of the header cannot be read back from the
    // cassette. What the cassette does prove is that the request the header
    // rode on was accepted, which is the behavioral half. The header itself is
    // pinned in-process by `providers::llamacpp::client`'s unit tests, which
    // read it off a recording HTTP client rather than off a fixture.
    let request = recorded_json_request("llamacpp", SCENARIO);
    assert_eq!(request["model"], serde_json::json!(CASSETTE_MODEL));
}

/// The same tool-call stream, decoded by a provider whose
/// `EMITS_COMPLETE_SINGLE_CHUNK_TOOL_CALLS` is `false`.
///
/// llama.cpp emits a whole tool call in one chunk; `openai`'s extension does
/// not claim that, so the shared streaming layer holds the call until the
/// stream ends instead of emitting it on arrival. Both must still produce the
/// same answer — a const that changed *what a stream means* rather than *when
/// it is delivered* would break here and nowhere else.
#[tokio::test]
async fn tool_call_streams_decode_without_the_single_chunk_const() {
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
                .await;
            let response = collect_stream_final_response(&mut stream)
                .await
                .expect("streaming tool prompt should succeed");

            assert_mentions_expected_number(&response, -3);
        },
    )
    .await;
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

            assert_nonempty_response(&response);
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
                .get_text_response()
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
