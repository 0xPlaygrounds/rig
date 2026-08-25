//! Edge matrix for the output-token-cap spelling on OpenAI Chat Completions.
//!
//! **Bug.** Rig always sent the cap as the legacy `max_tokens` field. OpenAI
//! deprecated that spelling for `/v1/chat/completions` and its reasoning-class
//! models reject it outright:
//!
//! ```text
//! Unsupported parameter: 'max_tokens' is not supported with this model.
//! Use 'max_completion_tokens' instead.
//! ```
//!
//! So `agent.max_tokens(n)` — or any capped request — could not succeed at all
//! against `gpt-5*` / `o*` through the Chat Completions surface. The fix sends
//! `max_completion_tokens` **only for the models that reject the legacy field**
//! (`OpenAICompatibleProvider::requires_modern_output_cap`, implemented for
//! OpenAI as `is_openai_reasoning_model`: the `gpt-5`-and-up and `o`-series
//! families). Scoping it to those models rather than to the whole endpoint is
//! deliberate: this same extension is how rig reaches OpenAI-*compatible*
//! servers (mistral.rs, vLLM, llama.cpp, gateways), and OpenAI's own older
//! models still take `max_tokens` — so nothing that worked before changes a
//! byte, which the untouched `mistralrs` suite and cells 9-13 below prove.
//!
//! **How these cells fail on `origin/main`.** The cassette harness matches the
//! recorded *request body*, so every **reasoning-model** cell below is a mock
//! miss on `main` (which sends `max_tokens`) and passes with the fix. The
//! non-reasoning cells are controls: their recorded bodies are byte-identical
//! on both sides, which is exactly the claim they exist to make.
//!
//! | # | cell | model | class | transport | level | cap | status |
//! |---|------|-------|-------|-----------|-----|-----|--------|
//! | 1 | `reasoning_gpt5_nano_blocking_cap` | gpt-5-nano | reasoning | blocking | raw model | set | recorded |
//! | 2 | `reasoning_gpt5_nano_streaming_cap` | gpt-5-nano | reasoning | streaming | raw model | set | recorded |
//! | 3 | `reasoning_o4_mini_blocking_cap` | o4-mini | reasoning | blocking | raw model | set | recorded |
//! | 4 | `reasoning_o4_mini_streaming_cap` | o4-mini | reasoning | streaming | raw model | set | recorded |
//! | 5 | `reasoning_gpt5_nano_agent_blocking_cap` | gpt-5-nano | reasoning | blocking | agent | set | recorded |
//! | 6 | `reasoning_gpt5_nano_agent_streaming_cap` | gpt-5-nano | reasoning | streaming | agent | set | recorded |
//! | 7 | `reasoning_gpt5_nano_tool_turn_cap` | gpt-5-nano | reasoning | blocking | agent + tool | set | recorded |
//! | 8 | `reasoning_gpt5_nano_tool_turn_streaming_cap` | gpt-5-nano | reasoning | streaming | agent + tool | set | recorded |
//! | 9 | `legacy_gpt_4o_mini_blocking_cap` | gpt-4o-mini | non-reasoning (control) | blocking | raw model | set | recorded |
//! | 10 | `legacy_gpt_4o_mini_streaming_cap` | gpt-4o-mini | non-reasoning (control) | streaming | raw model | set | recorded |
//! | 11 | `legacy_gpt_4_1_nano_blocking_cap` | gpt-4.1-nano | non-reasoning (control) | blocking | raw model | set | recorded |
//! | 12 | `legacy_gpt_3_5_turbo_blocking_cap` | gpt-3.5-turbo | oldest (control) | blocking | raw model | set | recorded |
//! | 13 | `legacy_gpt_3_5_turbo_streaming_cap` | gpt-3.5-turbo | oldest (control) | streaming | raw model | set | recorded |
//! | 14 | `uncapped_blocking_sends_neither_spelling` | gpt-4o-mini | non-reasoning | blocking | raw model | unset | recorded |
//! | 15 | `uncapped_streaming_sends_neither_spelling` | gpt-4o-mini | non-reasoning | streaming | raw model | unset | recorded |
//! | 16 | `caller_modern_spelling_wins_over_cap` | gpt-5-nano | reasoning | blocking | raw model | both | recorded |
//! | 17 | `caller_modern_spelling_alone_survives` | gpt-5-nano | reasoning | blocking | raw model | additional only | recorded |
//! | 18 | `caller_legacy_spelling_is_upgraded` | gpt-5-nano | reasoning | blocking | raw model | additional legacy | recorded |
//! | 19 | `responses_surface_keeps_max_output_tokens` | gpt-4o-mini | control | blocking | raw model | set | recorded |
//! | 20 | `responses_surface_reasoning_model_cap` | gpt-5-nano | control | blocking | raw model | set | recorded |
//! | 21 | `responses_surface_streaming_cap` | gpt-4o-mini | control | streaming | raw model | set | recorded |
//! | 22 | `capped_extractor_turn_on_reasoning_model` | gpt-5-nano | reasoning | blocking | extractor | set | recorded |
//!
//! Unit cells (the field-spelling rule itself is definitory — no live turn can
//! observe a body rig did not send) live beside the fix in
//! `crates/rig-core/src/providers/openai/completion/mod.rs`:
//! `request_body_*` and `modern_output_cap_*`. They cover the legacy spelling,
//! the modern one, absent caps, caller-supplied spellings on both keys, that no
//! other request field moves, and the base-URL gate itself — including that an
//! OpenAI-compatible server reached through this extension keeps `max_tokens`,
//! which the untouched `mistralrs`, `vllm`, and `doubleword` suites replay as
//! their own proof.

use rig::client::completion::CompletionClient;
use rig::completion::CompletionModel;
use rig::extractor::ExtractorBuilder;
use rig::prelude::*;
use rig::providers::openai;
use rig::streaming::StreamingPrompt;
use serde_json::json;

use super::super::support::with_openai_max_tokens_cassette;
use crate::support::{
    Adder, SmokePerson, assert_nonempty_response, assistant_text_response,
    collect_raw_stream_observation, collect_stream_observation,
};

/// A cap large enough that a reasoning model still has budget for visible
/// text after its hidden reasoning tokens.
const CAP: u64 = 256;
/// Tool-calling and extraction turns reason more before they emit anything, so
/// they need headroom the plain-answer cells do not — a cap these turns
/// exhaust would test truncation (see `truncated_turn_matrix`) rather than the
/// spelling of the field that carries it.
const TOOL_CAP: u64 = 4096;
const PROMPT: &str = "Reply with the single word OK.";
const PREAMBLE: &str = "You are a terse assistant.";

// ---------------------------------------------------------------------------
// Reasoning-class models: these could not be capped at all before the fix.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn reasoning_gpt5_nano_blocking_cap() {
    with_openai_max_tokens_cassette(
        "max_completion_tokens_matrix/reasoning_gpt5_nano_blocking_cap",
        |client| async move {
            let model = client.completions_api().completion_model("gpt-5-nano");
            let request = model.completion_request(PROMPT).max_tokens(CAP).build();

            let response = model
                .completion(request)
                .await
                .expect("a capped reasoning-model turn must be accepted");

            assert_nonempty_response(
                &assistant_text_response(&response.choice).expect("assistant text"),
            );
        },
    )
    .await;
}

#[tokio::test]
async fn reasoning_gpt5_nano_streaming_cap() {
    with_openai_max_tokens_cassette(
        "max_completion_tokens_matrix/reasoning_gpt5_nano_streaming_cap",
        |client| async move {
            let model = client.completions_api().completion_model("gpt-5-nano");
            let request = model.completion_request(PROMPT).max_tokens(CAP).build();

            let stream = model
                .stream(request)
                .await
                .expect("a capped reasoning-model stream must connect");
            let observed = collect_raw_stream_observation(stream).await;

            assert_nonempty_response(&observed.text);
        },
    )
    .await;
}

#[tokio::test]
async fn reasoning_o4_mini_blocking_cap() {
    with_openai_max_tokens_cassette(
        "max_completion_tokens_matrix/reasoning_o4_mini_blocking_cap",
        |client| async move {
            let model = client.completions_api().completion_model("o4-mini");
            let request = model.completion_request(PROMPT).max_tokens(CAP).build();

            let response = model
                .completion(request)
                .await
                .expect("a capped o-series turn must be accepted");

            assert_nonempty_response(
                &assistant_text_response(&response.choice).expect("assistant text"),
            );
        },
    )
    .await;
}

#[tokio::test]
async fn reasoning_o4_mini_streaming_cap() {
    with_openai_max_tokens_cassette(
        "max_completion_tokens_matrix/reasoning_o4_mini_streaming_cap",
        |client| async move {
            let model = client.completions_api().completion_model("o4-mini");
            let request = model.completion_request(PROMPT).max_tokens(CAP).build();

            let stream = model
                .stream(request)
                .await
                .expect("a capped o-series stream must connect");
            let observed = collect_raw_stream_observation(stream).await;

            assert_nonempty_response(&observed.text);
        },
    )
    .await;
}

#[tokio::test]
async fn reasoning_gpt5_nano_agent_blocking_cap() {
    with_openai_max_tokens_cassette(
        "max_completion_tokens_matrix/reasoning_gpt5_nano_agent_blocking_cap",
        |client| async move {
            let agent = client
                .completions_api()
                .agent("gpt-5-nano")
                .preamble(PREAMBLE)
                .max_tokens(CAP)
                .build();

            let response = agent
                .prompt(PROMPT)
                .await
                .expect("an agent-level cap must reach a reasoning model");

            assert_nonempty_response(&response);
        },
    )
    .await;
}

#[tokio::test]
async fn reasoning_gpt5_nano_agent_streaming_cap() {
    with_openai_max_tokens_cassette(
        "max_completion_tokens_matrix/reasoning_gpt5_nano_agent_streaming_cap",
        |client| async move {
            let agent = client
                .completions_api()
                .agent("gpt-5-nano")
                .preamble(PREAMBLE)
                .max_tokens(CAP)
                .build();

            let mut stream = agent.stream_prompt(PROMPT).await;
            let observed = collect_stream_observation(&mut stream).await;

            assert!(observed.errors.is_empty(), "{:?}", observed.errors);
            assert_nonempty_response(&observed.all_streamed_text);
        },
    )
    .await;
}

#[tokio::test]
async fn reasoning_gpt5_nano_tool_turn_cap() {
    with_openai_max_tokens_cassette(
        "max_completion_tokens_matrix/reasoning_gpt5_nano_tool_turn_cap",
        |client| async move {
            let agent = client
                .completions_api()
                .agent("gpt-5-nano")
                .preamble("Use the add tool to answer arithmetic questions.")
                .max_tokens(TOOL_CAP)
                .tool(Adder)
                .build();

            let response = agent
                .prompt("What is 3 + 4?")
                .max_turns(3)
                .await
                .expect("a capped tool-calling turn must be accepted");

            assert!(response.contains('7'), "{response}");
        },
    )
    .await;
}

#[tokio::test]
async fn reasoning_gpt5_nano_tool_turn_streaming_cap() {
    with_openai_max_tokens_cassette(
        "max_completion_tokens_matrix/reasoning_gpt5_nano_tool_turn_streaming_cap",
        |client| async move {
            let agent = client
                .completions_api()
                .agent("gpt-5-nano")
                .preamble("Use the add tool to answer arithmetic questions.")
                .max_tokens(TOOL_CAP)
                .tool(Adder)
                .build();

            let mut stream = agent.stream_prompt("What is 3 + 4?").max_turns(3).await;
            let observed = collect_stream_observation(&mut stream).await;

            assert!(observed.errors.is_empty(), "{:?}", observed.errors);
            assert_eq!(observed.tool_calls, vec!["add".to_string()]);
        },
    )
    .await;
}

// ---------------------------------------------------------------------------
// Non-reasoning models: controls. These still take the legacy field, so their
// recorded request bodies must be byte-identical to what `main` sends.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn legacy_gpt_4o_mini_blocking_cap() {
    with_openai_max_tokens_cassette(
        "max_completion_tokens_matrix/legacy_gpt_4o_mini_blocking_cap",
        |client| async move {
            let model = client
                .completions_api()
                .completion_model(openai::GPT_4O_MINI);
            let request = model.completion_request(PROMPT).max_tokens(CAP).build();

            let response = model
                .completion(request)
                .await
                .expect("a non-reasoning model's request must not change at all");

            assert_nonempty_response(
                &assistant_text_response(&response.choice).expect("assistant text"),
            );
        },
    )
    .await;
}

#[tokio::test]
async fn legacy_gpt_4o_mini_streaming_cap() {
    with_openai_max_tokens_cassette(
        "max_completion_tokens_matrix/legacy_gpt_4o_mini_streaming_cap",
        |client| async move {
            let model = client
                .completions_api()
                .completion_model(openai::GPT_4O_MINI);
            let request = model.completion_request(PROMPT).max_tokens(CAP).build();

            let stream = model.stream(request).await.expect("stream should connect");
            let observed = collect_raw_stream_observation(stream).await;

            assert_nonempty_response(&observed.text);
        },
    )
    .await;
}

#[tokio::test]
async fn legacy_gpt_4_1_nano_blocking_cap() {
    with_openai_max_tokens_cassette(
        "max_completion_tokens_matrix/legacy_gpt_4_1_nano_blocking_cap",
        |client| async move {
            let model = client
                .completions_api()
                .completion_model(openai::GPT_4_1_NANO);
            let request = model.completion_request(PROMPT).max_tokens(CAP).build();

            let response = model.completion(request).await.expect("capped turn");

            assert_nonempty_response(
                &assistant_text_response(&response.choice).expect("assistant text"),
            );
        },
    )
    .await;
}

/// The oldest chat model rig names: the far end of the untouched set.
#[tokio::test]
async fn legacy_gpt_3_5_turbo_blocking_cap() {
    with_openai_max_tokens_cassette(
        "max_completion_tokens_matrix/legacy_gpt_3_5_turbo_blocking_cap",
        |client| async move {
            let model = client.completions_api().completion_model("gpt-3.5-turbo");
            let request = model.completion_request(PROMPT).max_tokens(CAP).build();

            let response = model
                .completion(request)
                .await
                .expect("the oldest chat model rig names keeps the legacy field");

            assert_nonempty_response(
                &assistant_text_response(&response.choice).expect("assistant text"),
            );
        },
    )
    .await;
}

#[tokio::test]
async fn legacy_gpt_3_5_turbo_streaming_cap() {
    with_openai_max_tokens_cassette(
        "max_completion_tokens_matrix/legacy_gpt_3_5_turbo_streaming_cap",
        |client| async move {
            let model = client.completions_api().completion_model("gpt-3.5-turbo");
            let request = model.completion_request(PROMPT).max_tokens(CAP).build();

            let stream = model.stream(request).await.expect("stream should connect");
            let observed = collect_raw_stream_observation(stream).await;

            assert_nonempty_response(&observed.text);
        },
    )
    .await;
}

// ---------------------------------------------------------------------------
// Boundary: no cap at all, and caller-supplied spellings.
// ---------------------------------------------------------------------------

/// An uncapped request must carry *neither* spelling — the recorded body is
/// the assertion.
#[tokio::test]
async fn uncapped_blocking_sends_neither_spelling() {
    with_openai_max_tokens_cassette(
        "max_completion_tokens_matrix/uncapped_blocking_sends_neither_spelling",
        |client| async move {
            let model = client
                .completions_api()
                .completion_model(openai::GPT_4O_MINI);
            let request = model.completion_request(PROMPT).build();

            let response = model.completion(request).await.expect("uncapped turn");

            assert_nonempty_response(
                &assistant_text_response(&response.choice).expect("assistant text"),
            );
        },
    )
    .await;
}

#[tokio::test]
async fn uncapped_streaming_sends_neither_spelling() {
    with_openai_max_tokens_cassette(
        "max_completion_tokens_matrix/uncapped_streaming_sends_neither_spelling",
        |client| async move {
            let model = client
                .completions_api()
                .completion_model(openai::GPT_4O_MINI);
            let request = model.completion_request(PROMPT).build();

            let stream = model.stream(request).await.expect("stream should connect");
            let observed = collect_raw_stream_observation(stream).await;

            assert_nonempty_response(&observed.text);
        },
    )
    .await;
}

/// A caller who already spells the modern field keeps their own value; the
/// legacy key still has to leave the body.
///
/// Proven behaviorally rather than by body shape alone: rig's cap here is far
/// too small for this model to say anything, so a turn that produces text can
/// only have been run under the caller's larger value.
#[tokio::test]
async fn caller_modern_spelling_wins_over_cap() {
    with_openai_max_tokens_cassette(
        "max_completion_tokens_matrix/caller_modern_spelling_wins_over_cap",
        |client| async move {
            let model = client.completions_api().completion_model("gpt-5-nano");
            let request = model
                .completion_request(PROMPT)
                .max_tokens(16)
                .additional_params(json!({ "max_completion_tokens": CAP }))
                .build();

            let response = model.completion(request).await.expect("capped turn");

            assert_nonempty_response(
                &assistant_text_response(&response.choice)
                    .expect("the caller's larger cap must be the one in force"),
            );
        },
    )
    .await;
}

#[tokio::test]
async fn caller_modern_spelling_alone_survives() {
    with_openai_max_tokens_cassette(
        "max_completion_tokens_matrix/caller_modern_spelling_alone_survives",
        |client| async move {
            let model = client.completions_api().completion_model("gpt-5-nano");
            let request = model
                .completion_request(PROMPT)
                .additional_params(json!({ "max_completion_tokens": CAP }))
                .build();

            let response = model.completion(request).await.expect("capped turn");

            assert_nonempty_response(
                &assistant_text_response(&response.choice).expect("assistant text"),
            );
        },
    )
    .await;
}

/// A caller who hand-spells the *legacy* field through `additional_params` is
/// upgraded too: the endpoint rejects the field's mere presence, so leaving it
/// alone would be a 400 the caller cannot see coming.
#[tokio::test]
async fn caller_legacy_spelling_is_upgraded() {
    with_openai_max_tokens_cassette(
        "max_completion_tokens_matrix/caller_legacy_spelling_is_upgraded",
        |client| async move {
            let model = client.completions_api().completion_model("gpt-5-nano");
            let request = model
                .completion_request(PROMPT)
                .additional_params(json!({ "max_tokens": CAP }))
                .build();

            let response = model
                .completion(request)
                .await
                .expect("a hand-spelled legacy cap must be upgraded, not rejected");

            assert_nonempty_response(
                &assistant_text_response(&response.choice).expect("assistant text"),
            );
        },
    )
    .await;
}

// ---------------------------------------------------------------------------
// Control: the Responses surface has its own field and must not move.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn responses_surface_keeps_max_output_tokens() {
    with_openai_max_tokens_cassette(
        "max_completion_tokens_matrix/responses_surface_keeps_max_output_tokens",
        |client| async move {
            let model = client.completion_model(openai::GPT_4O_MINI);
            let request = model.completion_request(PROMPT).max_tokens(CAP).build();

            let response = model.completion(request).await.expect("capped turn");

            assert_nonempty_response(
                &assistant_text_response(&response.choice).expect("assistant text"),
            );
        },
    )
    .await;
}

#[tokio::test]
async fn responses_surface_reasoning_model_cap() {
    with_openai_max_tokens_cassette(
        "max_completion_tokens_matrix/responses_surface_reasoning_model_cap",
        |client| async move {
            let model = client.completion_model("gpt-5-nano");
            let request = model.completion_request(PROMPT).max_tokens(CAP).build();

            let response = model.completion(request).await.expect("capped turn");

            assert_nonempty_response(
                &assistant_text_response(&response.choice).expect("assistant text"),
            );
        },
    )
    .await;
}

#[tokio::test]
async fn responses_surface_streaming_cap() {
    with_openai_max_tokens_cassette(
        "max_completion_tokens_matrix/responses_surface_streaming_cap",
        |client| async move {
            let model = client.completion_model(openai::GPT_4O_MINI);
            let request = model.completion_request(PROMPT).max_tokens(CAP).build();

            let stream = model.stream(request).await.expect("stream should connect");
            let observed = collect_raw_stream_observation(stream).await;

            assert_nonempty_response(&observed.text);
        },
    )
    .await;
}

/// The extractor drives the same request builder, so a capped extraction on a
/// reasoning model was unreachable for the same reason.
#[tokio::test]
async fn capped_extractor_turn_on_reasoning_model() {
    with_openai_max_tokens_cassette(
        "max_completion_tokens_matrix/capped_extractor_turn_on_reasoning_model",
        |client| async move {
            let extractor: rig::extractor::Extractor<SmokePerson> =
                ExtractorBuilder::new(client.completions_api().completion_model("gpt-5-nano"))
                    .max_tokens(TOOL_CAP)
                    .build();

            let person = extractor
                .extract("Ada Lovelace works as a mathematician.")
                .await
                .expect("a capped extraction must be accepted");

            assert_eq!(person.data.first_name.as_deref(), Some("Ada"));
        },
    )
    .await;
}
