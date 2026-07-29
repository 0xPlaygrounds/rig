//! Migrated from `examples/openai_agent_completions_api.rs`.

use rig::OneOrMany;
use rig::completion::CompletionModel;
use rig::completion::CompletionRequest;
use rig::completion::Prompt;
use rig::message::{AssistantContent, Message, ToolChoice};
use rig::prelude::*;
use rig::providers::openai;
use rig::streaming::StreamingPrompt;
use rig::telemetry::ProviderResponseExt;

use super::super::support::with_openai_completions_cassette;
use crate::support::{
    ALPHA_SIGNAL_OUTPUT, AlphaSignal, BETA_SIGNAL_OUTPUT, BetaSignal, ORDERED_TOOL_STREAM_PREAMBLE,
    ORDERED_TOOL_STREAM_PROMPT, RAW_TEXT_RESPONSE_PREAMBLE, RAW_TEXT_RESPONSE_PROMPT,
    REQUIRED_ZERO_ARG_TOOL_PROMPT, TWO_TOOL_STREAM_PREAMBLE, TWO_TOOL_STREAM_PROMPT,
    assert_contains_all_case_insensitive, assert_nonempty_response,
    assert_raw_stream_contains_distinct_tool_calls_before_text, assert_raw_stream_text_contains,
    assert_raw_stream_tool_call_precedes_text, assert_stream_contains_zero_arg_tool_call_named,
    assert_tool_call_precedes_later_text, assert_two_tool_roundtrip_contract,
    assistant_text_response, collect_raw_stream_observation, collect_stream_observation,
    zero_arg_tool_definition,
};

#[tokio::test]
async fn completions_api_agent_prompt() {
    with_openai_completions_cassette(
        "completions_api/completions_api_agent_prompt",
        |client| async move {
            let agent = client
                .agent(openai::GPT_4O)
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

#[tokio::test]
async fn completions_api_raw_response_text_matches_normalized_choice_text() {
    const SCENARIO: &str =
        "completions_api/completions_api_raw_response_text_matches_normalized_choice_text";
    with_openai_completions_cassette("completions_api/completions_api_raw_response_text_matches_normalized_choice_text", |client| async move {
        let model = client.completion_model(openai::GPT_4O);
        let request = CompletionRequest::with_history(
            Some(RAW_TEXT_RESPONSE_PREAMBLE),
            Vec::new(),
            RAW_TEXT_RESPONSE_PROMPT,
        );
        let response = model
            .completion(request)
            .await
            .expect("raw completions api request should succeed");

        let normalized_text = assistant_text_response(&response.choice)
            .expect("normalized completions api response should contain assistant text");
        assert_nonempty_response(&normalized_text);
        assert_contains_all_case_insensitive(&normalized_text, &["cedar", "maple"]);

        // The normalized response no longer carries the provider-typed
        // `raw_response`; parse the recorded wire body to compare the raw
        // completions text against the normalized choice text (replay only:
        // the cassette file is written after the test body in record mode).
        if crate::cassettes::CassetteMode::current() == crate::cassettes::CassetteMode::Replay {
            let bodies = crate::cassettes::recorded_response_bodies("openai", SCENARIO);
            assert_eq!(bodies.len(), 1, "scenario should record a single interaction");
            let raw: openai::completion::CompletionResponse = serde_json::from_str(&bodies[0])
                .expect("recorded body should deserialize as a completions api response");
            let raw_text = raw
                .get_text_response()
                .expect("raw completions api response should contain assistant text");

            assert_nonempty_response(&raw_text);
            assert_contains_all_case_insensitive(&raw_text, &["cedar", "maple"]);
            assert_eq!(raw_text.trim(), normalized_text.trim());
        }
    })
    .await;
}

#[tokio::test]
async fn completions_api_streams_two_tool_calls_before_final_answer() {
    with_openai_completions_cassette(
        "completions_api/completions_api_streams_two_tool_calls_before_final_answer",
        |client| async move {
            let agent = client
                .agent(openai::GPT_4O)
                .preamble(TWO_TOOL_STREAM_PREAMBLE)
                .tool(AlphaSignal)
                .tool(BetaSignal)
                .build();

            let mut stream = agent
                .stream_prompt(TWO_TOOL_STREAM_PROMPT)
                .max_turns(8)
                .await;
            let observation = collect_stream_observation(&mut stream).await;

            assert_two_tool_roundtrip_contract(
                &observation,
                &["lookup_harbor_label", "lookup_orchard_label"],
                &[ALPHA_SIGNAL_OUTPUT, BETA_SIGNAL_OUTPUT],
            );
        },
    )
    .await;
}

#[tokio::test]
async fn completions_api_raw_stream_emits_required_zero_arg_tool_call() {
    with_openai_completions_cassette(
        "completions_api/completions_api_raw_stream_emits_required_zero_arg_tool_call",
        |client| async move {
            let model = client.completion_model(openai::GPT_4O);
            let request = CompletionRequest {
                tools: vec![zero_arg_tool_definition("ping")],
                tool_choice: Some(ToolChoice::Required),
                ..CompletionRequest::from_prompt(REQUIRED_ZERO_ARG_TOOL_PROMPT)
            };
            let stream = model.stream(request).await.expect("stream should start");

            assert_stream_contains_zero_arg_tool_call_named(stream, "ping", true).await;
        },
    )
    .await;
}

#[tokio::test]
async fn completions_api_raw_stream_accepts_null_tool_calls_delta() {
    with_openai_completions_cassette(
        "completions_api/completions_api_raw_stream_accepts_null_tool_calls_delta",
        |client| async move {
            let model = client.completion_model(openai::GPT_4O);
            let request =
                CompletionRequest::from_prompt("Reply with exactly: cassette null tool calls ok");

            let observation = collect_raw_stream_observation(
                model
                    .stream(request)
                    .await
                    .expect("raw completions api stream should start"),
            )
            .await;

            assert!(
                observation.tool_calls.is_empty(),
                "null tool_calls deltas should not emit tool calls: {:?}",
                observation.tool_calls
            );
            assert_raw_stream_text_contains(&observation, &["cassette null tool calls ok"]);
        },
    )
    .await;
}

#[tokio::test]
async fn completions_api_raw_stream_surfaces_two_distinct_tool_calls_before_text() {
    with_openai_completions_cassette(
        "completions_api/completions_api_raw_stream_surfaces_two_distinct_tool_calls_before_text",
        |client| async move {
            let model = client.completion_model(openai::GPT_4O);
            let request = CompletionRequest {
                tools: vec![
                    rig::tool::tool_definition(&AlphaSignal),
                    rig::tool::tool_definition(&BetaSignal),
                ],
                ..CompletionRequest::with_history(
                    Some(TWO_TOOL_STREAM_PREAMBLE),
                    Vec::new(),
                    TWO_TOOL_STREAM_PROMPT,
                )
            };

            let observation = collect_raw_stream_observation(
                model
                    .stream(request)
                    .await
                    .expect("raw completions api stream should start"),
            )
            .await;

            assert_raw_stream_contains_distinct_tool_calls_before_text(
                &observation,
                &["lookup_harbor_label", "lookup_orchard_label"],
            );
        },
    )
    .await;
}

#[tokio::test]
async fn completions_api_stream_emits_tool_call_before_later_text() {
    with_openai_completions_cassette(
        "completions_api/completions_api_stream_emits_tool_call_before_later_text",
        |client| async move {
            let agent = client
                .agent(openai::GPT_4O)
                .preamble(ORDERED_TOOL_STREAM_PREAMBLE)
                .tool(AlphaSignal)
                .build();

            let mut stream = agent
                .stream_prompt(ORDERED_TOOL_STREAM_PROMPT)
                .max_turns(5)
                .await;
            let observation = collect_stream_observation(&mut stream).await;

            assert_tool_call_precedes_later_text(
                &observation,
                "lookup_harbor_label",
                &[ALPHA_SIGNAL_OUTPUT],
            );
        },
    )
    .await;
}

#[tokio::test]
async fn completions_api_raw_followup_uses_tool_result_without_new_tool_calls() {
    with_openai_completions_cassette(
        "completions_api/completions_api_raw_followup_uses_tool_result_without_new_tool_calls",
        |client| async move {
            let model = client.completion_model(openai::GPT_4O);
            let request = CompletionRequest {
                tools: vec![rig::tool::tool_definition(&AlphaSignal)],
                ..CompletionRequest::with_history(
                    Some(ORDERED_TOOL_STREAM_PREAMBLE),
                    Vec::new(),
                    ORDERED_TOOL_STREAM_PROMPT,
                )
            };

            let first_turn = collect_raw_stream_observation(
                model
                    .stream(request)
                    .await
                    .expect("raw completions api stream should start"),
            )
            .await;

            assert_raw_stream_tool_call_precedes_text(&first_turn, "lookup_harbor_label");

            let tool_call = first_turn
                .tool_calls
                .iter()
                .find(|tool_call| tool_call.function.name == "lookup_harbor_label")
                .cloned()
                .expect("raw completions api stream should yield lookup_harbor_label");
            let assistant_message = Message::Assistant {
                id: None,
                content: OneOrMany::one(AssistantContent::ToolCall(tool_call.clone())),
            };
            let tool_result_message =
                Message::tool_result_with_call_id(tool_call.id, tool_call.call_id, ALPHA_SIGNAL_OUTPUT);
            let followup_request = CompletionRequest::with_history(
                Some("Use the provided tool result and answer directly."),
                vec![assistant_message, tool_result_message],
                "Now reply in one short sentence using the provided tool result. Do not call any tools.",
            );

            let second_turn = collect_raw_stream_observation(
                model
                    .stream(followup_request)
                    .await
                    .expect("raw completions api followup stream should start"),
            )
            .await;

            assert!(
                second_turn.tool_calls.is_empty(),
                "follow-up raw completions api stream should not emit fresh tool calls, saw {:?}",
                second_turn
                    .tool_calls
                    .iter()
                    .map(|tool_call| tool_call.function.name.as_str())
                    .collect::<Vec<_>>()
            );
            let alpha_signal_markers = ALPHA_SIGNAL_OUTPUT.split('-').collect::<Vec<_>>();
            assert_raw_stream_text_contains(&second_turn, &alpha_signal_markers);
        },
    )
    .await;
}


#[tokio::test]
async fn completions_api_pure_functions_replay_recorded_request() {
    // Replays the exchange recorded by `completions_api_agent_prompt`
    // through the pure free-function path (functions::complete over
    // HttpRuntime). The cassette server only serves the recorded response
    // when the incoming request matches the recording — so a passing test
    // proves `functions::build_request_body` emits byte-identical request
    // bytes to the classic agent path, and exercises parse_response
    // end to end.
    use std::panic::AssertUnwindSafe;

    use futures::FutureExt;
    use rig::OneOrMany;
    use rig::completion::CompletionRequest;
    use rig::http_runtime::HttpRuntime;
    use rig::message::Message;

    use crate::cassettes::ProviderCassette;

    let cassette = ProviderCassette::start(
        "openai",
        "completions_api/completions_api_agent_prompt",
        "https://api.openai.com/v1",
    )
    .await;
    let config = openai::functions::Config::new(openai::GPT_4O)
        .with_api_key(cassette.api_key("OPENAI_API_KEY"))
        .with_base_url(cassette.base_url());
    let runtime = HttpRuntime::new();

    let request = CompletionRequest {
        model: None,
        preamble: None,
        chat_history: OneOrMany::many(vec![
            Message::system("You are a helpful assistant."),
            Message::user("Hello world!"),
        ])
        .expect("history should be non-empty"),
        documents: Vec::new(),
        tools: Vec::new(),
        temperature: None,
        max_tokens: None,
        tool_choice: None,
        additional_params: None,
        output_schema: None,
        record_telemetry_content: false,
    };

    let result = AssertUnwindSafe(async {
        let response = openai::functions::complete(&config, &runtime, request)
            .await
            .expect("pure-function completion should replay the recording");
        assert_eq!(response.provider, "openai");
        let text = assistant_text_response(&response.choice)
            .expect("response should contain assistant text");
        assert_nonempty_response(&text);
    })
    .catch_unwind()
    .await;
    cassette.finish_after_test(result).await;
}
