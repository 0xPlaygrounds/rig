//! Migrated from `examples/openai_agent_completions_api.rs`.

use rig::OneOrMany;
use rig::client::CompletionClient;
use rig::completion::CompletionModel;
use rig::completion::Prompt;
use rig::message::{AssistantContent, Message, ToolChoice};
use rig::providers::openai;
use rig::streaming::StreamingPrompt;
use rig::telemetry::ProviderResponseExt;
use rig::tool::Tool;

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
                .completion_model(openai::GPT_4O)
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

#[tokio::test]
async fn completions_api_raw_response_text_matches_normalized_choice_text() {
    with_openai_completions_cassette(
        "completions_api/completions_api_raw_response_text_matches_normalized_choice_text",
        |client| async move {
            let response = client
                .completion_model(openai::GPT_4O)
                .completion_request(RAW_TEXT_RESPONSE_PROMPT)
                .preamble(RAW_TEXT_RESPONSE_PREAMBLE.to_string())
                .send()
                .await
                .expect("raw completions api request should succeed");

            let normalized_text = assistant_text_response(&response.choice)
                .expect("normalized completions api response should contain assistant text");
            let raw_text = response
                .raw_response
                .get_text_response()
                .expect("raw completions api response should contain assistant text");

            assert_nonempty_response(&normalized_text);
            assert_nonempty_response(&raw_text);
            assert_contains_all_case_insensitive(&raw_text, &["cedar", "maple"]);
            assert_eq!(raw_text.trim(), normalized_text.trim());
        },
    )
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
                .multi_turn(8)
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
            let request = model
                .completion_request(REQUIRED_ZERO_ARG_TOOL_PROMPT)
                .tool(zero_arg_tool_definition("ping"))
                .tool_choice(ToolChoice::Required)
                .build();
            let stream = model.stream(request).await.expect("stream should start");

            assert_stream_contains_zero_arg_tool_call_named(stream, "ping", true).await;
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
            let request = model
                .completion_request(TWO_TOOL_STREAM_PROMPT)
                .preamble(TWO_TOOL_STREAM_PREAMBLE.to_string())
                .tool(AlphaSignal.definition(String::new()).await)
                .tool(BetaSignal.definition(String::new()).await)
                .build();

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
                .multi_turn(5)
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
            let request = model
                .completion_request(ORDERED_TOOL_STREAM_PROMPT)
                .preamble(ORDERED_TOOL_STREAM_PREAMBLE.to_string())
                .tool(AlphaSignal.definition(String::new()).await)
                .build();

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
            let followup_request = model
                .completion_request(
                    "Now reply in one short sentence using the provided tool result. Do not call any tools.",
                )
                .preamble("Use the provided tool result and answer directly.".to_string())
                .message(assistant_message)
                .message(tool_result_message)
                .build();

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
            assert_raw_stream_text_contains(&second_turn, &[ALPHA_SIGNAL_OUTPUT]);
        },
    )
    .await;
}
