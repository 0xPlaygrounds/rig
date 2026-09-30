//! Migrated from `examples/openai_agent_completions_api.rs`.
use rig::message::{AssistantContent, Message, ToolChoice, ToolResultContent, UserContent};
use rig::providers::openai;

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
use rig::completion::CompletionRequest;

#[tokio::test]
async fn completions_api_agent_prompt() {
    with_openai_completions_cassette(
        "completions_api/completions_api_agent_prompt",
        |client| async move {
            let agent = rig::AgentBuilder::new(client.completion(openai::GPT_4O))
                .preamble("You are a helpful assistant.")
                .build();

            let response = agent
                .prompt("Hello world!")
                .await
                .expect("completions api prompt should succeed")
                .output;

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
            let model = client.chat(openai::GPT_4O);
            let request = CompletionRequest::new(RAW_TEXT_RESPONSE_PROMPT)
                .preamble(RAW_TEXT_RESPONSE_PREAMBLE.to_string());

            // The cassette records exactly one interaction, and one is all this
            // needs: the provider's own reply rides serialized on
            // `CompletionResponse::raw` beside the normalized view, so both
            // texts come off a single request.
            let response = model
                .call(request)
                .await
                .expect("completions api request should succeed");
            let reply: openai::completion::CompletionResponse =
                serde_json::from_value(response.raw.clone())
                    .expect("`raw` is the serialized openai::completion::CompletionResponse");
            let raw_text = reply
                .choices
                .iter()
                .filter_map(|choice| match &choice.message {
                    openai::completion::Message::Assistant { content, .. } => Some(content),
                    _ => None,
                })
                .flatten()
                .filter_map(|content| match content {
                    openai::completion::AssistantContent::Text { text } => Some(text.as_str()),
                    openai::completion::AssistantContent::Refusal { .. } => None,
                })
                .collect::<Vec<_>>()
                .join("\n");

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

#[tokio::test]
async fn completions_api_streams_two_tool_calls_before_final_answer() {
    with_openai_completions_cassette(
        "completions_api/completions_api_streams_two_tool_calls_before_final_answer",
        |client| async move {
            let agent = rig::AgentBuilder::new(client.completion(openai::GPT_4O))
                .preamble(TWO_TOOL_STREAM_PREAMBLE)
                .tool(AlphaSignal)
                .tool(BetaSignal)
                .build();

            let mut stream = agent.prompt(TWO_TOOL_STREAM_PROMPT).max_turns(8).stream();
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
            let model = client.chat(openai::GPT_4O);
            let request = CompletionRequest::new(REQUIRED_ZERO_ARG_TOOL_PROMPT)
                .tool(zero_arg_tool_definition("ping"))
                .tool_choice(ToolChoice::Required);
            let stream = model.stream(request).expect("stream should start");

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
            let model = client.chat(openai::GPT_4O);
            let request = CompletionRequest::new("Reply with exactly: cassette null tool calls ok");

            let observation = collect_raw_stream_observation(
                model
                    .stream(request)
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
            let model = client.chat(openai::GPT_4O);
            let request = CompletionRequest::new(TWO_TOOL_STREAM_PROMPT)
                .preamble(TWO_TOOL_STREAM_PREAMBLE.to_string())
                .tool(rig::tool::tool_definition(&AlphaSignal))
                .tool(rig::tool::tool_definition(&BetaSignal));

            let observation = collect_raw_stream_observation(
                model
                    .stream(request)
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
            let agent = rig::AgentBuilder::new(client.completion(openai::GPT_4O))
                .preamble(ORDERED_TOOL_STREAM_PREAMBLE)
                .tool(AlphaSignal)
                .build();

            let mut stream = agent
                .prompt(ORDERED_TOOL_STREAM_PROMPT)
                .max_turns(5)
                .stream();
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
            let model = client.chat(openai::GPT_4O);
            let request = CompletionRequest::new(ORDERED_TOOL_STREAM_PROMPT)
                .preamble(ORDERED_TOOL_STREAM_PREAMBLE.to_string())
                .tool(rig::tool::tool_definition(&AlphaSignal));

            let first_turn = collect_raw_stream_observation(
                model
                    .stream(request)
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
                content: vec![AssistantContent::ToolCall(tool_call.clone())],
            };
            let tool_result_message =
                Message::User {
        content: vec![UserContent::tool_result(tool_call.id.clone(), tool_call.function.name.clone(), vec![ToolResultContent::text(ALPHA_SIGNAL_OUTPUT)])],
    };
            let followup_request = CompletionRequest::new(
                    "Now reply in one short sentence using the provided tool result. Do not call any tools.",
                )
                .preamble("Use the provided tool result and answer directly.")
                .message(assistant_message)
                .message(tool_result_message);

            let second_turn = collect_raw_stream_observation(
                model
                    .stream(followup_request)
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

/// `updates()` over two parallel tool calls on Chat Completions: the stream
/// keeps the index contract, and each call's argument JSON and name are the
/// ones its recorded fragments assemble, read from the cassette's frames.
#[tokio::test]
async fn completions_api_updates_keep_parallel_tool_calls_in_place() {
    with_openai_completions_cassette(
        "completions_api/completions_api_raw_stream_surfaces_two_distinct_tool_calls_before_text",
        |client| async move {
            let model = client.chat(openai::GPT_4O);
            let request = CompletionRequest::new(TWO_TOOL_STREAM_PROMPT)
                .preamble(TWO_TOOL_STREAM_PREAMBLE)
                .tool(rig::tool::tool_definition(&AlphaSignal))
                .tool(rig::tool::tool_definition(&BetaSignal));
            let stream = model.stream(request).expect("the stream should start");
            let updates = rig_test_support::updates::collect_updates(stream).await;
            let (_, parts) = rig_test_support::updates::assert_update_contract(&updates);

            let frames = crate::cassettes::recorded_sse_json_frames(
                "openai",
                "completions_api/completions_api_raw_stream_surfaces_two_distinct_tool_calls_before_text",
            );
            let mut recorded: std::collections::BTreeMap<u64, (String, String)> =
                std::collections::BTreeMap::new();
            let mut recorded_text = String::new();
            for frame in &frames {
                let delta = &frame["choices"][0]["delta"];
                if let Some(text) = delta["content"].as_str() {
                    recorded_text.push_str(text);
                }
                for call in delta["tool_calls"].as_array().into_iter().flatten() {
                    let entry = recorded
                        .entry(call["index"].as_u64().unwrap_or_default())
                        .or_default();
                    if let Some(name) = call["function"]["name"].as_str() {
                        entry.0.push_str(name);
                    }
                    if let Some(arguments) = call["function"]["arguments"].as_str() {
                        entry.1.push_str(arguments);
                    }
                }
            }
            assert!(recorded.len() >= 2, "the cassette records parallel calls");

            let delivered: Vec<(String, serde_json::Value)> = parts
                .iter()
                .filter_map(|part| match &part.part {
                    rig::message::AssistantContent::ToolCall(call) => Some((
                        call.function.name.to_string(),
                        serde_json::from_str(&part.text).expect("argument JSON"),
                    )),
                    _ => None,
                })
                .collect();
            let mut expected: Vec<(String, serde_json::Value)> = recorded
                .into_values()
                .map(|(name, arguments)| {
                    (name, serde_json::from_str(&arguments).expect("recorded argument JSON"))
                })
                .collect();
            let mut delivered_sorted = delivered.clone();
            delivered_sorted.sort_by(|a, b| a.0.cmp(&b.0));
            expected.sort_by(|a, b| a.0.cmp(&b.0));
            assert_eq!(delivered_sorted, expected);

            let text: String = parts
                .iter()
                .filter(|part| part.kind == rig::streaming::PartKind::Text)
                .map(|part| part.text.as_str())
                .collect();
            assert_eq!(text, recorded_text);
        },
    )
    .await;
}
