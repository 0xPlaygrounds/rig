//! Perplexity cassette coverage for regressions found during the #2040 provider migration.

use rig::OneOrMany;
use rig::client::CompletionClient;
use rig::completion::CompletionModel;
use rig::message::{AssistantContent, Message, ToolCall, ToolChoice, ToolFunction, UserContent};
use rig::providers::perplexity;
use serde_json::json;

use crate::support::{
    SmokeStructuredOutput, assert_contains_any_case_insensitive, assert_nonempty_response,
    assistant_text_response, zero_arg_tool_definition,
};

use super::super::support::with_perplexity_cassette;

#[tokio::test]
async fn text_only_content_parts_are_flattened() {
    with_perplexity_cassette(
        "migration_pain_points/text_only_content_parts_are_flattened",
        |client| async move {
            let model = client.completion_model(perplexity::SONAR);
            let prompt = Message::User {
                content: OneOrMany::many(vec![
                    UserContent::text("First text part: amber."),
                    UserContent::text("Second text part: rig."),
                ])
                .expect("prompt should contain text parts"),
            };

            let response = model
                .completion_request(prompt)
                .preamble("Reply with the two words joined by a hyphen.".to_string())
                .max_tokens(32)
                .additional_params(json!({"search_context_size": "low"}))
                .send()
                .await
                .expect("Perplexity should accept flattened text-only content parts");

            let text = assistant_text_response(&response.choice)
                .expect("response should contain assistant text");
            assert_contains_any_case_insensitive(&text, &["amber-rig", "amber"]);
        },
    )
    .await;
}

#[tokio::test]
async fn tool_exchange_history_is_stripped_and_remerged() {
    with_perplexity_cassette(
        "migration_pain_points/tool_exchange_history_is_stripped_and_remerged",
        |client| async move {
            let model = client.completion_model(perplexity::SONAR);
            let tool_call = ToolCall::new(
                "call_amber".to_string(),
                ToolFunction::new("lookup_code_word".to_string(), json!({})),
            );

            let response = model
                .completion_request("What code word appears in the surviving conversation history?")
                .preamble("Answer in one short sentence.".to_string())
                .message(Message::user("Remember this code word: amber-rig."))
                .message(Message::Assistant {
                    id: None,
                    content: OneOrMany::one(AssistantContent::ToolCall(tool_call)),
                })
                .message(Message::tool_result("call_amber", "tool result: amber-rig"))
                .message(Message::user(
                    "Use the history, not web search, if possible.",
                ))
                .max_tokens(32)
                .additional_params(json!({"search_context_size": "low"}))
                .send()
                .await
                .expect("Perplexity should accept sanitized tool-exchange history");

            let text = assistant_text_response(&response.choice)
                .expect("response should contain assistant text");
            assert_contains_any_case_insensitive(&text, &["amber-rig", "amber"]);
        },
    )
    .await;
}

#[tokio::test]
async fn unsupported_tools_and_multi_name_tool_choice_are_dropped() {
    with_perplexity_cassette(
        "migration_pain_points/unsupported_tools_and_multi_name_tool_choice_are_dropped",
        |client| async move {
            let model = client.completion_model(perplexity::SONAR);
            let response = model
                .completion_request("Reply with exactly: tools dropped ok")
                .preamble("Follow the user's requested exact reply.".to_string())
                .tool(zero_arg_tool_definition("lookup_alpha"))
                .tool(zero_arg_tool_definition("lookup_beta"))
                .tool_choice(ToolChoice::Specific {
                    function_names: vec!["lookup_alpha".to_string(), "lookup_beta".to_string()],
                })
                .max_tokens(32)
                .additional_params(json!({"search_context_size": "low"}))
                .send()
                .await
                .expect(
                    "unsupported tools and multi-name tool choice should be dropped before validation",
                );

            let text = assistant_text_response(&response.choice)
                .expect("response should contain assistant text");
            assert_contains_any_case_insensitive(&text, &["tools dropped ok"]);
        },
    )
    .await;
}

#[tokio::test]
async fn output_schema_is_dropped_instead_of_sent_as_response_format() {
    with_perplexity_cassette(
        "migration_pain_points/output_schema_is_dropped_instead_of_sent_as_response_format",
        |client| async move {
            let model = client.completion_model(perplexity::SONAR);
            let response = model
                .completion_request(
                    "Name one Rust programming language benefit in a short sentence.",
                )
                .preamble("Answer briefly.".to_string())
                .output_schema(schemars::schema_for!(SmokeStructuredOutput))
                .max_tokens(48)
                .additional_params(json!({"search_context_size": "low"}))
                .send()
                .await
                .expect("Perplexity should ignore unsupported response_format mapping");

            let text = assistant_text_response(&response.choice)
                .expect("response should contain assistant text");
            assert_nonempty_response(&text);
        },
    )
    .await;
}
