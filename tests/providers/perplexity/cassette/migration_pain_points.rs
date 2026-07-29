//! Perplexity cassette coverage for regressions found during the #2040 provider migration.

use rig::OneOrMany;
use rig::completion::{CompletionModel, CompletionRequest};
use rig::message::{AssistantContent, Message, ToolCall, ToolChoice, ToolFunction, UserContent};
use rig::prelude::*;
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

            let request = CompletionRequest {
                max_tokens: Some(32),
                additional_params: Some(json!({"search_context_size": "low"})),
                ..CompletionRequest::with_history(
                    Some("Reply with the two words joined by a hyphen."),
                    Vec::new(),
                    prompt,
                )
            };
            let response = model
                .completion(request)
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

            let request = CompletionRequest {
                max_tokens: Some(32),
                additional_params: Some(json!({"search_context_size": "low"})),
                ..CompletionRequest::with_history(
                    Some("Answer in one short sentence."),
                    vec![
                        Message::user("Remember this code word: amber-rig."),
                        Message::Assistant {
                            id: None,
                            content: OneOrMany::one(AssistantContent::ToolCall(tool_call)),
                        },
                        Message::tool_result("call_amber", "tool result: amber-rig"),
                        Message::user("Use the history, not web search, if possible."),
                    ],
                    "What code word appears in the surviving conversation history?",
                )
            };
            let response = model
                .completion(request)
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
            let request = CompletionRequest {
                tools: vec![
                    zero_arg_tool_definition("lookup_alpha"),
                    zero_arg_tool_definition("lookup_beta"),
                ],
                tool_choice: Some(ToolChoice::Specific {
                    function_names: vec!["lookup_alpha".to_string(), "lookup_beta".to_string()],
                }),
                max_tokens: Some(32),
                additional_params: Some(json!({"search_context_size": "low"})),
                ..CompletionRequest::with_history(
                    Some("Follow the user's requested exact reply."),
                    Vec::new(),
                    "Reply with exactly: tools dropped ok",
                )
            };
            let response = model
                .completion(request)
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
            let request = CompletionRequest {
                output_schema: Some(schemars::schema_for!(SmokeStructuredOutput)),
                max_tokens: Some(48),
                additional_params: Some(json!({"search_context_size": "low"})),
                ..CompletionRequest::with_history(
                    Some("Answer briefly."),
                    Vec::new(),
                    "Name one Rust programming language benefit in a short sentence.",
                )
            };
            let response = model
                .completion(request)
                .await
                .expect("Perplexity should ignore unsupported response_format mapping");

            let text = assistant_text_response(&response.choice)
                .expect("response should contain assistant text");
            assert_nonempty_response(&text);
        },
    )
    .await;
}
