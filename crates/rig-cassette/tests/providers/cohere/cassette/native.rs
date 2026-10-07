//! Cassette-backed coverage of Cohere's native chat API: documents with
//! citations, the tool plan and streamed tool calls, and a conversation that
//! moves between the native and Compatibility APIs.

use std::collections::HashMap;

use futures::StreamExt;
use rig::completion::{CompletionRequest, CompletionResponse, Cost, Document, Message};
use rig::message::{AssistantContent, Source, SourceLocation, ToolResultContent, UserContent};
use rig::providers::cohere::extension::{CohereExt, CohereExtras};
use rig::providers::cohere::{ChatRoute, CohereChat};
use rig::providers::ollama::extension::OllamaExt;
use rig::streaming::{Item, StreamEvent};
use rig::tool::Tool;
use serde_json::Value;

use super::super::{
    CASSETTE_MODEL,
    support::{IntegerSubtract, with_cohere_cassette},
};
use crate::support::{STREAMING_TOOLS_PREAMBLE, STREAMING_TOOLS_PROMPT};

fn documents() -> Vec<Document> {
    vec![
        Document {
            id: "harbor-record-1".to_owned(),
            text: "Beacon code amber-73 is assigned to Dock Seven.".to_owned(),
            additional_props: HashMap::from([("source".to_owned(), "harbor-registry".to_owned())]),
        },
        Document {
            id: "harbor-record-2".to_owned(),
            text: "Dock Seven closes for repairs every Tuesday.".to_owned(),
            additional_props: HashMap::new(),
        },
    ]
}

/// Each citation on `response`'s text blocks: the text it spans and its
/// sources.
fn citations(response: &CompletionResponse) -> Vec<(String, Vec<Source>)> {
    response
        .choice
        .iter()
        .filter_map(|block| match block {
            AssistantContent::Text(text) => Some(text),
            _ => None,
        })
        .flat_map(|text| {
            text.citations().iter().map(|citation| {
                let cited = citation.span.and(text.cited(citation)).unwrap_or_default();
                (cited.to_owned(), citation.sources.clone())
            })
        })
        .collect()
}

/// A request document cited by its id.
fn document(id: &str) -> Source {
    Source::new(SourceLocation::Document {
        index: None,
        id: Some(id.to_owned()),
        within: None,
    })
}

/// The billed units priced at Command A's $2.50 input and $10 output per
/// million tokens.
fn billed(input: f64, output: f64) -> Option<Cost> {
    Some(Cost::from_parts(
        input * 2.5 / 1e6,
        output * 10.0 / 1e6,
        0.0,
        0.0,
    ))
}

/// The events of a streamed reply, and the response it folds into.
async fn streamed(
    model: &rig::Model<CohereChat>,
    request: CompletionRequest,
) -> (Vec<StreamEvent>, CompletionResponse) {
    let mut stream = model.stream(request).expect("the stream opens");
    let mut events = Vec::new();
    while let Some(item) = stream.next().await {
        if let Item::Event(event) = item.expect("the stream yields no error") {
            events.push(event);
        }
    }
    let response = stream.finish().await.expect("the stream folds");
    (events, response)
}

/// On the `Auto` route a streamed answer grounded in documents cites them:
/// each citation spans the characters it quotes and names its source
/// document, and its JSON sits in the provider item of the text it cites.
/// The second citation arrives after the last text it spans. The cost
/// prices the billed units, not the larger token counts.
#[tokio::test]
async fn documents_ground_a_streamed_answer_with_citations() {
    with_cohere_cassette(
        "native/documents_ground_a_streamed_answer_with_citations",
        |client| async move {
            let mut model = client.completion(CASSETTE_MODEL);
            model.wire = model.wire.with_route(ChatRoute::Auto);
            let mut request =
                CompletionRequest::new("Which dock has beacon amber-73, and when does it close?")
                    .max_tokens(96);
            request.documents = documents();
            let (events, response) = streamed(&model, request).await;

            assert!(
                events
                    .iter()
                    .any(|event| matches!(event, StreamEvent::Text { .. })),
                "the answer streams as text: {events:?}"
            );
            assert_eq!(response.origin.api.as_str(), "cohere.chat");
            assert_eq!(
                citations(&response),
                [
                    ("Dock Seven".to_owned(), vec![document("harbor-record-1")]),
                    (
                        "closes for repairs every Tuesday.".to_owned(),
                        vec![document("harbor-record-2")]
                    ),
                ]
            );
            let item = response
                .choice
                .iter()
                .find_map(AssistantContent::native_item)
                .expect("the text keeps its item");
            assert_eq!(item["citations"].as_array().map(Vec::len), Some(2));
            assert_eq!(response.usage.input_tokens, Some(1696));
            assert_eq!(response.usage.cost, billed(45.0, 17.0));
            // The streamed reply's extras read what a unary reply's do.
            let extras = extras(&response);
            assert_eq!(
                extras.id.as_deref(),
                Some("6f0073cc-024b-4df8-866e-2d4712519fca")
            );
            assert_eq!(extras.finish_reason.as_deref(), Some("COMPLETE"));
            let billed = extras.billed_units.expect("billed units");
            assert_eq!(
                (billed.input_tokens, billed.output_tokens),
                (Some(45.0), Some(17.0))
            );
            let tokens = extras.tokens.expect("tokens");
            assert_eq!(
                (tokens.input_tokens, tokens.output_tokens),
                (Some(1696.0), Some(41.0))
            );
            assert_eq!(extras.cached_tokens, Some(112.0));
        },
    )
    .await;
}

/// On the native route a streamed tool turn plans as reasoning, streams its
/// call's arguments fragment by fragment, and replays its plan and call to
/// the native API on the next turn.
#[tokio::test]
async fn a_streamed_tool_plan_and_call_replay_natively() {
    with_cohere_cassette(
        "native/a_streamed_tool_plan_and_call_replay_natively",
        |client| async move {
            let mut model = client.completion(CASSETTE_MODEL);
            model.wire = model.wire.with_route(ChatRoute::Native);
            let subtract = rig::completion::ToolDefinition::new(
                rig::message::ToolName::new(IntegerSubtract::NAME).expect("a tool name"),
                IntegerSubtract.description(),
                IntegerSubtract.parameters(),
            );
            let first = CompletionRequest::new(STREAMING_TOOLS_PROMPT)
                .preamble(STREAMING_TOOLS_PREAMBLE)
                .tool(subtract.clone());
            let (events, turn) = streamed(&model, first.clone()).await;

            let fragments = events
                .iter()
                .filter(|event| matches!(event, StreamEvent::Arguments { .. }))
                .count();
            assert!(
                fragments > 1,
                "the call's arguments stream in fragments: {events:?}"
            );
            let plan = turn
                .choice
                .iter()
                .find_map(|block| match block {
                    AssistantContent::Reasoning(plan) => Some(plan),
                    _ => None,
                })
                .expect("the turn plans its call");
            assert!(!plan.text.is_empty());
            // The streamed reply's raw is the chat response, so its extras
            // read the plan where a unary reply states it.
            let turn_extras = extras(&turn);
            assert_eq!(turn_extras.tool_plan.as_deref(), Some(plan.text.as_str()));
            assert_eq!(turn_extras.finish_reason.as_deref(), Some("TOOL_CALL"));
            let call = turn
                .choice
                .iter()
                .find_map(|block| match block {
                    AssistantContent::ToolCall(call) => Some(call.clone()),
                    _ => None,
                })
                .expect("the turn calls the tool");
            assert_eq!(call.function.name.as_str(), "subtract");

            let answer = call.result(vec![ToolResultContent::text("-3")]);
            let mut next = first;
            next.chat_history
                .push(turn.message().expect("an assistant turn"));
            next.chat_history.push(Message::User {
                content: vec![UserContent::ToolResult(answer)],
            });
            let reply = model.call(next).await.expect("the next turn succeeds");
            let text: String = reply
                .choice
                .iter()
                .filter_map(|block| match block {
                    AssistantContent::Text(text) => Some(text.text.as_str()),
                    _ => None,
                })
                .collect();
            assert!(text.contains("-3"), "{text}");
            // The answer cites the tool's output by its id.
            assert_eq!(
                citations(&reply),
                [(
                    "-3".to_owned(),
                    vec![Source::new(SourceLocation::ToolOutput {
                        id: "subtract_d41p5xnhmncp:0".to_owned()
                    })]
                )]
            );
        },
    )
    .await;
}

/// The typed extras of `reply`, a Cohere reply.
fn extras(reply: &CompletionResponse) -> CohereExtras {
    reply
        .extras::<CohereExt>()
        .expect("a Cohere reply")
        .expect("the extras read")
}

/// A conversation the `Auto` route moves between the APIs replays on each:
/// a native turn grounded in documents, a Compatibility API turn without
/// them that reads the first canonically, and a native turn that sends the
/// first back with its citations and the second canonically. Each reply's
/// typed extras read what its route returns.
#[tokio::test]
async fn a_conversation_switching_routes_replays() {
    with_cohere_cassette(
        "native/a_conversation_switching_routes_replays",
        |client| async move {
            let mut model = client.completion(CASSETTE_MODEL);
            model.wire = model.wire.with_route(ChatRoute::Auto);
            let ask = |history: &[Message], prompt: &str, grounded: bool| {
                let mut request = CompletionRequest::new(prompt).max_tokens(64);
                request.chat_history.splice(0..0, history.iter().cloned());
                if grounded {
                    request.documents = documents();
                }
                request
            };
            let mut history = Vec::new();

            let question = "Which dock has beacon amber-73?";
            let first = model
                .call(ask(&history, question, true))
                .await
                .expect("the native turn succeeds");
            assert_eq!(first.origin.api.as_str(), "cohere.chat");
            assert_eq!(
                citations(&first),
                [("Dock Seven".to_owned(), vec![document("harbor-record-1")])]
            );
            assert_eq!(first.usage.cost, billed(39.0, 9.0));
            let first_extras = extras(&first);
            assert_eq!(
                first_extras.id.as_deref(),
                Some("203d7b74-eb9e-4de8-a159-7684818071fe")
            );
            assert_eq!(first_extras.finish_reason.as_deref(), Some("COMPLETE"));
            let billed = first_extras.billed_units.expect("billed units");
            assert_eq!(
                (
                    billed.input_tokens,
                    billed.output_tokens,
                    billed.search_units
                ),
                (Some(39.0), Some(9.0), None)
            );
            let tokens = first_extras.tokens.expect("tokens");
            assert_eq!(
                (tokens.input_tokens, tokens.output_tokens),
                (Some(1690.0), Some(23.0))
            );
            assert_eq!(first_extras.cached_tokens, Some(704.0));
            assert_eq!(first_extras.tool_plan, None);
            assert_eq!(first_extras.logprobs, None);
            assert!(first.extras::<OllamaExt>().is_none());
            history.push(Message::user(question));
            history.push(first.message().expect("an assistant turn"));

            let question = "Repeat the dock's name in capital letters.";
            let second = model
                .call(ask(&history, question, false))
                .await
                .expect("the Compatibility API turn succeeds");
            assert_eq!(second.origin.api.as_str(), "openai.chat");
            // A Compatibility API reply carries only the id.
            let second_extras = extras(&second);
            assert_eq!(
                second_extras.id.as_deref(),
                Some("8bb2c505-4106-4988-9a52-24fb0bb78cd7")
            );
            assert_eq!(second_extras.finish_reason, None);
            assert_eq!(second_extras.billed_units, None);
            assert_eq!(second_extras.tokens, None);
            assert_eq!(second_extras.cached_tokens, None);
            history.push(Message::user(question));
            history.push(second.message().expect("an assistant turn"));

            let third = model
                .call(ask(&history, "When does that dock close?", true))
                .await
                .expect("the native turn after a switch succeeds");
            assert_eq!(third.origin.api.as_str(), "cohere.chat");
            let third_extras = extras(&third);
            assert_eq!(
                third_extras.id.as_deref(),
                Some("bb085c96-3b3f-4531-b6d2-724501c09272")
            );
            assert_eq!(
                third_extras
                    .billed_units
                    .and_then(|units| units.input_tokens),
                Some(68.0)
            );
            assert_eq!(
                third_extras.tokens.and_then(|tokens| tokens.input_tokens),
                Some(1735.0)
            );
            assert_eq!(third_extras.cached_tokens, Some(112.0));
            let text = third
                .choice
                .iter()
                .filter_map(|block| match block {
                    AssistantContent::Text(text) => Some(text.text.to_lowercase()),
                    _ => None,
                })
                .collect::<String>();
            assert!(text.contains("tuesday"), "{text}");
            let raw: &Value = &third.raw;
            assert!(raw.get("message").is_some(), "{raw}");
        },
    )
    .await;
}

/// The Compatibility route keeps documents off the native API: they reach
/// the model as text in the history, and the answer has no citations.
#[tokio::test]
async fn the_compatibility_route_sends_documents_as_text() {
    with_cohere_cassette(
        "native/the_compatibility_route_sends_documents_as_text",
        |client| async move {
            let mut model = client.completion(CASSETTE_MODEL);
            model.wire = model.wire.with_route(ChatRoute::Compatibility);
            let mut request = CompletionRequest::new(
                "Which dock has beacon amber-73? Answer with the dock's name only.",
            )
            .max_tokens(64);
            request.documents = documents();
            let response = model.call(request).await.expect("the request succeeds");
            assert_eq!(response.origin.api.as_str(), "openai.chat");
            assert!(citations(&response).is_empty(), "{:?}", response.choice);
            let text = response
                .choice
                .iter()
                .filter_map(|block| match block {
                    AssistantContent::Text(text) => Some(text.text.to_lowercase()),
                    _ => None,
                })
                .collect::<String>();
            assert!(text.contains("seven"), "{text}");
        },
    )
    .await;
}
