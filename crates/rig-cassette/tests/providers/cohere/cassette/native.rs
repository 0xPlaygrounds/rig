//! Cassette-backed coverage of Cohere's native chat API: documents with
//! citations, the tool plan and streamed tool calls, and a conversation that
//! moves between the native and Compatibility APIs.

use std::collections::HashMap;

use futures::StreamExt;
use rig::completion::{CompletionRequest, CompletionResponse, Document, Message};
use rig::message::{AssistantContent, ToolResultContent, UserContent};
use rig::providers::cohere::{ChatRoute, CohereChat};
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

/// The document ids the citations on `response`'s text blocks name.
fn cited_documents(response: &CompletionResponse) -> Vec<String> {
    response
        .choice
        .iter()
        .filter(|block| matches!(block, AssistantContent::Text(_)))
        .filter_map(AssistantContent::native_item)
        .flat_map(|item| item["citations"].as_array().cloned().unwrap_or_default())
        .flat_map(|citation| citation["sources"].as_array().cloned().unwrap_or_default())
        .filter_map(|source| source["id"].as_str().map(str::to_owned))
        .collect()
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

/// A streamed answer grounded in documents cites them: each citation, with
/// its source document, sits in the provider item of the text it cites.
#[tokio::test]
async fn documents_ground_a_streamed_answer_with_citations() {
    with_cohere_cassette(
        "native/documents_ground_a_streamed_answer_with_citations",
        |client| async move {
            let model = client.completion(CASSETTE_MODEL);
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
            let cited = cited_documents(&response);
            assert!(
                cited.iter().any(|id| id.starts_with("harbor-record-")),
                "the answer cites the documents: {:?}",
                response.choice
            );
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
        },
    )
    .await;
}

/// A conversation that moves between the APIs replays on each: a native
/// turn grounded in documents, a Compatibility API turn without them that
/// reads the first canonically, and a native turn that sends the first
/// back with its citations and the second canonically.
#[tokio::test]
async fn a_conversation_switching_routes_replays() {
    with_cohere_cassette(
        "native/a_conversation_switching_routes_replays",
        |client| async move {
            let model = client.completion(CASSETTE_MODEL);
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
            assert!(!cited_documents(&first).is_empty(), "{:?}", first.choice);
            history.push(Message::user(question));
            history.push(first.message().expect("an assistant turn"));

            let question = "Repeat the dock's name in capital letters.";
            let second = model
                .call(ask(&history, question, false))
                .await
                .expect("the Compatibility API turn succeeds");
            assert_eq!(second.origin.api.as_str(), "openai.chat");
            history.push(Message::user(question));
            history.push(second.message().expect("an assistant turn"));

            let third = model
                .call(ask(&history, "When does that dock close?", true))
                .await
                .expect("the native turn after a switch succeeds");
            assert_eq!(third.origin.api.as_str(), "cohere.chat");
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
            assert!(
                cited_documents(&response).is_empty(),
                "{:?}",
                response.choice
            );
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
