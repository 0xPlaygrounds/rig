//! OpenAI Responses over WebSocket mode, recorded against
//! `wss://api.openai.com/v1/responses`: each turn is the transport's
//! `response.create` and the events OpenAI answered it with. Every turn
//! sends `store: false`; chained turns continue the connection's own cache.

use futures::StreamExt;
use rig::completion::{AssistantContent, CompletionRequest, CompletionResponse};
use rig::providers::openai;
use rig::streaming::Item;
use serde_json::json;

use super::super::support::with_openai_websocket_turn_cassette;
use crate::support::{Adder, Subtract, TOOLS_PREAMBLE, TOOLS_PROMPT};

fn unstored(prompt: &str) -> CompletionRequest {
    let mut request = CompletionRequest::new(prompt);
    request.additional_params = Some(json!({ "store": false }));
    request
}

fn text(response: &CompletionResponse) -> String {
    response
        .choice
        .iter()
        .filter_map(|content| match content {
            AssistantContent::Text(text) => Some(text.text.as_str()),
            _ => None,
        })
        .collect()
}

/// A warmup, then two turns queued at once on one chaining connection: the
/// second, streamed, waits for the first and continues its response.
#[tokio::test]
async fn queued_turns_chain_after_a_warmup() {
    with_openai_websocket_turn_cassette(
        "websocket_turns/queued_turns_chain_after_a_warmup",
        |client, backend| async move {
            let socket = client
                .openai
                .responses(openai::GPT_5_4_MINI)
                .responses_websocket()
                .chaining()
                .connect_with(&backend)
                .await
                .expect("the connection opens");
            let mut warmup = unstored("Answer every question with a number only.");
            warmup.chat_history.insert(
                0,
                rig::completion::Message::system("You are a terse arithmetic assistant.").into(),
            );
            let warmed = socket.warmup(warmup).await.expect("the warmup prepares");
            assert!(warmed.starts_with("resp_"), "{warmed}");

            let streamed = async {
                let mut stream = socket
                    .stream(unstored("Now double it."))
                    .expect("the turn opens");
                let mut events = 0;
                while let Some(item) = stream.next().await {
                    if let Item::Event(_) = item.expect("the turn streams") {
                        events += 1;
                    }
                }
                (events, stream.finish().await.expect("the turn folds"))
            };
            let (first, (events, second)) = tokio::join!(
                socket.call(unstored("What is the smallest prime?")),
                streamed
            );
            let first = first.expect("the first turn completes");

            assert!(text(&first).contains('2'), "{}", text(&first));
            assert!(events > 1, "the second turn streamed {events} event(s)");
            assert!(text(&second).contains('4'), "{}", text(&second));
            assert_eq!(
                socket.transport.last_response_id(),
                second.origin.response_id
            );
            socket
                .transport
                .close()
                .await
                .expect("the connection closes");
        },
    )
    .await;
}

/// A turn dropped after its first event is read to its end before the next
/// turn is sent, which gets its own answer.
#[tokio::test]
async fn a_dropped_turn_is_drained_before_the_next() {
    with_openai_websocket_turn_cassette(
        "websocket_turns/a_dropped_turn_is_drained_before_the_next",
        |client, backend| async move {
            let socket = client
                .openai
                .responses(openai::GPT_5_4_MINI)
                .responses_websocket()
                .connect_with(&backend)
                .await
                .expect("the connection opens");
            let mut dropped = socket
                .stream(unstored("Count from one to twenty in words."))
                .expect("the turn opens");
            dropped
                .next()
                .await
                .expect("an item")
                .expect("the turn streams");
            drop(dropped);

            let next = socket
                .call(unstored("What is 3 + 4? Answer with the number only."))
                .await
                .expect("the next turn completes");
            assert!(text(&next).contains('7'), "{}", text(&next));
            socket
                .transport
                .close()
                .await
                .expect("the connection closes");
        },
    )
    .await;
}

/// An agent's tool loop over the connection: the model erases to the
/// agent's model, and with chaining off each turn carries the whole
/// conversation.
#[tokio::test]
async fn an_agent_tool_loop_runs_over_the_connection() {
    with_openai_websocket_turn_cassette(
        "websocket_turns/an_agent_tool_loop_runs_over_the_connection",
        |client, backend| async move {
            let socket = client
                .openai
                .responses(openai::GPT_5_4_MINI)
                .responses_websocket()
                .connect_with(&backend)
                .await
                .expect("the connection opens");
            let agent = rig::AgentBuilder::new(socket.clone())
                .preamble(TOOLS_PREAMBLE)
                .tool(Adder)
                .tool(Subtract)
                .additional_params(json!({ "store": false }))
                .default_max_turns(3)
                .build();

            let answer = agent
                .prompt(TOOLS_PROMPT)
                .await
                .expect("the tool loop completes")
                .output();

            crate::support::assert_mentions_expected_number(&answer, -3);
            socket
                .transport
                .close()
                .await
                .expect("the connection closes");
        },
    )
    .await;
}
