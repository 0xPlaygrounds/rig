//! Cells shared by the OpenAI-compatible vendors' Responses routes: one
//! plain turn under typed options, and a tool round trip with `store: false`
//! whose reasoning the vendor returns as the item's own `reasoning_text`.

use rig::completion::{CompletionRequest, CompletionResponse, GenerationOptions, ToolDefinition};
use rig::driver::Model;
use rig::message::{Message, ToolResultContent, UserContent};
use rig::providers::openai::responses_api::wire::Responses;
use serde_json::{Value, json};

const ARITHMETIC: &str = "What is 17 * 23? Reply with only the number.";
const WEATHER: &str = "What is the weather in Paris? Use the tool, then answer in one sentence.";

/// One turn under `options`: the model answers the arithmetic question.
pub(crate) async fn plain_turn(
    model: &Model<Responses>,
    options: GenerationOptions,
) -> CompletionResponse {
    let reply = model
        .call(
            CompletionRequest::new(ARITHMETIC)
                .options(options)
                .max_tokens(512),
        )
        .await
        .expect("the turn answers");
    assert!(reply.text().contains("391"), "{:?}", reply.choice);
    reply
}

fn weather_tool() -> ToolDefinition {
    ToolDefinition {
        name: rig::message::ToolName::new("get_weather").expect("tool name"),
        description: "Look up the current weather for a city.".to_owned(),
        parameters: json!({
            "type": "object",
            "properties": { "city": { "type": "string" } },
            "required": ["city"],
        }),
        annotations: Default::default(),
    }
}

/// A tool call and its answer, both turns sent with `store: false` under
/// `options`; returns the second reply.
pub(crate) async fn stateless_tool_round_trip(
    model: &Model<Responses>,
    options: GenerationOptions,
) -> CompletionResponse {
    let stateless = json!({ "store": false });
    let first = model
        .call(
            CompletionRequest::new(WEATHER)
                .tool(weather_tool())
                .options(options.clone())
                .additional_params(stateless.clone())
                .max_tokens(1024),
        )
        .await
        .expect("turn 1 answers");
    let call = first
        .tool_calls()
        .next()
        .cloned()
        .expect("turn 1 calls the tool");
    let mut history = vec![Message::user(WEATHER)];
    history.extend(first.message());
    history.push(Message::from(UserContent::tool_result(
        call.id.clone(),
        call.function.name.clone(),
        vec![ToolResultContent::text("Sunny, 21 °C.")],
    )));
    let second = model
        .call(
            CompletionRequest::from(history)
                .tool(weather_tool())
                .options(options)
                .additional_params(stateless)
                .max_tokens(1024),
        )
        .await
        .expect("turn 2 answers from the tool result");
    assert!(second.text().contains("21"), "{:?}", second.choice);
    second
}

/// Turn 2's recorded request: turn 1's reasoning item goes back as the
/// vendor returned it, ahead of the call it led to, and the call keeps its
/// item id and is answered.
pub(crate) fn assert_reasoning_replayed(provider: &str, scenario: &str) {
    let recorded = crate::cassettes::recorded_interaction_bodies(provider, scenario);
    let bodies: Vec<(Value, Value)> = recorded
        .iter()
        .map(|(request, response)| {
            (
                serde_json::from_str(request).expect("request is JSON"),
                serde_json::from_str(response).expect("response is JSON"),
            )
        })
        .collect();
    let [(first, first_reply), (second, _)] = bodies.as_slice() else {
        panic!("two turns recorded: {}", bodies.len());
    };
    assert_eq!(first["store"], false);
    assert_eq!(second["store"], false);
    assert!(first.get("include").is_none(), "{first}");
    let output = first_reply["output"].as_array().expect("turn 1 output");
    let reasoning = output
        .iter()
        .find(|item| item["type"] == "reasoning")
        .expect("turn 1 reasons");
    assert!(
        reasoning["content"]
            .as_array()
            .is_some_and(|content| !content.is_empty()),
        "{reasoning}"
    );
    let input = second["input"].as_array().expect("turn 2 input");
    let at = |item: &Value| input.iter().position(|sent| sent == item);
    let reasoning_at = at(reasoning).expect("the reasoning item goes back as returned");
    let call = output
        .iter()
        .find(|item| item["type"] == "function_call")
        .expect("turn 1 calls");
    let call_at = input
        .iter()
        .position(|item| item["type"] == "function_call")
        .expect("the call goes back");
    assert!(reasoning_at < call_at, "{input:?}");
    assert_eq!(input[call_at]["id"], call["id"], "{input:?}");
    assert_eq!(input[call_at + 1]["type"], "function_call_output");
}
