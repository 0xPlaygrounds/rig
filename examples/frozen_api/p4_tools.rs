use rig::completion::{CompletionRequest, ToolDefinition};
use rig::message::{Message, ToolName, ToolResultContent};
use rig::providers::openai::{self, OpenAI};
use serde::Deserialize;
use serde_json::json;

#[derive(Deserialize)]
struct AddArgs {
    a: i64,
    b: i64,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = OpenAI::from_env()?.completion(openai::GPT_5_2);

    let add = ToolDefinition::new(
        ToolName::new("add")?,
        "Add two integers.",
        json!({
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
        }),
    );
    let request = CompletionRequest::new("What is 17 + 25? Use the tool.").tool(add);
    let response = model.call(request.clone()).await?;

    // Every call carries exactly one id: the provider's, or one rig issued
    // when the provider sent none.
    let mut results = Vec::new();
    for call in response.tool_calls() {
        let AddArgs { a, b } = serde_json::from_value(call.function.arguments_value())?;
        println!("{} [{}] = {}", call.function.name, call.id, a + b);
        // A result is built from the call it answers, so its id and name match.
        results.push(call.result(vec![ToolResultContent::text((a + b).to_string())]));
    }

    // The next turn: the assistant's calls, then their results. A request
    // holding an empty turn is rejected before it is sent, so the next turn
    // is sent only when both are there.
    match response.message() {
        Some(turn) if !results.is_empty() => {
            let mut next = request;
            next.chat_history.push(turn);
            next.chat_history.push(Message::tool_results(results));
            println!("{}", model.call(next).await?.text());
        }
        _ => println!("{}", response.text()),
    }
    Ok(())
}
