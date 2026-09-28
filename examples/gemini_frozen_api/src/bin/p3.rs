use futures::StreamExt;
use rig::AgentBuilder;
use rig::agent::MultiTurnStreamItem;
use rig::message::AssistantContent;
use rig::providers::gemini::{self, Gemini, api};
use rig::streaming::{Item, StreamEvent, StreamedUserContent};
use rig::tool::PortableTool;
use serde::Deserialize;
use serde_json::{Value, json};

#[derive(Deserialize)]
struct SearchArgs {
    query: String,
}

struct SearchIssues;

impl PortableTool for SearchIssues {
    const NAME: &'static str = "search_issues";
    type Args = SearchArgs;
    type Output = Value;
    type Error = std::convert::Infallible;

    fn description(&self) -> String {
        "Search open issues. Returns id, title and labels.".to_owned()
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": { "query": { "type": "string" } },
            "required": ["query"]
        })
    }

    async fn call(&self, args: SearchArgs) -> Result<Value, Self::Error> {
        Ok(json!([
            { "id": 41, "title": format!("{} panics on empty input", args.query), "labels": ["bug"] },
            { "id": 57, "title": format!("{} is slow on large files", args.query), "labels": ["perf", "bug"] }
        ]))
    }
}

/// The app's UI: a terminal here, a websocket in production.
struct Ui;

impl Ui {
    fn text(&self, delta: &str) {
        print!("{delta}");
    }
    fn thought(&self, delta: &str) {
        eprint!("{delta}");
    }
    fn tool_call(&self, name: &str, args: &Value) {
        println!("\n[calling {name} {args}]");
    }
    fn tool_result(&self, name: &str) {
        println!("[{name} returned]");
    }
    fn hosted_code(&self, code: &str) {
        println!("\n[gemini ran]\n{code}");
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let gemini = Gemini::from_env()?;
    // Function and hosted tools together: rig sets includeServerSideToolInvocations.
    let model = gemini
        .completion(gemini::GEMINI_3_8_FLASH)
        .settings(api::RequestSettings {
            generation_config: api::GenerationSettings {
                thinking_config: Some(api::ThinkingConfig {
                    include_thoughts: Some(true),
                    ..Default::default()
                }),
                ..Default::default()
            },
            tools: vec![api::HostedTool {
                code_execution: Some(api::CodeExecution::default()),
                ..Default::default()
            }],
            ..Default::default()
        });
    let agent = AgentBuilder::new(model)
        .preamble("You triage issues. Use search_issues, then count with code.")
        .tool(SearchIssues)
        .build();

    let ui = Ui;
    let mut stream = agent
        .prompt("Which open issues mention the parser, and how many are there per label?")
        .max_turns(8)
        .stream();

    while let Some(item) = stream.next().await {
        match item? {
            MultiTurnStreamItem::StreamAssistantItem(Item::Event(event)) => match event {
                StreamEvent::Text { text, .. } => ui.text(&text),
                StreamEvent::Reasoning { text, .. } => ui.thought(&text),
                StreamEvent::End {
                    content: AssistantContent::ToolCall(call),
                    ..
                } => ui.tool_call(&String::from(call.function.name), &call.function.arguments),
                StreamEvent::End {
                    content: AssistantContent::Native(native),
                    ..
                } => {
                    let part = api::Part::try_from(&native)?;
                    if let Some(code) = part.executable_code {
                        ui.hosted_code(code.code.as_deref().unwrap_or_default());
                    }
                }
                _ => {}
            },
            MultiTurnStreamItem::StreamUserItem(StreamedUserContent::ToolResult {
                tool_result,
            }) => ui.tool_result(&String::from(tool_result.name)),
            MultiTurnStreamItem::FinalResponse(response) => {
                println!("\n\nusage: {:?}", response.usage);
            }
            _ => {}
        }
    }
    Ok(())
}
