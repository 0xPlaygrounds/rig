use rig::AgentBuilder;
use rig::message::{AssistantContent, Message};
use rig::providers::gemini::{self, Gemini, api};
use rig::tool::PortableTool;
use serde::Deserialize;
use serde_json::{Value, json};

#[derive(Deserialize)]
struct AccountArgs {
    account: String,
}

struct Holdings;

impl PortableTool for Holdings {
    const NAME: &'static str = "portfolio_holdings";
    type Args = AccountArgs;
    type Output = Value;
    type Error = std::convert::Infallible;

    fn description(&self) -> String {
        "Tickers and share counts held in an account.".to_owned()
    }

    fn parameters(&self) -> Value {
        json!({ "type": "object", "properties": { "account": { "type": "string" } }, "required": ["account"] })
    }

    async fn call(&self, args: AccountArgs) -> Result<Value, Self::Error> {
        Ok(json!({ "account": args.account, "holdings": [
            { "ticker": "GOOGL", "shares": 12 },
            { "ticker": "MSFT", "shares": 5 }
        ]}))
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let gemini = Gemini::from_env()?;

    let model = gemini
        .completion(gemini::GEMINI_3_8_FLASH)
        .settings(api::RequestSettings {
            tools: vec![
                api::HostedTool {
                    code_execution: Some(api::CodeExecution::default()),
                    ..Default::default()
                },
                api::HostedTool {
                    google_search: Some(api::GoogleSearch::default()),
                    ..Default::default()
                },
            ],
            ..Default::default()
        });

    let agent = AgentBuilder::new(model)
        .preamble(
            "Use portfolio_holdings for positions, Google Search for prices, code for arithmetic.",
        )
        .tool(Holdings)
        .default_max_turns(8)
        .build();

    let mut history = Vec::new();
    let first = agent
        .chat(
            "What is account ACC-9 worth at today's closing prices?".to_owned(),
            &mut history,
        )
        .await?;
    println!("{}", first.output);
    let second = agent
        .chat("And if every price drops 7%?".to_owned(), &mut history)
        .await?;
    println!("{}", second.output);

    for message in &history {
        if let Message::Assistant { content, .. } = message {
            for item in content.iter() {
                if let AssistantContent::Native(native) = item {
                    let part = api::Part::try_from(native)?;
                    if let Some(code) = &part.executable_code {
                        println!("gemini ran:\n{}", code.code.as_deref().unwrap_or_default());
                    }
                    if let Some(result) = &part.code_execution_result {
                        println!("-> {:?}", result.output);
                    }
                    if let Some(call) = &part.tool_call {
                        println!("server tool: {:?}", call.tool_type);
                    }
                }
            }
        }
    }
    Ok(())
}
