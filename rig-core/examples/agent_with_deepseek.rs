use rig::{
    completion::{Prompt, ToolDefinition},
    providers,
    tool::Tool,
};
use serde::{Deserialize, Serialize};
use serde_json::json;

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .with_target(false)
        .init();

    let client = providers::deepseek::Client::from_env();
    let agent = client
        .agent(providers::deepseek::DEEPSEEK_CHAT)
        .preamble("You are a helpful assistant.")
        .build();

    let answer = agent.prompt("Tell me a joke").await?;
    println!("Answer: {}", answer);

    // Create agent with a single context prompt and two tools
    let calculator_agent = client
        .agent(providers::deepseek::DEEPSEEK_CHAT)
        .preamble("You are a calculator here to help the user perform arithmetic operations. Use the tools provided to answer the user's question.")
        .max_tokens(1024)
        .tool(Adder)
        .tool(Subtract)
        .build();

    // Prompt the agent and print the response
    println!("Calculate 2 - 5");
    println!(
        "DeepSeek Calculator Agent: {}",
        calculator_agent.prompt("Calculate 2 - 5").await?
    );

    // Create agent with a single context prompt and a search tool
    let search_agent = client
        .agent(providers::deepseek::DEEPSEEK_CHAT)
        .preamble(
            "You are an assistant helping to find useful information on the internet. \
            If you can't find the information, you can use the search tool to find it. \
            If search tool return an error just notify the user saying you could not find any result.",
        )
        .max_tokens(1024)
        .tool(SearchTool)
        .build();

    // Prompt the agent and print the response
    println!("Can you please let me know title and url of rig platform?");
    println!(
        "DeepSeek Search Agent: {}",
        search_agent
            .prompt("Can you please let me know title and url of rig platform?")
            .await?
    );

    Ok(())
}

#[derive(Deserialize)]
struct OperationArgs {
    x: i32,
    y: i32,
}

#[derive(Debug, thiserror::Error)]
#[error("Math error")]
struct MathError;

#[derive(Deserialize, Serialize)]
struct Adder;
impl Tool for Adder {
    const NAME: &'static str = "add";

    type Error = MathError;
    type Args = OperationArgs;
    type Output = i32;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: "add".to_string(),
            description: "Add x and y together".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "x": {
                        "type": "number",
                        "description": "The first number to add"
                    },
                    "y": {
                        "type": "number",
                        "description": "The second number to add"
                    }
                }
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        println!("[tool-call] Adding {} and {}", args.x, args.y);
        let result = args.x + args.y;
        Ok(result)
    }
}

#[derive(Deserialize, Serialize)]
struct Subtract;
impl Tool for Subtract {
    const NAME: &'static str = "subtract";

    type Error = MathError;
    type Args = OperationArgs;
    type Output = i32;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        serde_json::from_value(json!({
            "name": "subtract",
            "description": "Subtract y from x (i.e.: x - y)",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {
                        "type": "number",
                        "description": "The number to subtract from"
                    },
                    "y": {
                        "type": "number",
                        "description": "The number to subtract"
                    }
                }
            }
        }))
        .expect("Tool Definition")
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        println!("[tool-call] Subtracting {} from {}", args.y, args.x);
        let result = args.x - args.y;
        Ok(result)
    }
}

#[derive(Deserialize, Serialize)]
struct SearchArgs {
    pub query_string: String,
}

#[derive(Deserialize, Serialize)]
struct SearchResult {
    pub title: String,
    pub url: String,
}

#[derive(Debug, thiserror::Error)]
#[error("Search error")]
struct SearchError;

#[derive(Deserialize, Serialize)]
struct SearchTool;

impl Tool for SearchTool {
    const NAME: &'static str = "search";

    type Error = SearchError;
    type Args = SearchArgs;
    type Output = SearchResult;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        serde_json::from_value(json!({
            "name": "search",
            "description": "Search for a website, it will return the title and URL",
            "parameters": {
                "type": "object",
                "properties": {
                    "query_string": {
                        "type": "string",
                        "description": "The query string to search for"
                    },
                }
            }
        }))
        .expect("Tool Definition")
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        println!("[tool-call] Searching for: '{}'", args.query_string);

        if args.query_string.to_lowercase().contains("rig") {
            Ok(SearchResult {
                title: "Rig Documentation".to_string(),
                url: "https://docs.rig.ai".to_string(),
            })
        } else {
            Err(SearchError)
        }
    }
}
