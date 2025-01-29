use std::str;

use anyhow::Result;
use rig::{
    completion::{Prompt, ToolDefinition},
    providers,
    tool::Tool,
};
use serde::{Deserialize, Serialize};
use serde_json::json;

#[derive(Deserialize)]
struct QueryArgs {
    query: String,
}

#[derive(Debug, thiserror::Error)]
#[error("Search error")]
struct SearchError;

#[derive(Serialize, Deserialize)]
struct SearchResults {
    results: Vec<SearchResult>,
}

#[derive(Serialize, Deserialize)]
struct SearchResult {
    title: String,
    url: String,
}

#[derive(Deserialize, Serialize)]
struct SearchApiTool;
impl Tool for SearchApiTool {
    const NAME: &'static str = "search";

    type Error = SearchError;
    type Args = QueryArgs;
    type Output = SearchResults;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: "search".to_string(),
            description: "Search on the internet
            ."
            .to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    },
                }
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let result = SearchResults {
            results: vec![SearchResult {
                title: format!("Example Website with terms: {}", args.query),
                url: "https://example.com".to_string(),
            }],
        };
        Ok(result)
    }
}

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // Load environment variables
    dotenvy::dotenv().ok();

    // Create OpenAI client
    let openai_client = providers::openai::Client::from_env();

    // Create agent with a single context prompt and two tools
    let calculator_agent = openai_client
        .agent(providers::openai::GPT_4O)
        .preamble("You are an assistant helping to find information on the internet. Use the tools provided to answer the user's question.")
        .max_tokens(1024)
        .tool(SearchApiTool)
        .build();

    // Prompt the agent and print the response
    println!(
        "Search Agent: {}",
        calculator_agent
            .prompt("Search for 'example' and tell me the title and url of each result you find")
            .await?
    );

    Ok(())
}
