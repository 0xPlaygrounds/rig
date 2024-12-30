use rig::agent::AgentBuilder;
use rig::completion::ToolDefinition;
use rig::loaders::FileLoader;
use rig::providers::eternalai::{CompletionModel, NOUS_RESEARCH_HERMES_3_LLAMA_3_1_70B_FP8};
use rig::tool::Tool;
use rig::{completion::Prompt, providers};
use serde::{Deserialize, Serialize};
use serde_json::json;

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    println!("Running basic agent with eternalai");
    basic_eternalai().await?;

    println!("\nRunning eternalai agent with tools");
    tools_eternalai().await?;

    println!("\nRunning eternalai agent with loaders");
    loaders_eternalai().await?;

    println!("\nRunning eternalai agent with context");
    context_eternalai().await?;

    println!("\n\nAll agents ran successfully");
    Ok(())
}

fn client() -> providers::eternalai::Client {
    providers::eternalai::Client::from_env()
}

fn partial_agent_eternalai() -> AgentBuilder<CompletionModel> {
    let client = client();
    client.agent(
        NOUS_RESEARCH_HERMES_3_LLAMA_3_1_70B_FP8,
        // Option::from("45762"),
        None,
    )
}

async fn basic_eternalai() -> Result<(), anyhow::Error> {
    let comedian_agent = partial_agent_eternalai()
        .preamble("You are a comedian here to entertain the user using humour and jokes.")
        .build();

    // Prompt the agent and print the response
    let response = comedian_agent.prompt("Entertain me!").await?;
    println!("{}", response);

    Ok(())
}

async fn tools_eternalai() -> Result<(), anyhow::Error> {
    // Create agent with a single context prompt and two tools
    let calculator_agent = partial_agent_eternalai()
        .preamble("You are a calculator here to help the user perform arithmetic operations. Use the tools provided to answer the user's question.")
        .max_tokens(1024)
        .tool(EternalAIAdder)
        .tool(EternalAISubtract)
        .build();

    // Prompt the agent and print the response
    println!("Calculate 2 - 5");
    println!(
        "Calculator Agent: {}",
        calculator_agent.prompt("Calculate 2 - 5").await?
    );

    Ok(())
}

async fn loaders_eternalai() -> Result<(), anyhow::Error> {
    let model = client().completion_model(
        providers::eternalai::NOUS_RESEARCH_HERMES_3_LLAMA_3_1_70B_FP8,
        // Option::from("45762"),
        None,
    );

    // Load in all the rust examples
    let examples = FileLoader::with_glob("rig-core/examples/*.rs")?
        .read_with_path()
        .ignore_errors()
        .into_iter();

    // Create an agent with multiple context documents
    let agent = examples
        .fold(AgentBuilder::new(model), |builder, (path, content)| {
            builder.context(format!("Rust Example {:?}:\n{}", path, content).as_str())
        })
        .build();

    // Prompt the agent and print the response
    let response = agent
        .prompt("Which rust example is best suited for the operation 1 + 2")
        .await?;

    println!("{}", response);

    Ok(())
}

async fn context_eternalai() -> Result<(), anyhow::Error> {
    let model = client().completion_model(
        providers::eternalai::NOUS_RESEARCH_HERMES_3_LLAMA_3_1_70B_FP8,
        // Option::from("45762"),
        None,
    );

    // Create an agent with multiple context documents
    let agent = AgentBuilder::new(model)
        .context("Definition of a *flurbo*: A flurbo is a green alien that lives on cold planets")
        .context("Definition of a *glarb-glarb*: A glarb-glarb is a ancient tool used by the ancestors of the inhabitants of planet Jiro to farm the land.")
        .context("Definition of a *linglingdong*: A term used by inhabitants of the far side of the moon to describe humans.")
        .build();

    // Prompt the agent and print the response
    let response = agent.prompt("What does \"glarb-glarb\" mean?").await?;

    println!("{}", response);

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
struct EternalAIAdder;
impl Tool for EternalAIAdder {
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
        let result = args.x + args.y;
        Ok(result)
    }
}

#[derive(Deserialize, Serialize)]
struct EternalAISubtract;
impl Tool for EternalAISubtract {
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
                        "description": "The number to substract from"
                    },
                    "y": {
                        "type": "number",
                        "description": "The number to substract"
                    }
                }
            }
        }))
        .expect("Tool Definition")
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let result = args.x - args.y;
        Ok(result)
    }
}
