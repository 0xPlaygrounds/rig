use anyhow::Result;
use rig::integrations::cli_chatbot::ChatBotBuilder;
use rig::prelude::*;
use rig::{
    agent::{Agent, AgentBuilder},
    completion::{Chat, CompletionModel, PromptError, ToolDefinition},
    providers::openai::Client as OpenAIClient,
    tool::Tool,
};
use serde::Deserialize;
use serde_json::json;

// Define a wrapper around an agent so that it can be provided to another agent
// as a tool
struct TranslatorTool<M: CompletionModel>(Agent<M>);

// The input that will be sent to the translator agent from the main agent
#[derive(Deserialize)]
struct TranslatorArgs {
    prompt: String,
}

impl<M: CompletionModel> Tool for TranslatorTool<M> {
    const NAME: &'static str = "translator";

    type Args = TranslatorArgs;
    type Error = PromptError;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        serde_json::from_value(json!({
            "name": Self::NAME,
            "description": "Translate any text to English. If already in English, fix grammar and syntax issues.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "The text to translate to English"
                    },
                },
                "required": ["prompt"]
            }
        }))
        .expect("Tool Definition")
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        match self.0.chat(&args.prompt, vec![]).await {
            Ok(response) => {
                println!("Translated prompt: {response}");
                Ok(response)
            }
            Err(e) => Err(e),
        }
    }
}

/// A multi agent application that consists of two components:
/// an agent specialized in translating prompt into english and a simple GPT-4 model.
/// When provided with a prompt in a language besides english, the application will use
/// the translator agent to translate the prompt in english, before answering it with GPT-4.
/// The answer in english is returned.
#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // Create OpenAI client
    let openai_client = OpenAIClient::from_env();
    let model = openai_client.completion_model("gpt-4");

    let translator_agent = AgentBuilder::new(model.clone())
                .preamble(
                    "You are a translator assistant that will translate any input text into english. \
                    If the text is already in english, simply respond with the original text but fix any mistakes (grammar, syntax, etc.)."
                )
                .build();

    let translator_tool = TranslatorTool(translator_agent);

    let multi_agent_system = AgentBuilder::new(model)
        .preamble(&format!(
            "You are a helpful assistant that can work with text in any language. \
            When you receive input that is not in English, or contains grammatical errors \
            use the {} tool first to ensure proper English, then provide your response. \
            Always show both the translated text and your final response.",
            translator_tool.name()
        ))
        .tool(translator_tool)
        .build();

    // Spin up a CLI chatbot using the multi-agent system
    let chatbot = ChatBotBuilder::new()
        .agent(multi_agent_system)
        .multi_turn_depth(1)
        .build();

    chatbot.run().await?;

    Ok(())
}
