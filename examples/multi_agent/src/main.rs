//! A multi agent application: a translator agent exposed as a tool to a main
//! agent.
//!
//! The translator sub-agent is wrapped in a [`PortableDynamicTool`] whose
//! callback closes over a clone of the agent (`Agent` is `Clone`) and prompts
//! it. When the main agent receives text that is not in English (or has
//! grammatical errors), it calls the translator tool first, then answers.
use anyhow::Result;
use rig::integrations::cli_chatbot::ChatBotBuilder;
use rig::prelude::*;
use rig::providers::openai;
use rig::tool::{PortableDynamicTool, ToolExecutionError, ToolOutput};
use serde_json::json;

const TRANSLATOR_TOOL_NAME: &str = "translator";

/// A multi agent application that consists of two components:
/// an agent specialized in translating prompt into english and a simple GPT-4 model.
/// When provided with a prompt in a language besides english, the application will use
/// the translator agent to translate the prompt in english, before answering it with GPT-4.
/// The answer in english is returned.
#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    let client = openai::Client::from_env()?;

    let translator_agent = client
                .agent(openai::GPT_4O)
                .preamble(
                    "You are a translator assistant that will translate any input text into english. \
                    If the text is already in english, simply respond with the original text but fix any mistakes (grammar, syntax, etc.)."
                )
                .build();

    // Expose the translator agent as a tool by closing over it in a dynamic
    // tool callback.
    let inner = translator_agent.clone();
    let translator_tool = PortableDynamicTool::new(
        TRANSLATOR_TOOL_NAME,
        "Translate any text to English. If already in English, fix grammar and syntax issues.",
        json!({
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "The text to translate to English"
                }
            },
            "required": ["prompt"]
        }),
        move |args| {
            let inner = inner.clone();
            Box::pin(async move {
                let prompt = args
                    .get("prompt")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default()
                    .to_string();
                let response = inner
                    .prompt(prompt)
                    .await
                    .map_err(|e| ToolExecutionError::other(e.to_string()))?;
                println!("Translated prompt: {response}");
                Ok(ToolOutput::text(response))
            })
        },
    );

    let multi_agent_system = client
        .agent(openai::GPT_4O)
        .preamble(&format!(
            "You are a helpful assistant that can work with text in any language. \
            When you receive input that is not in English, or contains grammatical errors \
            use the {} tool first to ensure proper English, then provide your response. \
            Always show both the translated text and your final response.",
            TRANSLATOR_TOOL_NAME
        ))
        .dynamic_tool(translator_tool)
        .build();

    // Spin up a CLI chatbot using the multi-agent system
    let chatbot = ChatBotBuilder::new(multi_agent_system).max_turns(2).build();

    chatbot.run().await?;

    Ok(())
}
