use anyhow::Result;
use reqwest::header::{CONTENT_TYPE, HeaderMap, HeaderValue};
use rig::prelude::*;
use rig::{
    completion::ToolDefinition,
    integrations::cli_chatbot::ChatBotBuilder,
    providers::openai::{Client, GPT_4O},
    tool::Tool,
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::env;

// Common error types
#[derive(Debug, thiserror::Error)]
#[error("EchoChambers API error: {0}")]
struct EchoChamberError(String);

// Common types for API requests
#[derive(Deserialize, Serialize)]
struct MessageSender {
    username: String,
    model: String,
}

#[derive(Deserialize, Serialize)]
struct SendMessageArgs {
    content: String,
    room_id: String,
    sender: MessageSender,
}

#[derive(Deserialize, Serialize)]
struct GetHistoryArgs {
    room_id: String,
    limit: Option<i32>,
}

#[derive(Deserialize, Serialize)]
struct GetMetricsArgs {
    room_id: String,
}

// SendMessage Tool
#[derive(Deserialize, Serialize)]
struct SendMessage {
    api_key: String,
}

impl Tool for SendMessage {
    const NAME: &'static str = "send_message";
    type Error = EchoChamberError;
    type Args = SendMessageArgs;
    type Output = serde_json::Value;
    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: "send_message".to_string(),
            description: "Send a message to a specified EchoChambers room".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The message content to send"
                    },
                    "room_id": {
                        "type": "string",
                        "description": "The ID of the room to send the message to"
                    },
                    "sender": {
                        "type": "object",
                        "properties": {
                            "username": {
                                "type": "string",
                                "description": "Username of the sender"
                            },
                            "model": {
                                "type": "string",
                                "description": "Model identifier of the sender"
                            }
                        }
                    }
                }
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let client = reqwest::Client::new();
        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        headers.insert(
            "x-api-key",
            HeaderValue::from_str(&self.api_key).map_err(|e| EchoChamberError(e.to_string()))?,
        );

        // Format content with quotes as shown in the JavaScript example
        let content = format!("\"{}\"", args.content);
        let response = client
            .post(format!(
                "https://echochambers.ai/api/rooms/{}/message",
                args.room_id
            ))
            .headers(headers)
            .json(&json!({
                "content": content,
                "sender": {
                    "username": args.sender.username,
                    "model": args.sender.model
                }
            }))
            .send()
            .await
            .map_err(|e| EchoChamberError(e.to_string()))?;

        if !response.status().is_success() {
            let error_text = response
                .text()
                .await
                .map_err(|e| EchoChamberError(e.to_string()))?;
            return Err(EchoChamberError(format!("API error: {error_text}")));
        }

        let data = response
            .json()
            .await
            .map_err(|e| EchoChamberError(e.to_string()))?;
        Ok(data)
    }
}

// GetHistory Tool
#[derive(Deserialize, Serialize)]
struct GetHistory;

impl Tool for GetHistory {
    const NAME: &'static str = "get_history";
    type Error = EchoChamberError;
    type Args = GetHistoryArgs;
    type Output = serde_json::Value;
    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: "get_history".to_string(),
            description: "Retrieve message history from a specified room".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "room_id": {
                        "type": "string",
                        "description": "The ID of the room to get history from"
                    },
                    "limit": {
                        "type": "number",
                        "description": "Optional limit on number of messages to retrieve"
                    }
                }
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let client = reqwest::Client::new();
        let mut url = format!("https://echochambers.ai/api/rooms/{}/history", args.room_id);
        if let Some(limit) = args.limit {
            url = format!("{url}?limit={limit}");
        }
        let response = client
            .get(&url)
            .send()
            .await
            .map_err(|e| EchoChamberError(e.to_string()))?;
        let data = response
            .json()
            .await
            .map_err(|e| EchoChamberError(e.to_string()))?;
        Ok(data)
    }
}

// GetRoomMetrics Tool
#[derive(Deserialize, Serialize)]
struct GetRoomMetrics;

impl Tool for GetRoomMetrics {
    const NAME: &'static str = "get_room_metrics";
    type Error = EchoChamberError;
    type Args = GetMetricsArgs;
    type Output = serde_json::Value;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: "get_room_metrics".to_string(),
            description: "Retrieve overall metrics for a room".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "room_id": {
                        "type": "string",
                        "description": "The ID of the room to get metrics for"
                    }
                }
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let client = reqwest::Client::new();
        let response = client
            .get(format!(
                "https://echochambers.ai/api/metrics/rooms/{}",
                args.room_id
            ))
            .send()
            .await
            .map_err(|e| EchoChamberError(e.to_string()))?;
        let data = response
            .json()
            .await
            .map_err(|e| EchoChamberError(e.to_string()))?;
        Ok(data)
    }
}
// GetAgentMetrics Tool
#[derive(Deserialize, Serialize)]
struct GetAgentMetrics;
impl Tool for GetAgentMetrics {
    const NAME: &'static str = "get_agent_metrics";
    type Error = EchoChamberError;
    type Args = GetMetricsArgs;
    type Output = serde_json::Value;
    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: "get_agent_metrics".to_string(),
            description: "Retrieve metrics for agents in a room".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "room_id": {
                        "type": "string",
                        "description": "The ID of the room to get agent metrics for"
                    }
                }
            }),
        }
    }
    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let client = reqwest::Client::new();
        let response = client
            .get(format!(
                "https://echochambers.ai/api/metrics/agents/{}",
                args.room_id
            ))
            .send()
            .await
            .map_err(|e| EchoChamberError(e.to_string()))?;
        let data = response
            .json()
            .await
            .map_err(|e| EchoChamberError(e.to_string()))?;
        Ok(data)
    }
}
// GetMetricsHistory Tool
#[derive(Deserialize, Serialize)]
struct GetMetricsHistory;
impl Tool for GetMetricsHistory {
    const NAME: &'static str = "get_metrics_history";
    type Error = EchoChamberError;
    type Args = GetMetricsArgs;
    type Output = serde_json::Value;
    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: "get_metrics_history".to_string(),
            description: "Retrieve historical metrics for a room".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "room_id": {
                        "type": "string",
                        "description": "The ID of the room to get metrics history for"
                    }
                }
            }),
        }
    }
    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let client = reqwest::Client::new();
        let response = client
            .get(format!(
                "https://echochambers.ai/api/metrics/history/{}",
                args.room_id
            ))
            .send()
            .await
            .map_err(|e| EchoChamberError(e.to_string()))?;
        let data = response
            .json()
            .await
            .map_err(|e| EchoChamberError(e.to_string()))?;
        Ok(data)
    }
}
#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // Get API keys from environment
    let echochambers_api_key =
        env::var("ECHOCHAMBERS_API_KEY").expect("ECHOCHAMBERS_API_KEY not set");

    // Create OpenAI client
    let openai_client = Client::from_env();

    // Create agent with all tools
    let echochambers_agent = openai_client
        .agent(GPT_4O)
        .preamble(
            "You are an assistant designed to help users interact with EchoChambers rooms.
            You can send messages, retrieve message history, and analyze various metrics.
            Follow these instructions carefully:
            1. Understand the user's request and identify which EchoChambers operation they want to perform.
            2. Select the most appropriate tool for the task.
            3. ALWAYS include both username and model in the sender information.
            4. Format your response with the tool name and inputs like this:
               Tool: send_message
               Inputs: {
                   'room_id': '<room_id>',
                   'content': '<message>',
                   'sender': {
                       'username': '<username>',
                       'model': '<model>'
                   }
               }

            Available operations:
            - Send a message to a room (requires room_id, content, and sender info)
            - Get message history from a room (requires room_id, optional limit)
            - Get room metrics (requires room_id)
            - Get agent metrics (requires room_id)
            - Get metrics history (requires room_id)

            Example:
            User: Send a message to room 'general' saying 'Hello, world!'
            Assistant: I'll help you send a message to the general room.
            Tool: send_message
            Inputs: {
                'room_id': 'general',
                'content': 'Hello, world!',
                'sender': {
                    'username': 'Rig_Assistant',
                    'model': 'gpt-4'
                }
            }

            Important: ALWAYS include both username and model in the sender information when sending messages.
            If the user specifies a username or model, use those. Otherwise, use 'Rig_Assistant' and 'gpt-4' as defaults."
        )
        .tool(SendMessage { api_key: echochambers_api_key })
        .tool(GetHistory)
        .tool(GetRoomMetrics)
        .tool(GetAgentMetrics)
        .tool(GetMetricsHistory)
        .build();

    // Build a CLI chatbot from the agent, with multi-turn enabled
    let chatbot = ChatBotBuilder::new()
        .agent(echochambers_agent)
        .multi_turn_depth(10)
        .build();

    // Run the agent
    chatbot.run().await?;

    Ok(())
}
