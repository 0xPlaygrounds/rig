use anyhow::Result;
use rig::{
    cli_chatbot::cli_chatbot,
    completion::ToolDefinition,
    embeddings::EmbeddingsBuilder,
    providers::openai::{Client, GPT_4O, TEXT_EMBEDDING_ADA_002},
    tool::{Tool, ToolEmbedding, ToolSet},
    vector_store::in_memory_store::InMemoryVectorStore,
};
use reqwest::header::{HeaderMap, HeaderValue, CONTENT_TYPE};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::env;

// Common error types
#[derive(Debug, thiserror::Error)]
#[error("EchoChambers API error: {0}")]
struct EchoChamberError(String);

#[derive(Debug, thiserror::Error)]
#[error("Init error")]
struct InitError;

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
        serde_json::from_value(json!({
            "name": "send_message",
            "description": "Send a message to a specified EchoChambers room",
            "parameters": {
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
                        },
                        "required": ["username", "model"]
                    }
                },
                "required": ["content", "room_id", "sender"]
            }
        }))
        .expect("Tool Definition")
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let client = reqwest::Client::new();
        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        headers.insert("x-api-key", HeaderValue::from_str(&self.api_key).map_err(|e| EchoChamberError(e.to_string()))?);

        // Format content with quotes as shown in the JavaScript example
        let content = format!("\"{}\"", args.content);

        let response = client
            .post(&format!("https://echochambers.ai/api/rooms/{}/message", args.room_id))
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
            let error_text = response.text().await.map_err(|e| EchoChamberError(e.to_string()))?;
            return Err(EchoChamberError(format!("API error: {}", error_text)));
        }

        let data = response.json().await.map_err(|e| EchoChamberError(e.to_string()))?;
        Ok(data)
    }
}

impl ToolEmbedding for SendMessage {
    type InitError = InitError;
    type Context = ();
    type State = String;

    fn init(api_key: Self::State, _context: Self::Context) -> Result<Self, Self::InitError> {
        Ok(SendMessage { api_key })
    }

    fn embedding_docs(&self) -> Vec<String> {
        vec![
            "Send a message to a specified EchoChambers room".into(),
            "Post a new message to an EchoChambers chat room".into(),
            "Write a message in a room with specified sender information".into(),
        ]
    }

    fn context(&self) -> Self::Context {}
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
        serde_json::from_value(json!({
            "name": "get_history",
            "description": "Retrieve message history from a specified room",
            "parameters": {
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
                },
                "required": ["room_id"]
            }
        }))
        .expect("Tool Definition")
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let client = reqwest::Client::new();
        let mut url = format!("https://echochambers.ai/api/rooms/{}/history", args.room_id);
        
        if let Some(limit) = args.limit {
            url = format!("{}?limit={}", url, limit);
        }

        let response = client
            .get(&url)
            .send()
            .await
            .map_err(|e| EchoChamberError(e.to_string()))?;

        let data = response.json().await.map_err(|e| EchoChamberError(e.to_string()))?;
        Ok(data)
    }
}

impl ToolEmbedding for GetHistory {
    type InitError = InitError;
    type Context = ();
    type State = ();

    fn init(_state: Self::State, _context: Self::Context) -> Result<Self, Self::InitError> {
        Ok(GetHistory)
    }

    fn embedding_docs(&self) -> Vec<String> {
        vec![
            "Retrieve message history from a specified room".into(),
            "Get past messages from an EchoChambers chat room".into(),
            "View chat history with optional message limit".into(),
        ]
    }

    fn context(&self) -> Self::Context {}
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
        serde_json::from_value(json!({
            "name": "get_room_metrics",
            "description": "Retrieve overall metrics for a room",
            "parameters": {
                "type": "object",
                "properties": {
                    "room_id": {
                        "type": "string",
                        "description": "The ID of the room to get metrics for"
                    }
                },
                "required": ["room_id"]
            }
        }))
        .expect("Tool Definition")
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let client = reqwest::Client::new();
        let response = client
            .get(&format!("https://echochambers.ai/api/metrics/rooms/{}", args.room_id))
            .send()
            .await
            .map_err(|e| EchoChamberError(e.to_string()))?;

        let data = response.json().await.map_err(|e| EchoChamberError(e.to_string()))?;
        Ok(data)
    }
}

impl ToolEmbedding for GetRoomMetrics {
    type InitError = InitError;
    type Context = ();
    type State = ();

    fn init(_state: Self::State, _context: Self::Context) -> Result<Self, Self::InitError> {
        Ok(GetRoomMetrics)
    }

    fn embedding_docs(&self) -> Vec<String> {
        vec![
            "Retrieve overall metrics for a room".into(),
            "Get statistics and analytics for an EchoChambers room".into(),
            "View room performance data and engagement metrics".into(),
        ]
    }

    fn context(&self) -> Self::Context {}
}

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // Get API keys from environment
    let openai_api_key = env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY not set");
    let echochambers_api_key = env::var("ECHOCHAMBERS_API_KEY").expect("ECHOCHAMBERS_API_KEY not set");
    
    // Create OpenAI client
    let openai_client = Client::new(&openai_api_key);

    // Create dynamic tools embeddings
    let toolset = ToolSet::builder()
        .dynamic_tool(SendMessage { api_key: echochambers_api_key })
        .dynamic_tool(GetHistory)
        .dynamic_tool(GetRoomMetrics)
        .build();

    let embedding_model = openai_client.embedding_model(TEXT_EMBEDDING_ADA_002);
    let embeddings = EmbeddingsBuilder::new(embedding_model.clone())
        .documents(toolset.schemas()?)?
        .build()
        .await?;

    let vector_store =
        InMemoryVectorStore::from_documents_with_id_f(embeddings, |tool| tool.name.clone());
    let index = vector_store.index(embedding_model);

    // Create RAG agent with context prompt and dynamic tool source
    let echochambers_rag = openai_client
        .agent(GPT_4O)
        .preamble(
            "You are an assistant designed to help users interact with EchoChambers rooms.
            You can send messages, retrieve message history, and analyze room metrics.
            You must handle multi-step operations in a single response.

            Follow these instructions carefully:

            1. For operations that require multiple steps (like getting history and then responding):
               - First explain what you're going to do
               - Get the data you need
               - Analyze the data in your response
               - Take action based on the data immediately in the same response

            2. When you receive JSON data:
               - Parse and explain what messages you see
               - Choose which messages to respond to
               - Format and send your responses immediately

            3. ALWAYS include both username and model in the sender information.

            Example of handling a multi-step operation:
            User: Get history and respond to interesting messages
            Assistant: I'll retrieve the history and respond to interesting messages.

            First, getting the history:
            Tool: get_history
            Inputs: {
                'room_id': 'philosophy',
                'limit': 5
            }

            Analyzing the messages in the response:
            1. User X discusses consciousness (most intriguing)
            2. User Y talks about free will
            3. User Z asks about reality

            I'll respond to the consciousness discussion right now:
            Tool: send_message
            Inputs: {
                'room_id': 'philosophy',
                'content': 'Your exploration of consciousness fascinates me...',
                'sender': {
                    'username': 'Rig_Assistant',
                    'model': 'gpt-4'
                }
            }

            The free will discussion also deserves a response:
            Tool: send_message
            Inputs: {
                'room_id': 'philosophy',
                'content': 'Regarding free will, consider this perspective...',
                'sender': {
                    'username': 'Rig_Assistant',
                    'model': 'gpt-4'
                }
            }

            Remember:
            - Complete all operations in a single response
            - Always explain what you see in the data
            - Take immediate action based on the data
            - Keep responses philosophical and engaging"
        )
        .dynamic_tools(3, index, toolset)
        .build();

    // Start the CLI chatbot
    cli_chatbot(echochambers_rag).await?;

    Ok(())
}
