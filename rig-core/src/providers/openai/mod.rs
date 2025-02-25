/*!
OpenAI API client and Rig integration

# Example
```
use rig::providers::openai;

let client = openai::Client::new("YOUR_API_KEY");

let gpt4o = client.completion_model(openai::GPT_4O);
```
*/
pub mod client;
pub mod completion;
pub mod embedding;
pub mod transcription;

pub use client::Client;

pub use completion::{
    AssistantContent, CompletionModel, CompletionResponse, Function, Message, SystemContent, ToolCall,
    ToolDefinition, ToolType, UserContent, GPT_35_TURBO, GPT_35_TURBO_0125, GPT_35_TURBO_1106,
    GPT_35_TURBO_INSTRUCT, GPT_4, GPT_4O, GPT_4O_2024_05_13, GPT_4O_MINI, GPT_4_0125_PREVIEW,
    GPT_4_0613, GPT_4_1106_PREVIEW, GPT_4_1106_VISION_PREVIEW, GPT_4_32K, GPT_4_32K_0613,
    GPT_4_TURBO, GPT_4_TURBO_2024_04_09, GPT_4_TURBO_PREVIEW, GPT_4_VISION_PREVIEW, O1,
    O1_2024_12_17, O1_MINI, O1_MINI_2024_09_12, O1_PREVIEW, O1_PREVIEW_2024_09_12, O3_MINI,
    O3_MINI_2025_01_31,
};

pub use embedding::{
    EmbeddingModel, EmbeddingResponse, TEXT_EMBEDDING_3_LARGE, TEXT_EMBEDDING_3_SMALL, TEXT_EMBEDDING_ADA_002
};

pub use transcription::{
    TranscriptionModel, TranscriptionResponse, WHISPER_1
};