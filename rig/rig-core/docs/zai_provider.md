# Zai Provider

The Zai provider provides an integration with the Zai API, which offers GLM (General Language Model) models with state-of-the-art capabilities for coding, reasoning, and agentic tasks.

## Overview

Zai provides an Anthropic-compatible API with the following flagship models:

- **GLM-4.7** - Latest flagship model with SOTA coding, reasoning, and agentic capabilities
- **GLM-4.6** - High performance, strong coding, 200K context
- **GLM-4.5** - Base model in the 4.5 series
- **GLM-4.5-X** - Enhanced version of GLM-4.5
- **GLM-4.5-Air** - Lightweight model for faster responses
- **GLM-4.5-AirX** - Ultra-lightweight model
- **GLM-4.5-Flash** - Fastest model for quick responses

## API Endpoint

The Zai provider uses the Anthropic-compatible API endpoint: `https://api.z.ai/api/anthropic`

## Installation

The Zai provider is included in the `rig-core` crate. No additional dependencies are required.

## Configuration

### API Key

Set your Zai API key as an environment variable:

```bash
export ZAI_API_KEY="your-zai-api-key"
```

### Client Initialization

```rust
use rig::providers::zai;

// Initialize client from environment
let client = zai::Client::from_env();

// Or initialize with a specific API key
let client = zai::Client::builder()
    .api_key("your-api-key")
    .build()
    .unwrap();
```

## Usage Examples

### Basic Agent

```rust
use rig::prelude::*;
use rig::providers::zai;

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    let client = zai::Client::from_env();
    
    let agent = client
        .agent(zai::completion::GLM_4_7)
        .preamble("You are a helpful assistant.")
        .temperature(0.7)
        .build();
    
    let response = agent.prompt("Explain quantum computing in simple terms.").await?;
    println!("{response}");
    
    Ok(())
}
```

### Streaming Responses

```rust
use rig::prelude::*;
use rig::agent::stream_to_stdout;
use rig::providers::zai;

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    let client = zai::Client::from_env();
    
    let agent = client
        .agent(zai::completion::GLM_4_7)
        .preamble("Be precise and concise.")
        .build();
    
    let mut stream = agent
        .stream_prompt("What is the capital of France?")
        .await;
    
    let res = stream_to_stdout(&mut stream).await?;
    println!("Token usage: {usage:?}", usage = res.usage());
    
    Ok(())
}
```

### Agent with Tools

```rust
use rig::prelude::*;
use rig::providers::zai;
use rig::tool::Tool;
use serde::{Deserialize, Serialize};
use serde_json::json;

#[derive(Deserialize)]
struct CalculatorArgs {
    x: f64,
    y: f64,
}

#[derive(Debug, thiserror::Error)]
#[error("Calculator error")]
struct CalculatorError;

#[derive(Deserialize, Serialize)]
struct AddTool;

impl Tool for AddTool {
    const NAME: &'static str = "add";
    type Error = CalculatorError;
    type Args = CalculatorArgs;
    type Output = f64;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: "add".to_string(),
            description: "Add two numbers together".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "x": {"type": "number"},
                    "y": {"type": "number"}
                },
                "required": ["x", "y"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        Ok(args.x + args.y)
    }
}

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    let client = zai::Client::from_env();
    
    let agent = client
        .agent(zai::completion::GLM_4_7)
        .preamble("You are a helpful calculator assistant.")
        .tool(AddTool)
        .build();
    
    let response = agent.prompt("What is 123.45 + 678.90?").await?;
    println!("{response}");
    
    Ok(())
}
```

### Completion Model

```rust
use rig::prelude::*;
use rig::providers::zai;
use rig::completion::{Prompt, Message};

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    let client = zai::Client::from_env();
    
    let model = client.completion_model(zai::completion::GLM_4_6);
    
    let prompt = Prompt::new("Write a haiku about artificial intelligence.");
    let response = model.completion(prompt).await?;
    
    println!("{}", response.text);
    println!("Tokens: {}", response.usage.unwrap().token_usage());
    
    Ok(())
}
```

## Model Specifications

| Model | Max Tokens | Context | Best For |
|-------|-----------|---------|----------|
| GLM-4.7 | 64,000 | Large | Complex reasoning, coding, agentic tasks |
| GLM-4.6 | 64,000 | 200K | High-performance coding, large context |
| GLM-4.5 | 32,000 | Large | General purpose tasks |
| GLM-4.5-X | 32,000 | Large | Enhanced performance |
| GLM-4.5-Air | 32,000 | Large | Balanced performance/speed |
| GLM-4.5-AirX | 32,000 | Large | Lightweight tasks |
| GLM-4.5-Flash | 8,192 | Large | Fast responses, simple tasks |

## Advanced Configuration

### Custom API Version

```rust
let client = zai::Client::builder()
    .api_key("your-api-key")
    .zai_version("2023-06-01")
    .build()
    .unwrap();
```

### Beta Features

```rust
let client = zai::Client::builder()
    .api_key("your-api-key")
    .zai_beta("prompt-caching-2024-07-31")
    .zai_beta("max-tokens-3-5-sonnet-2024-07-15")
    .build()
    .unwrap();
```

### Prompt Caching

```rust
let agent = client
    .agent(zai::completion::GLM_4_7)
    .preamble("You are a helpful assistant with extensive knowledge.")
    .with_prompt_caching()
    .build();
```

## API Compatibility

The Zai provider is compatible with the Anthropic API, which means:

- It uses the same request/response format
- It supports the same features like streaming, tools, and prompt caching
- It uses the same `x-api-key` header for authentication
- It follows the same message structure with roles (user, assistant)

## Error Handling

```rust
use rig::completion::CompletionError;

match agent.prompt("Your question").await {
    Ok(response) => println!("{response}"),
    Err(CompletionError::ProviderError(msg)) => {
        eprintln!("Provider error: {}", msg);
    }
    Err(CompletionError::RequestError(err)) => {
        eprintln!("Request error: {}", err);
    }
    Err(err) => {
        eprintln!("Other error: {}", err);
    }
}
```

## Examples

See the following examples in the `rig-core/examples` directory:

- `zai_agent.rs` - Basic agent usage
- `zai_streaming.rs` - Streaming responses
- `zai_streaming_with_tools.rs` - Agent with tools and streaming

## Support

For issues or questions about the Zai provider, please refer to the main Rig documentation or open an issue on the repository.