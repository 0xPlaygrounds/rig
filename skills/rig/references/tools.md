# Rig Tool Reference

## Tool Trait

```rust
use rig::tool::Tool;

pub trait Tool: Sized + WasmCompatSend + WasmCompatSync {
    const NAME: &'static str;

    type Error: std::error::Error + WasmCompatSend + WasmCompatSync + 'static;
    type Args: for<'a> Deserialize<'a> + WasmCompatSend + WasmCompatSync;
    type Output: Serialize;

    fn name(&self) -> String {
        Self::NAME.to_string()  // default impl; override only if needed
    }

    async fn definition(&self, prompt: String) -> ToolDefinition;

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error>;
}
```

## ToolDefinition

```rust
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,  // JSON Schema
}
```

## Example: Calculator Tool

```rust
use rig::tool::{Tool, ToolDefinition};
use serde::{Deserialize, Serialize};
use serde_json::json;

#[derive(Debug, thiserror::Error)]
#[error("Math error: {0}")]
struct MathError(String);

#[derive(Deserialize)]
struct CalcArgs {
    x: f64,
    y: f64,
    operation: String,
}

struct Calculator;

impl Tool for Calculator {
    const NAME: &'static str = "calculator";

    type Error = MathError;
    type Args = CalcArgs;
    type Output = f64;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: "Perform arithmetic operations".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "x": { "type": "number", "description": "First operand" },
                    "y": { "type": "number", "description": "Second operand" },
                    "operation": {
                        "type": "string",
                        "enum": ["add", "subtract", "multiply", "divide"],
                        "description": "Operation to perform"
                    }
                },
                "required": ["x", "y", "operation"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        match args.operation.as_str() {
            "add" => Ok(args.x + args.y),
            "subtract" => Ok(args.x - args.y),
            "multiply" => Ok(args.x * args.y),
            "divide" => {
                if args.y == 0.0 {
                    Err(MathError("Division by zero".to_string()))
                } else {
                    Ok(args.x / args.y)
                }
            }
            op => Err(MathError(format!("Unknown operation: {op}"))),
        }
    }
}
```

## Attaching Tools

```rust
// Single tool
let agent = client.agent(openai::GPT_4O)
    .tool(Calculator)
    .build();

// Multiple tools
let agent = client.agent(openai::GPT_4O)
    .tool(Calculator)
    .tool(WebSearch)
    .build();

// Dynamic tools (agent selects from a vector-indexed tool set)
let tool_index = vector_store.index(embedding_model);
let toolset = ToolSet::builder()
    .dynamic_tool(Calculator)
    .dynamic_tool(WebSearch)
    .build();
let agent = client.agent(openai::GPT_4O)
    .dynamic_tools(3, tool_index, toolset)
    .build();
```

## Tool Choice

```rust
use rig::completion::ToolChoice;

let agent = client.agent(openai::GPT_4O)
    .tool(Calculator)
    .tool_choice(ToolChoice::Auto)      // Let model decide (default)
    // .tool_choice(ToolChoice::Required)   // Force tool use
    // .tool_choice(ToolChoice::None)       // Disable tools
    // .tool_choice(ToolChoice::Specific { function_names: vec!["calculator".into()] })
    .build();
```

## ToolEmbedding (for RAG-able tools)

For tools that can be dynamically selected via vector search:

```rust
pub trait ToolEmbedding: Tool {
    type InitError: std::error::Error + WasmCompatSend + WasmCompatSync + 'static;
    type Context: for<'a> Deserialize<'a> + Serialize;
    type State: WasmCompatSend;

    fn embedding_docs(&self) -> Vec<String>;
    fn context(&self) -> Self::Context;
    fn init(state: Self::State, context: Self::Context) -> Result<Self, Self::InitError>;
}
```
