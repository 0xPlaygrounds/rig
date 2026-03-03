# Rig Advanced Patterns

## Multi-Turn Agent

For conversational agents that maintain chat history:

```rust
use rig::completion::Chat;
use rig::message::Message;

let agent = client.agent(openai::GPT_4O)
    .preamble("You are a helpful assistant.")
    .build();

let mut history: Vec<Message> = vec![];

// First turn
let response = agent.chat("Hi, I'm Alice.", history.clone()).await?;
history.push("Hi, I'm Alice.".into());
history.push(response.into());

// Second turn (agent remembers context)
let response = agent.chat("What's my name?", history.clone()).await?;
// response: "Your name is Alice."
```

## Structured Extraction

Extract typed data from unstructured text:

```rust
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Deserialize, Serialize, JsonSchema)]
struct Person {
    pub name: Option<String>,
    pub age: Option<u8>,
    pub occupation: Option<String>,
}

let extractor = client.extractor::<Person>(openai::GPT_4O)
    .preamble("Extract person details from text.")
    .build();

let person = extractor
    .extract("John Doe is a 30-year-old software engineer.")
    .await?;
```

## Agent as Tool (Multi-Agent)

`Agent` implements the `Tool` trait directly — pass it to another agent's `.tool()`:

```rust
let research_agent = client.agent(openai::GPT_4O)
    .preamble("You are a research specialist.")
    .name("researcher")
    .description("Research a topic in depth")
    .build();

let orchestrator = client.agent(openai::GPT_4O)
    .preamble("You coordinate research tasks.")
    .tool(research_agent)  // Agent implements Tool directly
    .build();
```

## Prompt Hooks (Observability)

Observe and control the agent execution lifecycle. `PromptHook<M>` is generic
over the completion model type. All methods have defaults — implement only what
you need.

```rust
use rig::agent::prompt_request::{HookAction, ToolCallHookAction, hooks::PromptHook};
use rig::completion::CompletionModel;
use rig::message::Message;
use rig::WasmCompatSend;
use std::future::Future;

#[derive(Clone)]
struct LoggingHook;

impl<M: CompletionModel> PromptHook<M> for LoggingHook {
    fn on_completion_call(
        &self,
        prompt: &Message,
        history: &[Message],
    ) -> impl Future<Output = HookAction> + WasmCompatSend {
        async {
            println!("Sending prompt to model...");
            HookAction::cont()
        }
    }

    fn on_tool_call(
        &self,
        tool_name: &str,
        tool_call_id: Option<String>,
        internal_call_id: &str,
        args: &str,
    ) -> impl Future<Output = ToolCallHookAction> + WasmCompatSend {
        async move {
            println!("Tool call: {tool_name}({args})");
            ToolCallHookAction::cont()  // or ToolCallHookAction::skip("reason")
        }
    }
}

// Attach hook to a request
let response = agent
    .prompt("Do something")
    .with_hook(LoggingHook)
    .await?;
```

## Streaming

Stream responses chunk-by-chunk. The stream yields `MultiTurnStreamItem<R>`
which wraps nested content types:

```rust
use rig::streaming::StreamedAssistantContent;
use rig::agent::prompt_request::streaming::MultiTurnStreamItem;
use futures::StreamExt;

let mut stream = agent.stream_prompt("Tell me a story").await?;

while let Some(chunk) = stream.next().await {
    match chunk? {
        MultiTurnStreamItem::StreamAssistantItem(
            StreamedAssistantContent::Text(text)
        ) => {
            print!("{}", text.text);
        }
        MultiTurnStreamItem::StreamAssistantItem(
            StreamedAssistantContent::ToolCall { tool_call, .. }
        ) => {
            println!("Tool: {} args: {}", tool_call.function.name, tool_call.function.arguments);
        }
        MultiTurnStreamItem::FinalResponse(resp) => {
            println!("\nFinal: {}", resp.response());
        }
        _ => {}
    }
}
```

## Chaining Agents

Sequential agent execution:

```rust
let summarizer = client.agent(openai::GPT_4O)
    .preamble("Summarize the input text in one paragraph.")
    .build();

let translator = client.agent(openai::GPT_4O)
    .preamble("Translate the input text to French.")
    .build();

// Chain: summarize then translate
let long_text = "A very long document that needs summarizing...";
let summary = summarizer.prompt(long_text).await?;
let french_summary = translator.prompt(&summary).await?;
```

## Error Handling

Always use proper error types:

```rust
use rig::completion::PromptError;

match agent.prompt("Hello").await {
    Ok(response) => println!("{response}"),
    Err(PromptError::CompletionError(e)) => eprintln!("Completion error: {e}"),
    Err(e) => eprintln!("Error: {e}"),
}
```

## WASM Compatibility

When building for WebAssembly, use Rig's compatibility traits:

```rust
// Use these instead of Send/Sync
use rig::{WasmCompatSend, WasmCompatSync, WasmBoxedFuture};
use std::future::Future;

pub trait MyTrait: WasmCompatSend + WasmCompatSync {
    fn do_thing(&self) -> impl Future<Output = ()> + WasmCompatSend;
}
```
