# LLM Library
This project implements an opinionated LLM library based on the OpenAI API specs.

# LLM library
The LLM library contains various abstractions to simplify the implementation of complex multi-agent RAG applications.

## OpenAIClient
The LLM library implements an OpenAIClient for the `reqwest::Client` type. 
```rust
// Create OpenAI client
let openai_api_key = env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY not set");
let openai_client = reqwest::Client::from_openai_api_key(openai_api_key); // Alternatively use `llm::client::Client` instead of `reqwest::Client`
```

The Client can be used to create various types of agents and embeddings that can be composed together.
```rust
// Create OpenAI client
let openai_api_key = env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY not set");
let openai_client = reqwest::Client::from_openai_api_key(openai_api_key); // Alternatively use `llm::client::Client` instead of `reqwest::Client`

// Create model
let gpt4 = openai_client.model("gpt-4");

// Prompt the model
let response = gpt4.prompt("Who are you?", vec![]).await?;
```

See the `examples/` directory for more examples.

## Chatbots and `trait Prompt`
The library also provides a utility function `cli_chatbot` to easily create a CLI chatbot (this is useful to manually test a model). The function will start a main loop that will capture user input from the terminal and prompt the model in a loop (chat history is handled automatically).

```rust
use subgraph_rag::cli_chatbot::cli_chatbot;

// Create model
let gpt4 = openai_client.model("gpt-4");

// Create a chatbot from the model
cli_chatbot(pgt4).await?;
```

Any type that implements the `Prompt` trait can be turned into a chatbot. This includes the following: `Model`, `Agent` and `RagAgent`.

To create a chatbot for a custom model, agent or RAG, you must implement the `Prompt` trait for that type. 
```rust
pub trait Prompt {
    async fn prompt(
        &self,
        prompt: impl Into<String>,
        chat_history: Vec<Message>,
    ) -> anyhow::Result<String>;
}
```
Note: `anyhow::Result` is used throughout the library instead of the default `Result` type. This makes `Result` handling a breeze.

See `example/multi_agent.rs` for an example of a custom agent architecture that has a custom `Prompt` implementation.

# Running examples
## Install dependencies
`cargo build`

## Run the RAG application
First, export the following environment variables:
```bash
export OPENAI_API_KEY=...
```

Then, run the app with the following command:
```bash
cargo run --example simple_model
```
