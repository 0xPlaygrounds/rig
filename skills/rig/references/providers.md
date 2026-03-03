# Rig Provider Reference

## Client Initialization

All providers follow the same pattern:

```rust
// From environment variable (recommended)
let client = openai::Client::from_env();  // reads OPENAI_API_KEY

// Explicit API key
let client = openai::Client::new("sk-...");
```

## Provider Details

### OpenAI

```rust
use rig::providers::openai;

let client = openai::Client::from_env();  // OPENAI_API_KEY

// Completion models
let agent = client.agent(openai::GPT_4O).build();
let agent = client.agent(openai::GPT_4O_MINI).build();
let agent = client.agent(openai::GPT_5).build();

// Embedding models
let embedder = client.embedding_model(openai::TEXT_EMBEDDING_ADA_002);
let embedder = client.embedding_model(openai::TEXT_EMBEDDING_3_SMALL);
```

### Anthropic

```rust
use rig::providers::anthropic;

let client = anthropic::Client::from_env();  // ANTHROPIC_API_KEY

let agent = client.agent(anthropic::CLAUDE_4_OPUS).build();
let agent = client.agent(anthropic::CLAUDE_4_SONNET).build();
let agent = client.agent(anthropic::CLAUDE_3_5_HAIKU).build();
```

### Cohere

```rust
use rig::providers::cohere;

let client = cohere::Client::from_env();  // COHERE_API_KEY

let agent = client.agent(cohere::COMMAND_R_PLUS).build();
let agent = client.agent(cohere::COMMAND_R).build();
```

### Mistral

```rust
use rig::providers::mistral;

let client = mistral::Client::from_env();  // MISTRAL_API_KEY

let agent = client.agent(mistral::MISTRAL_LARGE).build();
let embedder = client.embedding_model(mistral::MISTRAL_EMBED);
```

### Gemini

```rust
use rig::providers::gemini;

let client = gemini::Client::from_env();  // GEMINI_API_KEY

let agent = client.agent("gemini-2.0-flash").build();
```

### Ollama (local)

```rust
use rig::providers::ollama;

let client = ollama::Client::from_env();  // OLLAMA_API_BASE_URL (default: http://localhost:11434)

let agent = client.agent("llama3.2").build();
```

### OpenRouter

```rust
use rig::providers::openrouter;

let client = openrouter::Client::from_env();  // OPENROUTER_API_KEY

let agent = client.agent("anthropic/claude-3.5-sonnet").build();
```

### Together

```rust
use rig::providers::together;

let client = together::Client::from_env();  // TOGETHER_API_KEY

let agent = client.agent("meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo").build();
```

### Groq

```rust
use rig::providers::groq;

let client = groq::Client::from_env();  // GROQ_API_KEY

let agent = client.agent("llama-3.1-70b-versatile").build();
```

### Perplexity

```rust
use rig::providers::perplexity;

let client = perplexity::Client::from_env();  // PERPLEXITY_API_KEY

let agent = client.agent("llama-3.1-sonar-large-128k-online").build();
```

### DeepSeek

```rust
use rig::providers::deepseek;

let client = deepseek::Client::from_env();  // DEEPSEEK_API_KEY

let agent = client.agent("deepseek-chat").build();
```

### xAI (Grok)

```rust
use rig::providers::xai;

let client = xai::Client::from_env();  // XAI_API_KEY

let agent = client.agent("grok-2").build();
```

## Common Client Methods

```rust
// Get a completion model handle
let model = client.completion_model("model-name");

// Get an embedding model handle
let embedder = client.embedding_model("model-name");

// Start building an agent
let builder = client.agent("model-name");

// Start building an extractor
let builder = client.extractor::<MyType>("model-name");
```

## Model Override at Request Time

The model can be overridden per-request:

```rust
let response = model
    .completion_request("Hello")
    .model("gpt-4o-mini")  // Override for this request only
    .send()
    .await?;
```
