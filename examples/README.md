# Rig examples

Each example is its own package. Run one with:

```sh
cargo run -p <example-name>
```

Most examples expect provider API keys in the environment (e.g. `OPENAI_API_KEY`,
`ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `COHERE_API_KEY`). See each example's source for specifics.

| Example | Description |
| --- | --- |
| `agent_autonomous` | Demonstrates an autonomous extractor loop that keeps feeding its own output back in. |
| `agent_evaluator_optimizer` | See source. |
| `agent_orchestrator` | See source. |
| `agent_parallelization` | See source. |
| `agent_prompt_chaining` | Demonstrates prompt chaining with two agents in sequence. |
| `agent_routing` | Demonstrates routing one prompt into different follow-up prompts. |
| `agent_run_stepping` | Drives the agent loop by hand with the sans-IO [`AgentRun`] state machine. |
| `agent_stream_chat` | Demonstrates `stream_chat` with prior conversation history. |
| `agent_with_agent_tool` | See source. |
| `agent_with_context` | Demonstrates adding small context documents directly to an agent. |
| `agent_with_default_max_turns` | Demonstrates extending the default agent loop budget for tool-heavy prompts. |
| `agent_with_echochambers` | See source. |
| `agent_with_loaders` | Demonstrates loading real example files into agent context. |
| `agent_with_memory_streaming` | Demonstrates Rig-managed conversation memory with streaming. |
| `agent_with_memory` | Demonstrates Rig-managed conversation memory with an in-memory backend. |
| `agent_with_tools_otel` | Agent multi-turn with tools, but with a tracing subscriber that sends all logs/traces to an OTel collector. |
| `agent_with_tools` | Demonstrates registering boxed tools on an agent. |
| `agent` | Demonstrates the smallest useful agent setup with OpenAI. |
| `calculator_chatbot` | See source. |
| `chain` | Demonstrates a retrieval-augmented pipeline with `parallel!` and `lookup`. |
| `complex_agentic_loop_claude` | See source. |
| `custom_vector_store` | Example: Implementing a custom vector store backend |
| `debate` | See source. |
| `discord_bot` | See source. |
| `enum_dispatch` | See source. |
| `extractor` | Demonstrates typed extraction and extraction with usage metadata. |
| `gemini_deep_research` | See source. |
| `gemini_default_api_recovery` | Demonstrates recovering from Gemini emitting a legacy `default_api` tool name. |
| `gemini_extractor_with_rag` | See source. |
| `gemini_nanobanana_image_generation` | See source. |
| `gemini_stream_kill_token_count` | Live Gemini example: obtaining a token-count estimate when a streaming |
| `gemini_video_understanding` | Demonstrates Gemini video understanding with provider-specific request parameters. |
| `manual_tool_calls` | Demonstrates manual tool-call handling with `Agent::completion()`. |
| `multi_agent` | See source. |
| `multi_extract` | Demonstrates fan-out structured extraction with `try_parallel!`. |
| `multi_turn_agent_extended` | See source. |
| `multi_turn_agent` | See source. |
| `openai_agent_completions_api_otel` | This example shows how you can use OpenAI's Completions API. |
| `openai_streaming_per_call_usage` | Shows how to inspect per-completion-call usage in an agent stream. |
| `openai_streaming_with_tools_otel` | See source. |
| `pdf_agent` | See source. |
| `rag_dynamic_tools_multi_turn` | See source. |
| `rag_dynamic_tools` | See source. |
| `rag_ollama` | See source. |
| `rag` | See source. |
| `reasoning_loop` | See source. |
| `request_hook` | Demonstrates observing prompt/response lifecycle events with `PromptHook`. |
| `reqwest_middleware` | Demonstrates supplying a custom reqwest client with retry middleware. |
| `rmcp_example` | An example of how you can use `rmcp` with Rig to create an MCP friendly agent. |
| `sentiment_classifier` | Demonstrates the smallest typed extractor for classification. |
| `transcription` | See source. |
| `vector_search_cohere` | Demonstrates vector search with separate Cohere document and query embeddings. |
| `vector_search_ollama` | Demonstrates vector search against a local Ollama embedding model. |
| `vector_search` | Demonstrates embedding documents and querying an in-memory vector index with OpenAI. |
