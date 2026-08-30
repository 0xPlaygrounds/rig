mod core_run_driver;
mod dependency_graph;
#[cfg(feature = "derive")]
mod embed_macro;
mod loaders;
mod name_keyed_serializers;
mod nightly_paths_registry;
mod prompt_response_messages;
mod provider_layout;
mod reasoning_stream_stats;
// The corpus needs exactly the providers its fixtures name; gating it on the
// `providers-all` aggregate made an otherwise-complete feature set compile
// zero conformance tests, which is the silent-skip failure this suite exists
// to prevent. The file drives xai/copilot/chatgpt clients *and* imports the
// in-crate fixture modules for the other five, so it needs all eight.
#[cfg(all(
    feature = "openai",
    feature = "gemini",
    feature = "anthropic",
    feature = "cohere",
    feature = "ollama",
    feature = "xai",
    feature = "copilot",
    feature = "chatgpt"
))]
mod streaming_conformance;
#[cfg(all(
    feature = "openai",
    feature = "gemini",
    feature = "anthropic",
    feature = "cohere",
    feature = "ollama",
    feature = "xai",
    feature = "copilot",
    feature = "chatgpt"
))]
mod streaming_conformance_registry;
#[cfg(all(
    feature = "openai",
    feature = "gemini",
    feature = "anthropic",
    feature = "cohere",
    feature = "ollama"
))]
mod streaming_conformance_suites;
#[cfg(feature = "derive")]
mod tool_macro;
