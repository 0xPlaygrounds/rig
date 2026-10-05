mod agent;
mod models;
mod multimodal;
mod streaming;
mod streaming_tools;
mod structured_output;
mod support;

/// The local model every Ollama cassette was recorded against.
const CASSETTE_MODEL: &str = "qwen3:4b";

mod cassette {
    mod agent;
    mod agentic;
    mod history_survival_matrix;
    mod models;
    mod portability_matrix;
    mod raw_capture_agent_matrix;
    mod raw_capture_matrix;
    mod raw_stream_capture_matrix;
    mod reasoning_roundtrip;
    mod reasoning_tool_roundtrip;
    mod streaming;
    mod streaming_grammar;
    mod structured_output;
    mod tools;
}
