mod agent_run_support;
mod support;
mod tools_support;

mod cassette {
    mod agent;
    mod agent_run_recovery;
    mod agent_run_resume;
    mod agent_run_stepping;
    mod agent_run_streamed;
    mod agent_tools_e2e;
    mod chat_history;
    mod document_ordering;
    mod dynamic_tools;
    mod embeddings;
    mod extractor;
    mod image_generation;
    mod interactions_api;
    mod models;
    mod multi_turn_streaming;
    mod reasoning_roundtrip;
    mod reasoning_tool_roundtrip;
    mod streaming;
    mod streaming_multimodal_tool_results;
    mod streaming_tools;
    mod structured_output;
    mod tool_choice;
    mod tool_definitions;
    mod tool_hooks;
    mod tool_server;
    mod transcription;
}

mod live {}
