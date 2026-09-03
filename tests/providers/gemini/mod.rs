mod agent_run_support;
mod hook_stress_support;
mod support;
mod tools_support;

mod cassette {
    mod agent;
    mod agent_run_recovery;
    mod agent_run_resume;
    mod agent_run_stepping;
    mod agent_run_streamed;
    mod agent_tools_e2e;
    mod cached_content_matrix;
    mod chat_history;
    mod code_execution_matrix;
    mod corpus_serving;
    mod document_ordering;
    mod dynamic_tools;
    mod embedding_matrix;
    mod embeddings;
    mod error_envelope;
    mod extractor;
    mod generate_behaviors;
    mod generate_sessions;
    mod generate_tool_args;
    mod generate_tool_modes;
    mod hook_stress;
    mod hook_stress_context;
    mod hook_stress_patch;
    mod hook_stress_streaming;
    mod hook_stress_tools;
    #[cfg(feature = "image")]
    mod image_generation;
    mod interactions_api;
    mod interactions_raw_capture_matrix;
    mod interactions_raw_stream_capture_matrix;
    mod lifecycle_matrix;
    mod models;
    mod multi_turn_streaming;
    mod prompt_caching;
    mod raw_capture_agent_matrix;
    mod raw_capture_matrix;
    mod raw_completion_parity_matrix;
    mod raw_stream_capture_matrix;
    mod reasoning_roundtrip;
    mod reasoning_tool_roundtrip;
    mod regression_suite;
    mod response_identity;
    mod stream_terminal_matrix;
    mod streaming;
    mod streaming_grammar;
    mod streaming_multimodal_tool_results;
    mod streaming_tools;
    mod structured_output;
    mod thought_text_matrix;
    mod tool_choice;
    mod tool_definitions;
    mod tool_hooks;
    mod tool_server;
    mod transcription;
    mod turn_termination_matrix;
}

mod live {
    mod image_tool_result;
}
