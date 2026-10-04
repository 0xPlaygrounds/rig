mod agent_run_support;
mod hook_stress_support;
mod support;
mod tools_support;

mod cassette {
    mod adversarial_matrix;
    mod agent;
    mod agent_run_recovery;
    mod agent_run_stepping;
    mod agent_run_streamed;
    mod auto_caching;
    mod cached_content_matrix;
    mod chat_history;
    mod code_execution_matrix;
    mod corpus_breadth;
    mod corpus_delta;
    mod corpus_faults;
    mod corpus_matrix;
    mod corpus_matrix_checkpoint;
    mod corpus_matrix_image;
    mod corpus_matrix_long_loop;
    mod corpus_retrieval;
    mod corpus_serving;
    mod document_ordering;
    mod dynamic_tools;
    mod ecs_extractor;
    mod ecs_faults;
    mod ecs_matrix;
    mod ecs_matrix_checkpoint;
    mod ecs_matrix_image;
    mod ecs_matrix_long_loop;
    mod ecs_parity;
    mod ecs_stress_context;
    #[path = "ecs_stress/main_golden.rs"]
    mod ecs_stress_main_golden;
    #[path = "ecs_stress/runtime.rs"]
    mod ecs_stress_runtime;
    #[path = "ecs_stress/streaming.rs"]
    mod ecs_stress_streaming_runtime;
    mod ecs_tools_e2e;
    mod embedding_matrix;
    mod embeddings;
    mod extractor;
    mod generate_behaviors;
    mod generate_tool_args;
    mod history_survival_matrix;
    mod hook_stress;
    mod image_input_matrix;
    mod interactions_api;
    mod interactions_raw_capture_matrix;
    mod live_facts;
    mod models;
    mod multi_turn_streaming;
    mod prompt_caching;
    mod raw_capture_matrix;
    mod raw_stream_capture_matrix;
    mod reasoning_tool_roundtrip;
    mod regression_suite;
    mod response_identity;
    mod restated_replies;
    mod stateful_chain_matrix;
    mod stream_faults;
    mod stream_terminal_matrix;
    mod streaming_grammar;
    mod streaming_multimodal_tool_results;
    mod structured_output;
    mod text_signature_matrix;
    mod thought_text_matrix;
    mod tool_choice;
    mod tool_definitions;
}

mod live {
    mod image_tool_result;
}
