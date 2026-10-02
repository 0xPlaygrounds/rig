mod support;

// The Responses family owns these regressions.
#[cfg(any())]
mod regressions;

// Only the Chat Completions modules run on the Chat family's branch; the
// Responses family owns the gated rest.
mod cassette {
    #[cfg(any())]
    mod additional_params_tools;
    #[cfg(any())]
    mod adversarial_matrix;
    #[cfg(any())]
    mod agent;
    #[cfg(any())]
    mod audio_params_matrix;
    #[cfg(any())]
    mod chat_history;
    #[cfg(any())]
    mod chat_history_roundtrip_matrix;
    #[cfg(any())]
    mod chat_streaming_logprobs_matrix;
    #[cfg(any())]
    mod chat_terminal_metadata_matrix;
    #[cfg(any())]
    mod chat_tool_lifecycle_matrix;
    #[cfg(any())]
    mod chat_tool_truncation_matrix;
    #[cfg(any())]
    mod completions_api;
    #[cfg(any())]
    mod corpus_breadth;
    #[cfg(any())]
    mod corpus_delta;
    #[cfg(any())]
    mod corpus_faults_chat;
    #[cfg(any())]
    mod corpus_faults_responses;
    #[cfg(any())]
    mod corpus_host;
    #[cfg(any())]
    mod corpus_matrix_chat;
    #[cfg(any())]
    mod corpus_matrix_checkpoint_chat;
    #[cfg(any())]
    mod corpus_matrix_checkpoint_responses;
    #[cfg(any())]
    mod corpus_matrix_image_chat;
    #[cfg(any())]
    mod corpus_matrix_image_responses;
    #[cfg(any())]
    mod corpus_matrix_long_loop_chat;
    #[cfg(any())]
    mod corpus_matrix_long_loop_responses;
    #[cfg(any())]
    mod corpus_matrix_responses;
    #[cfg(any())]
    mod corpus_output;
    #[cfg(any())]
    mod corpus_retrieval;
    #[cfg(any())]
    mod corpus_serving;
    #[cfg(any())]
    mod document_ordering;
    #[cfg(any())]
    mod ecs_chat_tool_lifecycle;
    #[cfg(any())]
    mod ecs_extractor;
    #[cfg(any())]
    mod ecs_extractor_usage;
    #[cfg(any())]
    mod ecs_faults_chat;
    #[cfg(any())]
    mod ecs_faults_responses;
    #[cfg(any())]
    mod ecs_lifecycle;
    #[cfg(any())]
    mod ecs_matrix_chat;
    #[cfg(any())]
    mod ecs_matrix_checkpoint_chat;
    #[cfg(any())]
    mod ecs_matrix_checkpoint_responses;
    #[cfg(any())]
    mod ecs_matrix_extra_chat;
    #[cfg(any())]
    mod ecs_matrix_extra_responses;
    #[cfg(any())]
    mod ecs_matrix_image_chat;
    #[cfg(any())]
    mod ecs_matrix_image_responses;
    #[cfg(any())]
    mod ecs_matrix_long_loop_chat;
    #[cfg(any())]
    mod ecs_matrix_long_loop_responses;
    #[cfg(any())]
    mod ecs_matrix_responses;
    #[cfg(any())]
    mod ecs_ordering;
    #[cfg(any())]
    mod ecs_parity;
    #[cfg(any())]
    mod ecs_prompt_caching;
    #[cfg(any())]
    mod ecs_stream_faults;
    #[cfg(any())]
    mod ecs_termination;
    #[cfg(any())]
    mod effect_corpus;
    #[cfg(any())]
    mod embedding_matrix;
    #[cfg(any())]
    mod error_envelope;
    #[cfg(any())]
    mod error_identity_edge;
    #[cfg(any())]
    mod extractor;
    #[cfg(any())]
    mod extractor_usage;
    #[cfg(any())]
    mod gpt_5_6_reasoning;
    #[cfg(any())]
    mod history_survival_matrix_chat;
    #[cfg(any())]
    mod history_survival_matrix_responses;
    #[cfg(any())]
    mod image_input_matrix;
    #[cfg(any())]
    mod image_params_matrix;
    #[cfg(any())]
    mod lifecycle_matrix;
    #[cfg(any())]
    mod long_run_caching;
    #[cfg(any())]
    mod long_run_features;
    #[cfg(any())]
    mod long_run_workloads;
    #[cfg(any())]
    mod max_completion_tokens_matrix;
    #[cfg(any())]
    mod models;
    #[cfg(any())]
    mod multi_extract;
    #[cfg(any())]
    mod openai_compatible_dual_reasoning_keys;
    #[cfg(any())]
    mod openai_compatible_reasoning_content;
    #[cfg(any())]
    mod permission_control;
    #[cfg(any())]
    mod portability_matrix_chat;
    #[cfg(any())]
    mod portability_matrix_responses;
    #[cfg(any())]
    mod prompt_caching;
    #[cfg(any())]
    mod raw_capture_agent_matrix;
    #[cfg(any())]
    mod raw_capture_matrix;
    #[cfg(any())]
    mod raw_completion_parity_matrix;
    #[cfg(any())]
    mod raw_stream_capture_matrix;
    #[cfg(any())]
    mod reasoning_roundtrip;
    #[cfg(any())]
    mod reasoning_tool_roundtrip;
    #[cfg(any())]
    mod refusal_matrix;
    #[cfg(any())]
    mod regression_suite;
    #[cfg(any())]
    mod request_hook;
    #[cfg(any())]
    mod request_identity_matrix;
    #[cfg(any())]
    mod response_identity;
    #[cfg(any())]
    mod response_identity_edge;
    #[cfg(any())]
    mod response_metadata_matrix;
    #[cfg(any())]
    mod response_retry;
    #[cfg(any())]
    mod response_schema;
    #[cfg(any())]
    mod responses_behaviors;
    #[cfg(any())]
    mod responses_input_item;
    #[cfg(any())]
    mod responses_sessions;
    #[cfg(any())]
    mod responses_tool_args;
    #[cfg(any())]
    mod responses_tool_choice;
    #[cfg(any())]
    mod session_matrix;
    #[cfg(any())]
    mod stateful_chain_matrix;
    #[cfg(any())]
    mod stateless_replay_matrix;
    #[cfg(any())]
    mod stream_faults;
    #[cfg(any())]
    mod streaming;
    #[cfg(any())]
    mod streaming_grammar;
    #[cfg(any())]
    mod streaming_grammar_chat;
    #[cfg(any())]
    mod streaming_tools;
    #[cfg(any())]
    mod strict_tool_matrix;
    #[cfg(any())]
    mod structured_output;
    #[cfg(any())]
    mod transcription_usage_matrix;
    #[cfg(any())]
    mod truncated_turn_matrix;
    #[cfg(any())]
    mod turn_termination_matrix;
    #[cfg(any())]
    mod typed_prompt_tools;
    #[cfg(any())]
    mod url_pdf_document;
    #[cfg(any())]
    mod vllm;
    #[cfg(any())]
    mod web_search_citations;
    #[cfg(any())]
    mod websocket_error_identity_matrix;
}

mod live {
    #[cfg(any())]
    mod audio_generation;
    #[cfg(any())]
    mod document_file_id;
    #[cfg(any())]
    mod gpt_5_5;
    #[cfg(any())]
    mod image_generation;
    #[cfg(any())]
    mod streaming_tools_reasoning;
    #[cfg(any())]
    mod transcription;
    #[cfg(any())]
    mod websocket;
}
