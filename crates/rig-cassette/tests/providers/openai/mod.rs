mod support;

mod regressions;

// Only the Chat Completions modules run on the Chat family's branch; the
// Responses family owns the gated rest.
mod cassette {
    mod additional_params_tools;
    mod adversarial_matrix;
    mod agent;
    mod audio_params_matrix;
    mod chat_history;
    mod chat_history_roundtrip_matrix;
    mod chat_streaming_logprobs_matrix;
    mod chat_terminal_metadata_matrix;
    mod chat_tool_lifecycle_matrix;
    mod chat_tool_truncation_matrix;
    mod completions_api;
    mod corpus_breadth;
    mod corpus_delta;
    mod corpus_faults_responses;
    mod corpus_host;
    mod corpus_matrix_chat;
    mod corpus_matrix_checkpoint_chat;
    mod corpus_matrix_checkpoint_responses;
    mod corpus_matrix_image_responses;
    mod corpus_matrix_long_loop_responses;
    mod corpus_matrix_responses;
    mod corpus_output;
    mod corpus_retrieval;
    mod corpus_serving;
    mod document_ordering;
    mod effect_corpus;
    mod embedding_matrix;
    mod error_identity_edge;
    mod gpt_5_6_reasoning;
    mod history_survival_matrix_responses;
    mod image_input_matrix;
    mod image_params_matrix;
    mod long_run_caching;
    mod long_run_workloads;
    mod max_completion_tokens_matrix;
    mod models;
    mod multi_extract;
    mod openai_compatible_dual_reasoning_keys;
    mod openai_compatible_reasoning_content;
    mod portability_matrix_chat;
    mod prompt_caching;
    mod raw_capture_agent_matrix;
    mod raw_capture_matrix;
    mod raw_completion_parity_matrix;
    mod raw_stream_capture_matrix;
    mod reasoning_roundtrip;
    mod reasoning_tool_roundtrip;
    mod refusal_matrix;
    mod response_identity;
    mod response_identity_edge;
    mod response_metadata_matrix;
    mod response_schema;
    mod responses_behaviors;
    mod responses_input_item;
    mod responses_tool_args;
    mod responses_tool_choice;
    mod stateful_chain_matrix;
    mod stateless_replay_matrix;
    mod stream_faults;
    mod streaming;
    mod streaming_grammar;
    mod streaming_grammar_chat;
    mod streaming_tools;
    mod structured_output;
    mod transcription_usage_matrix;
    mod typed_prompt_tools;
    mod web_search_citations;
    mod websocket_error_identity_matrix;
}

mod live {
    mod audio_generation;
    mod document_file_id;
    mod gpt_5_5;
    mod image_generation;
    mod streaming_tools_reasoning;
    mod transcription;
    mod websocket;
}
