mod support;

mod cassette {
    mod adversarial_matrix;
    mod agent;
    mod citations;
    mod context_binding;
    mod corpus_causal;
    mod corpus_endings;
    mod corpus_hooks;
    mod corpus_host;
    mod corpus_layers;
    mod corpus_matrix_checkpoint;
    mod corpus_matrix_long_loop;
    mod corpus_memory;
    mod corpus_oracle;
    mod corpus_outcome;
    mod corpus_output;
    mod corpus_request_shape;
    mod corpus_serving;
    mod corpus_shaping;
    mod document_file_id;
    mod effect_corpus;
    mod empty_stop_sequence_matrix;
    mod history_survival_matrix;
    mod lifecycle_matrix;
    mod long_run_features;
    mod long_run_workloads;
    mod malformed_tool_args_matrix;
    mod messages_strict_tools;
    mod messages_thinking;
    mod messages_tool_args;
    mod models;
    mod opus_4_7;
    mod opus_4_8;
    mod prompt_caching;
    mod raw_capture_matrix;
    mod raw_stream_capture_matrix;
    mod reasoning_tool_roundtrip;
    mod reasoning_usage_matrix;
    mod request_override;
    mod response_identity_edge;
    mod restated_history;
    mod stop_sequence_terminal_matrix;
    mod streaming;
    mod streaming_tools;
    mod strict_schema_integrations;
    mod strict_schema_matrix;
    mod strict_schema_streaming;
    mod structured_output;
}

mod live {}
