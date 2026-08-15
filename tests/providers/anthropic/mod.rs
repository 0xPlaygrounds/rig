mod support;

mod cassette {
    mod agent;
    mod default_max_turns;
    mod document_file_id;
    mod empty_end_turn;
    mod empty_stop_sequence_matrix;
    mod error_envelope;
    mod error_identity_edge;
    mod image;
    mod messages_behaviors;
    mod messages_sessions;
    mod messages_strict_tools;
    mod messages_thinking;
    mod messages_tool_args;
    mod messages_tool_choice;
    mod models;
    mod multi_turn_streaming;
    mod opus_4_7;
    mod opus_4_8;
    mod plaintext_document;
    mod prompt_caching;
    mod reasoning_roundtrip;
    mod reasoning_tool_roundtrip;
    mod regression_suite;
    mod request_override;
    mod response_identity;
    mod response_identity_edge;
    mod stop_sequence_terminal_matrix;
    mod streaming;
    mod streaming_grammar;
    mod streaming_tools;
    mod strict_schema_formats;
    mod strict_schema_integrations;
    mod strict_schema_limits;
    mod strict_schema_matrix;
    mod strict_schema_streaming;
    mod structured_output;
    mod think_tool;
    mod think_tool_with_other_tools;
    mod tool_call_rewrite_args;
    mod tool_result_rewrite;
    mod url_pdf_document;
}

mod live {}
