mod support;

mod regressions;

mod cassette {
    mod additional_params_tools;
    mod agent;
    #[cfg(feature = "audio")]
    mod audio_params_matrix;
    mod chat_history;
    mod chat_history_roundtrip_matrix;
    mod chat_streaming_logprobs_matrix;
    mod chat_terminal_metadata_matrix;
    mod chat_tool_lifecycle_matrix;
    mod chat_tool_truncation_matrix;
    mod completions_api;
    mod document_ordering;
    mod error_envelope;
    mod error_identity_edge;
    mod extractor;
    mod extractor_usage;
    mod gpt_5_6_reasoning;
    #[cfg(feature = "image")]
    mod image_params_matrix;
    mod max_completion_tokens_matrix;
    mod models;
    mod multi_extract;
    mod openai_compatible_reasoning_content;
    mod permission_control;
    mod reasoning_roundtrip;
    mod reasoning_tool_roundtrip;
    mod refusal_matrix;
    mod regression_suite;
    mod request_hook;
    mod response_identity;
    mod response_identity_edge;
    mod response_retry;
    mod response_schema;
    mod responses_behaviors;
    mod responses_input_item;
    mod responses_sessions;
    mod responses_tool_args;
    mod responses_tool_choice;
    mod streaming;
    mod streaming_grammar;
    mod streaming_grammar_chat;
    mod streaming_tools;
    mod structured_output;
    mod transcription_usage_matrix;
    mod truncated_turn_matrix;
    mod turn_termination_matrix;
    mod typed_prompt_tools;
    mod url_pdf_document;
    mod vllm;
    #[cfg(feature = "websocket")]
    mod websocket_error_identity_matrix;
}

mod live {
    #[cfg(feature = "audio")]
    mod audio_generation;
    mod document_file_id;
    mod gpt_5_5;
    #[cfg(feature = "image")]
    mod image_generation;
    mod streaming_tools_reasoning;
    mod transcription;
    #[cfg(feature = "websocket")]
    mod websocket;
}
