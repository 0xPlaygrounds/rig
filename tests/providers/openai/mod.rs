mod support;

mod cassette {
    mod agent;
    mod chat_history;
    mod completions_api;
    mod document_ordering;
    mod extractor;
    mod extractor_usage;
    mod gpt_5_6_reasoning;
    mod models;
    mod multi_extract;
    mod openai_compatible_reasoning_content;
    mod permission_control;
    mod reasoning_roundtrip;
    mod reasoning_tool_roundtrip;
    mod request_hook;
    mod response_schema;
    mod responses_behaviors;
    mod responses_input_item;
    mod responses_sessions;
    mod responses_tool_args;
    mod responses_tool_choice;
    mod streaming;
    mod streaming_tools;
    mod structured_output;
    mod typed_prompt_tools;
    mod url_pdf_document;
    mod vllm;
}

mod live {
    #[cfg(feature = "audio")]
    mod audio_generation;
    mod document_file_id;
    mod gpt_5_5;
    #[cfg(feature = "image")]
    mod image_generation;
    mod transcription;
    #[cfg(feature = "websocket")]
    mod websocket;
}
