mod support;

mod cassette {
    mod agent;
    mod agent_tool_sessions;
    mod document_file_data;
    mod document_ordering;
    mod extractor;
    mod extractor_usage;
    mod history_roundtrip_matrix;
    mod models;
    mod multi_extract;
    mod multimodal;
    mod openai_responses_compat;
    mod permission_control;
    mod prompt_caching;
    mod provider_selection;
    mod raw_capture_matrix;
    mod raw_completion_parity_matrix;
    mod raw_stream_capture_matrix;
    mod reasoning_roundtrip;
    mod reasoning_tool_order_matrix;
    mod reasoning_tool_roundtrip;
    mod reasoning_usage_matrix;
    mod refusal_matrix;
    mod request_hook;
    mod response_identity_edge;
    mod streaming;
    mod streaming_logprobs_matrix;
    mod streaming_tools;
    mod terminal_metadata_matrix;
    mod tool_lifecycle_matrix;
    mod tool_truncation_contract_matrix;
    mod transcription;
    mod typed_prompt_tools;
}

#[cfg(feature = "audio")]
mod audio_generation;
mod document_file_data;
mod file_id;

pub(super) const DEFAULT_MODEL: &str = "openai/gpt-4o-mini";
pub(super) const TOOL_MODEL: &str = "openai/gpt-4o";
