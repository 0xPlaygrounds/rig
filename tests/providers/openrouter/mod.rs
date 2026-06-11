mod support;

mod cassette {
    mod agent;
    mod document_file_data;
    mod document_ordering;
    mod extractor;
    mod extractor_usage;
    mod models;
    mod multi_extract;
    mod multimodal;
    mod openai_responses_compat;
    mod permission_control;
    mod provider_selection;
    mod reasoning_roundtrip;
    mod reasoning_tool_roundtrip;
    mod request_hook;
    mod streaming;
    mod streaming_tools;
    mod transcription;
    mod typed_prompt_tools;
}

#[cfg(feature = "audio")]
mod audio_generation;
mod document_file_data;
mod file_id;

pub(super) const DEFAULT_MODEL: &str = "openai/gpt-4o-mini";
pub(super) const TOOL_MODEL: &str = "openai/gpt-4o";
