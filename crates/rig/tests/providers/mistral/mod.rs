mod agent;
#[cfg(feature = "derive")]
mod embeddings;
mod extractor;
mod extractor_usage;
mod models;
mod multi_extract;
mod permission_control;
mod request_hook;
mod streaming;
mod streaming_tools;
mod transcription;
mod typed_prompt_tools;

pub(super) const DEFAULT_MODEL: &str = "mistral-small-latest";
pub(super) const TOOL_MODEL: &str = DEFAULT_MODEL;
