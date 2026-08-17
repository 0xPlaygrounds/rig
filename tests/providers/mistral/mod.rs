mod agent;
mod agent_tool_sessions;
mod capability_edges;
#[cfg(feature = "derive")]
mod embeddings;
mod extractor;
mod extractor_usage;
mod history_roundtrip_matrix;
mod logprobs_rejection_matrix;
mod models;
mod multi_extract;
mod multimodal_content;
mod permission_control;
mod raw_capture_matrix;
mod raw_stream_capture_matrix;
mod request_hook;
mod request_shape_matrix;
mod response_identity_edge;
mod streaming;
mod streaming_tools;
mod support;
mod terminal_metadata_matrix;
mod tool_lifecycle_matrix;
mod tool_truncation_matrix;
mod transcription;
mod typed_prompt_tools;

pub(super) const DEFAULT_MODEL: &str = "mistral-small-latest";
pub(super) const TOOL_MODEL: &str = DEFAULT_MODEL;
