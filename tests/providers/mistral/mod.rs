mod agent;
mod agent_tool_sessions;
mod capability_edges;
mod embedding_dimensions;
#[cfg(feature = "derive")]
mod embeddings;
mod extractor;
mod extractor_usage;
mod models;
mod multi_extract;
mod multimodal_content;
mod permission_control;
mod reasoning_content;
mod reasoning_roundtrip;
mod reasoning_tool_roundtrip;
mod request_hook;
mod response_identity_edge;
mod streaming;
mod streaming_tools;
mod support;
mod transcription;
mod typed_prompt_tools;

pub(super) const DEFAULT_MODEL: &str = "mistral-small-latest";
pub(super) const TOOL_MODEL: &str = DEFAULT_MODEL;
