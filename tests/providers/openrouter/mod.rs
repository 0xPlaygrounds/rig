mod agent;
mod extractor;
mod extractor_usage;
mod models;
mod multi_extract;
mod multimodal;
mod permission_control;
mod provider_selection;
mod reasoning_roundtrip;
mod reasoning_tool_roundtrip;
mod request_hook;
mod streaming;
mod streaming_tools;
mod typed_prompt_tools;

pub(super) const DEFAULT_MODEL: &str = "openai/gpt-4o-mini";
pub(super) const TOOL_MODEL: &str = "openai/gpt-4o";
