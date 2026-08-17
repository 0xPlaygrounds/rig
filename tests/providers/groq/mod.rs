mod response_identity_edge;
mod support;

mod agent;
mod agent_tool_sessions;
mod context;
mod extractor;
mod extractor_usage;
mod loaders;
mod models;
mod multi_extract;
mod permission_control;
mod raw_capture_matrix;
mod raw_completion_parity_matrix;
mod raw_stream_capture_matrix;
mod request_hook;
mod streaming;
mod streaming_reasoning;
mod streaming_tools;
mod tools;
mod transcription;
mod typed_prompt_tools;

pub(super) const AGENT_MODEL: &str = "allam-2-7b";
pub(super) const CONTEXT_MODEL: &str = "llama-3.1-8b-instant";
pub(super) const EXTRACTOR_MODEL: &str = "openai/gpt-oss-120b";
pub(super) const EXTRACTOR_USAGE_BACKWARD_MODEL: &str = "openai/gpt-oss-20b";
pub(super) const EXTRACTOR_USAGE_WITH_USAGE_MODEL: &str = "llama-3.3-70b-versatile";
pub(super) const EXTRACTOR_USAGE_CHAT_HISTORY_MODEL: &str =
    "meta-llama/llama-4-scout-17b-16e-instruct";
pub(super) const EXTRACTOR_USAGE_SAME_DATA_MODEL: &str = "qwen/qwen3-32b";
pub(super) const EXTRACTOR_USAGE_TRACKING_MODEL: &str = "openai/gpt-oss-120b";
pub(super) const LOADERS_MODEL: &str = "groq/compound";
pub(super) const MULTI_EXTRACT_NAMES_MODEL: &str = "openai/gpt-oss-120b";
pub(super) const MULTI_EXTRACT_TOPICS_MODEL: &str = "meta-llama/llama-4-scout-17b-16e-instruct";
pub(super) const MULTI_EXTRACT_SENTIMENT_MODEL: &str = "llama-3.3-70b-versatile";
pub(super) const PERMISSION_CONTROL_PROMPT_MODEL: &str = "openai/gpt-oss-20b";
pub(super) const PERMISSION_CONTROL_STREAMING_MODEL: &str = "openai/gpt-oss-20b";
pub(super) const RAW_CAPTURE_MODEL: &str = "llama-3.1-8b-instant";
/// The raw-capture matrices were re-recorded after Groq retired
/// `llama-3.1-8b-instant`; the parity fixtures above predate that and keep
/// the old id, so the two constants deliberately differ.
pub(super) const RAW_CAPTURE_MATRIX_MODEL: &str = "allam-2-7b";
pub(super) const REQUEST_HOOK_MODEL: &str = "llama-3.1-8b-instant";
pub(super) const STREAMING_MODEL: &str = "allam-2-7b";
pub(super) const STREAMING_REASONING_MODEL: &str = "openai/gpt-oss-120b";
pub(super) const STREAMING_TOOLS_RAW_MODEL: &str = "qwen/qwen3-32b";
pub(super) const STREAMING_TOOLS_MULTI_MODEL: &str = "openai/gpt-oss-20b";
pub(super) const STREAMING_TOOLS_ORDERED_MODEL: &str = "meta-llama/llama-4-scout-17b-16e-instruct";
pub(super) const TOOLS_MODEL: &str = "llama-3.1-8b-instant";
pub(super) const TYPED_PROMPT_TOOLS_MODEL: &str = "qwen/qwen3-32b";
