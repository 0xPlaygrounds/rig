mod agent;
mod context;
mod extractor;
mod extractor_usage;
mod loaders;
mod multi_extract;
mod permission_control;
mod request_hook;
mod streaming;
mod streaming_reasoning;
mod streaming_tools;
mod tools;
mod transcription;
mod typed_prompt_tools;

use rig::providers::groq;

pub(super) const DEFAULT_MODEL: &str = groq::DEEPSEEK_R1_DISTILL_LLAMA_70B;
pub(super) const TOOL_MODEL: &str = DEFAULT_MODEL;
