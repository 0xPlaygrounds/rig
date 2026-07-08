use super::client::HuggingFaceExt;
use crate::providers::openai;
use serde::Deserialize;
use serde_json::Value;

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum ApiResponse<T> {
    Ok(T),
    Err(Value),
}

// ================================================================
// Huggingface Completion API
// ================================================================

// Conversational LLMs
/// `google/gemma-2-2b-it` completion model
pub const GEMMA_2: &str = "google/gemma-2-2b-it";
/// `meta-llama/Meta-Llama-3.1-8B-Instruct` completion model
pub const META_LLAMA_3_1: &str = "meta-llama/Meta-Llama-3.1-8B-Instruct";
/// `PowerInfer/SmallThinker-3B-Preview` completion model
pub const SMALLTHINKER_PREVIEW: &str = "PowerInfer/SmallThinker-3B-Preview";
/// `Qwen/Qwen2.5-7B-Instruct` completion model
pub const QWEN2_5: &str = "Qwen/Qwen2.5-7B-Instruct";
/// `Qwen/Qwen2.5-Coder-32B-Instruct` completion model
pub const QWEN2_5_CODER: &str = "Qwen/Qwen2.5-Coder-32B-Instruct";

// Conversational VLMs

/// `Qwen/Qwen2-VL-7B-Instruct` visual-language completion model
pub const QWEN2_VL: &str = "Qwen/Qwen2-VL-7B-Instruct";
/// `Qwen/QVQ-72B-Preview` visual-language completion model
pub const QWEN_QVQ_PREVIEW: &str = "Qwen/QVQ-72B-Preview";

/// Huggingface completion model, driven by the shared OpenAI Chat Completions
/// path. The sub-provider's completion endpoint and model-identifier mapping
/// are applied by [`HuggingFaceExt`]'s `OpenAICompatibleProvider` impl.
pub type CompletionModel<H = reqwest::Client> =
    openai::completion::GenericCompletionModel<HuggingFaceExt, H>;
