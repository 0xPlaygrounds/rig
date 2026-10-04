//! Together AI's completion model identifiers.
//!
//! ```no_run
//! use rig_core::providers::together;
//! let model = together::from_env()?.chat(together::MIXTRAL_8X7B_INSTRUCT_V0_1);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

pub const LLAMA_3_8B_CHAT_HF: &str = "meta-llama/Llama-3-8b-chat-hf";
pub const LLAMA_3_8B_INSTRUCT: &str = "meta-llama/Meta-Llama-3-8B-Instruct";
pub const LLAMA_3_70B_INSTRUCT: &str = "meta-llama/Meta-Llama-3-70B-Instruct";
pub const LLAMA_2_70B_CHAT_TOGETHER: &str = "togethercomputer/llama-2-70b-chat";
pub const MIXTRAL_8X7B_INSTRUCT_V0_1: &str = "mistralai/Mixtral-8x7B-Instruct-v0.1";
pub const LLAMA_3_1_8B_INSTRUCT_TURBO: &str = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo";
pub const LLAMA_3_1_70B_INSTRUCT_TURBO: &str = "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo";
pub const QWEN2_5_72B_INSTRUCT_TURBO: &str = "Qwen/Qwen2.5-72B-Instruct-Turbo";
