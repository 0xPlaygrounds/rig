// ================================================================
//! Together AI Completion Integration
//! From [Together AI Reference](https://docs.together.ai/docs/chat-overview)
// ================================================================

use crate::{
    completion::{self, CompletionError},
    json_utils,
    providers::openai,
};

use serde_json::json;

use super::client::{together_ai_api_types::ApiResponse, Client};

// ================================================================
// Together Completion Models
// ================================================================

pub const YI_34B_CHAT: &str = "zero-one-ai/Yi-34B-Chat";
pub const OLMO_7B_INSTRUCT: &str = "allenai/OLMo-7B-Instruct";
pub const CHRONOS_HERMES_13B: &str = "Austism/chronos-hermes-13b";
pub const ML318BR: &str = "carson/ml318br";
pub const DOLPHIN_2_5_MIXTRAL_8X7B: &str = "cognitivecomputations/dolphin-2.5-mixtral-8x7b";
pub const DBRX_INSTRUCT: &str = "databricks/dbrx-instruct";
pub const DEEPSEEK_LLM_67B_CHAT: &str = "deepseek-ai/deepseek-llm-67b-chat";
pub const DEEPSEEK_CODER_33B_INSTRUCT: &str = "deepseek-ai/deepseek-coder-33b-instruct";
pub const PLATYPUS2_70B_INSTRUCT: &str = "garage-bAInd/Platypus2-70B-instruct";
pub const GEMMA_2_9B_IT: &str = "google/gemma-2-9b-it";
pub const GEMMA_2B_IT: &str = "google/gemma-2b-it";
pub const GEMMA_2_27B_IT: &str = "google/gemma-2-27b-it";
pub const GEMMA_7B_IT: &str = "google/gemma-7b-it";
pub const LLAMA_3_70B_INSTRUCT_GRADIENT_1048K: &str =
    "gradientai/Llama-3-70B-Instruct-Gradient-1048k";
pub const MYTHOMAX_L2_13B: &str = "Gryphe/MythoMax-L2-13b";
pub const MYTHOMAX_L2_13B_LITE: &str = "Gryphe/MythoMax-L2-13b-Lite";
pub const LLAVA_NEXT_MISTRAL_7B: &str = "llava-hf/llava-v1.6-mistral-7b-hf";
pub const ZEPHYR_7B_BETA: &str = "HuggingFaceH4/zephyr-7b-beta";
pub const KOALA_7B: &str = "togethercomputer/Koala-7B";
pub const VICUNA_7B_V1_3: &str = "lmsys/vicuna-7b-v1.3";
pub const VICUNA_13B_V1_5_16K: &str = "lmsys/vicuna-13b-v1.5-16k";
pub const VICUNA_13B_V1_5: &str = "lmsys/vicuna-13b-v1.5";
pub const VICUNA_13B_V1_3: &str = "lmsys/vicuna-13b-v1.3";
pub const KOALA_13B: &str = "togethercomputer/Koala-13B";
pub const VICUNA_7B_V1_5: &str = "lmsys/vicuna-7b-v1.5";
pub const CODE_LLAMA_34B_INSTRUCT: &str = "codellama/CodeLlama-34b-Instruct-hf";
pub const LLAMA_3_8B_CHAT_HF_INT4: &str = "togethercomputer/Llama-3-8b-chat-hf-int4";
pub const LLAMA_3_2_90B_VISION_INSTRUCT_TURBO: &str =
    "meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo";
pub const LLAMA_3_2_11B_VISION_INSTRUCT_TURBO: &str =
    "meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo";
pub const LLAMA_3_2_3B_INSTRUCT_TURBO: &str = "meta-llama/Llama-3.2-3B-Instruct-Turbo";
pub const LLAMA_3_8B_CHAT_HF_INT8: &str = "togethercomputer/Llama-3-8b-chat-hf-int8";
pub const LLAMA_3_1_70B_INSTRUCT_TURBO: &str = "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo";
pub const LLAMA_2_13B_CHAT: &str = "meta-llama/Llama-2-13b-chat-hf";
pub const LLAMA_3_70B_INSTRUCT_LITE: &str = "meta-llama/Meta-Llama-3-70B-Instruct-Lite";
pub const LLAMA_3_8B_CHAT_HF: &str = "meta-llama/Llama-3-8b-chat-hf";
pub const LLAMA_3_70B_CHAT_HF: &str = "meta-llama/Llama-3-70b-chat-hf";
pub const LLAMA_3_8B_INSTRUCT_TURBO: &str = "meta-llama/Meta-Llama-3-8B-Instruct-Turbo";
pub const LLAMA_3_8B_INSTRUCT_LITE: &str = "meta-llama/Meta-Llama-3-8B-Instruct-Lite";
pub const LLAMA_3_1_405B_INSTRUCT_LITE_PRO: &str =
    "meta-llama/Meta-Llama-3.1-405B-Instruct-Lite-Pro";
pub const LLAMA_2_7B_CHAT: &str = "meta-llama/Llama-2-7b-chat-hf";
pub const LLAMA_3_1_405B_INSTRUCT_TURBO: &str = "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo";
pub const LLAMA_VISION_FREE: &str = "meta-llama/Llama-Vision-Free";
pub const LLAMA_3_70B_INSTRUCT_TURBO: &str = "meta-llama/Meta-Llama-3-70B-Instruct-Turbo";
pub const LLAMA_3_1_8B_INSTRUCT_TURBO: &str = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo";
pub const CODE_LLAMA_7B_INSTRUCT_TOGETHER: &str = "togethercomputer/CodeLlama-7b-Instruct";
pub const CODE_LLAMA_34B_INSTRUCT_TOGETHER: &str = "togethercomputer/CodeLlama-34b-Instruct";
pub const CODE_LLAMA_13B_INSTRUCT: &str = "codellama/CodeLlama-13b-Instruct-hf";
pub const CODE_LLAMA_13B_INSTRUCT_TOGETHER: &str = "togethercomputer/CodeLlama-13b-Instruct";
pub const LLAMA_2_13B_CHAT_TOGETHER: &str = "togethercomputer/llama-2-13b-chat";
pub const LLAMA_2_7B_CHAT_TOGETHER: &str = "togethercomputer/llama-2-7b-chat";
pub const LLAMA_3_8B_INSTRUCT: &str = "meta-llama/Meta-Llama-3-8B-Instruct";
pub const LLAMA_3_70B_INSTRUCT: &str = "meta-llama/Meta-Llama-3-70B-Instruct";
pub const CODE_LLAMA_70B_INSTRUCT: &str = "codellama/CodeLlama-70b-Instruct-hf";
pub const LLAMA_2_70B_CHAT_TOGETHER: &str = "togethercomputer/llama-2-70b-chat";
pub const LLAMA_3_1_8B_INSTRUCT_REFERENCE: &str = "meta-llama/Meta-Llama-3.1-8B-Instruct-Reference";
pub const LLAMA_3_1_70B_INSTRUCT_REFERENCE: &str =
    "meta-llama/Meta-Llama-3.1-70B-Instruct-Reference";
pub const WIZARDLM_2_8X22B: &str = "microsoft/WizardLM-2-8x22B";
pub const MISTRAL_7B_INSTRUCT_V0_1: &str = "mistralai/Mistral-7B-Instruct-v0.1";
pub const MISTRAL_7B_INSTRUCT_V0_2: &str = "mistralai/Mistral-7B-Instruct-v0.2";
pub const MISTRAL_7B_INSTRUCT_V0_3: &str = "mistralai/Mistral-7B-Instruct-v0.3";
pub const MIXTRAL_8X7B_INSTRUCT_V0_1: &str = "mistralai/Mixtral-8x7B-Instruct-v0.1";
pub const MIXTRAL_8X22B_INSTRUCT_V0_1: &str = "mistralai/Mixtral-8x22B-Instruct-v0.1";
pub const NOUS_HERMES_2_MIXTRAL_8X7B_DPO: &str = "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO";
pub const NOUS_HERMES_LLAMA2_70B: &str = "NousResearch/Nous-Hermes-Llama2-70b";
pub const NOUS_HERMES_2_MIXTRAL_8X7B_SFT: &str = "NousResearch/Nous-Hermes-2-Mixtral-8x7B-SFT";
pub const NOUS_HERMES_LLAMA2_13B: &str = "NousResearch/Nous-Hermes-Llama2-13b";
pub const NOUS_HERMES_2_MISTRAL_DPO: &str = "NousResearch/Nous-Hermes-2-Mistral-7B-DPO";
pub const NOUS_HERMES_LLAMA2_7B: &str = "NousResearch/Nous-Hermes-llama-2-7b";
pub const NOUS_CAPYBARA_V1_9: &str = "NousResearch/Nous-Capybara-7B-V1p9";
pub const HERMES_2_THETA_LLAMA_3_70B: &str = "NousResearch/Hermes-2-Theta-Llama-3-70B";
pub const OPENCHAT_3_5: &str = "openchat/openchat-3.5-1210";
pub const OPENORCA_MISTRAL_7B_8K: &str = "Open-Orca/Mistral-7B-OpenOrca";
pub const QWEN_2_72B_INSTRUCT: &str = "Qwen/Qwen2-72B-Instruct";
pub const QWEN2_5_72B_INSTRUCT_TURBO: &str = "Qwen/Qwen2.5-72B-Instruct-Turbo";
pub const QWEN2_5_7B_INSTRUCT_TURBO: &str = "Qwen/Qwen2.5-7B-Instruct-Turbo";
pub const QWEN1_5_110B_CHAT: &str = "Qwen/Qwen1.5-110B-Chat";
pub const QWEN1_5_72B_CHAT: &str = "Qwen/Qwen1.5-72B-Chat";
pub const QWEN_2_1_5B_INSTRUCT: &str = "Qwen/Qwen2-1.5B-Instruct";
pub const QWEN_2_7B_INSTRUCT: &str = "Qwen/Qwen2-7B-Instruct";
pub const QWEN1_5_14B_CHAT: &str = "Qwen/Qwen1.5-14B-Chat";
pub const QWEN1_5_1_8B_CHAT: &str = "Qwen/Qwen1.5-1.8B-Chat";
pub const QWEN1_5_32B_CHAT: &str = "Qwen/Qwen1.5-32B-Chat";
pub const QWEN1_5_7B_CHAT: &str = "Qwen/Qwen1.5-7B-Chat";
pub const QWEN1_5_0_5B_CHAT: &str = "Qwen/Qwen1.5-0.5B-Chat";
pub const QWEN1_5_4B_CHAT: &str = "Qwen/Qwen1.5-4B-Chat";
pub const SNORKEL_MISTRAL_PAIRRM_DPO: &str = "snorkelai/Snorkel-Mistral-PairRM-DPO";
pub const SNOWFLAKE_ARCTIC_INSTRUCT: &str = "Snowflake/snowflake-arctic-instruct";
pub const ALPACA_7B: &str = "togethercomputer/alpaca-7b";
pub const OPENHERMES_2_MISTRAL_7B: &str = "teknium/OpenHermes-2-Mistral-7B";
pub const OPENHERMES_2_5_MISTRAL_7B: &str = "teknium/OpenHermes-2p5-Mistral-7B";
pub const GUANACO_65B: &str = "togethercomputer/guanaco-65b";
pub const GUANACO_13B: &str = "togethercomputer/guanaco-13b";
pub const GUANACO_33B: &str = "togethercomputer/guanaco-33b";
pub const GUANACO_7B: &str = "togethercomputer/guanaco-7b";
pub const REMM_SLERP_L2_13B: &str = "Undi95/ReMM-SLERP-L2-13B";
pub const TOPPY_M_7B: &str = "Undi95/Toppy-M-7B";
pub const SOLAR_10_7B_INSTRUCT_V1: &str = "upstage/SOLAR-10.7B-Instruct-v1.0";
pub const SOLAR_10_7B_INSTRUCT_V1_INT4: &str = "togethercomputer/SOLAR-10.7B-Instruct-v1.0-int4";
pub const WIZARDLM_13B_V1_2: &str = "WizardLM/WizardLM-13B-V1.2";

// =================================================================
// Rig Implementation Types
// =================================================================

#[derive(Clone)]
pub struct CompletionModel {
    pub(crate) client: Client,
    pub model: String,
}

impl CompletionModel {
    pub fn new(client: Client, model: &str) -> Self {
        Self {
            client,
            model: model.to_string(),
        }
    }

    pub(crate) fn create_completion_request(
        &self,
        completion_request: completion::CompletionRequest,
    ) -> Result<serde_json::Value, CompletionError> {
        let mut full_history: Vec<openai::Message> = match &completion_request.preamble {
            Some(preamble) => vec![openai::Message::system(preamble)],
            None => vec![],
        };
        if let Some(docs) = completion_request.normalized_documents() {
            let docs: Vec<openai::Message> = docs.try_into()?;
            full_history.extend(docs);
        }
        let chat_history: Vec<openai::Message> = completion_request
            .chat_history
            .into_iter()
            .map(|message| message.try_into())
            .collect::<Result<Vec<Vec<openai::Message>>, _>>()?
            .into_iter()
            .flatten()
            .collect();

        full_history.extend(chat_history);

        let mut request = if completion_request.tools.is_empty() {
            json!({
                "model": self.model,
                "messages": full_history,
                "temperature": completion_request.temperature,
            })
        } else {
            json!({
                "model": self.model,
                "messages": full_history,
                "temperature": completion_request.temperature,
                "tools": completion_request.tools.into_iter().map(openai::ToolDefinition::from).collect::<Vec<_>>(),
                "tool_choice": "auto",
            })
        };
        request = if let Some(params) = completion_request.additional_params {
            json_utils::merge(request, params)
        } else {
            request
        };
        Ok(request)
    }
}

impl completion::CompletionModel for CompletionModel {
    type Response = openai::CompletionResponse;

    #[cfg_attr(feature = "worker", worker::send)]
    async fn completion(
        &self,
        completion_request: completion::CompletionRequest,
    ) -> Result<completion::CompletionResponse<openai::CompletionResponse>, CompletionError> {
        let request = self.create_completion_request(completion_request)?;

        let response = self
            .client
            .post("/v1/chat/completions")
            .json(&request)
            .send()
            .await?;

        if response.status().is_success() {
            let t = response.text().await?;
            tracing::debug!(target: "rig", "Together completion error: {}", t);

            match serde_json::from_str::<ApiResponse<openai::CompletionResponse>>(&t)? {
                ApiResponse::Ok(response) => {
                    tracing::info!(target: "rig",
                        "Together completion token usage: {:?}",
                        response.usage.clone().map(|usage| format!("{usage}")).unwrap_or("N/A".to_string())
                    );
                    response.try_into()
                }
                ApiResponse::Error(err) => Err(CompletionError::ProviderError(err.error)),
            }
        } else {
            Err(CompletionError::ProviderError(response.text().await?))
        }
    }
}
