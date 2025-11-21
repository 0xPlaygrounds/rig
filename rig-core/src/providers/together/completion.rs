// ================================================================
//! Together AI Completion Integration
//! From [Together AI Reference](https://docs.together.ai/docs/chat-overview)
// ================================================================

use crate::{
    completion::{self, CompletionError},
    http_client::HttpClientExt,
    json_utils, models,
    providers::openai,
};

use super::client::{Client, together_ai_api_types::ApiResponse};
use crate::completion::CompletionRequest;
use crate::streaming::StreamingCompletionResponse;
use bytes::Bytes;
use serde::{Deserialize, Serialize};
use serde_json::json;
use tracing::{Instrument, info_span};

// ================================================================
// Together Completion Models
// ================================================================

models! {
    #[allow(non_camel_case_types)]
    pub enum CompletionModels {
        YI_34B_CHAT => "zero-one-ai/Yi-34B-Chat",
        OLMO_7B_INSTRUCT => "allenai/OLMo-7B-Instruct",
        CHRONOS_HERMES_13B => "Austism/chronos-hermes-13b",
        ML318BR => "carson/ml318br",
        DOLPHIN_2_5_MIXTRAL_8X7B => "cognitivecomputations/dolphin-2.5-mixtral-8x7b",
        DBRX_INSTRUCT => "databricks/dbrx-instruct",
        DEEPSEEK_LLM_67B_CHAT => "deepseek-ai/deepseek-llm-67b-chat",
        DEEPSEEK_CODER_33B_INSTRUCT => "deepseek-ai/deepseek-coder-33b-instruct",
        PLATYPUS2_70B_INSTRUCT => "garage-bAInd/Platypus2-70B-instruct",
        GEMMA_2_9B_IT => "google/gemma-2-9b-it",
        GEMMA_2B_IT => "google/gemma-2b-it",
        GEMMA_2_27B_IT => "google/gemma-2-27b-it",
        GEMMA_7B_IT => "google/gemma-7b-it",
        LLAMA_3_70B_INSTRUCT_GRADIENT_1048K => "gradientai/Llama-3-70B-Instruct-Gradient-1048k",
        MYTHOMAX_L2_13B => "Gryphe/MythoMax-L2-13b",
        MYTHOMAX_L2_13B_LITE => "Gryphe/MythoMax-L2-13b-Lite",
        LLAVA_NEXT_MISTRAL_7B => "llava-hf/llava-v1.6-mistral-7b-hf",
        ZEPHYR_7B_BETA => "HuggingFaceH4/zephyr-7b-beta",
        KOALA_7B => "togethercomputer/Koala-7B",
        VICUNA_7B_V1_3 => "lmsys/vicuna-7b-v1.3",
        VICUNA_13B_V1_5_16K => "lmsys/vicuna-13b-v1.5-16k",
        VICUNA_13B_V1_5 => "lmsys/vicuna-13b-v1.5",
        VICUNA_13B_V1_3 => "lmsys/vicuna-13b-v1.3",
        KOALA_13B => "togethercomputer/Koala-13B",
        VICUNA_7B_V1_5 => "lmsys/vicuna-7b-v1.5",
        CODE_LLAMA_34B_INSTRUCT => "codellama/CodeLlama-34b-Instruct-hf",
        LLAMA_3_8B_CHAT_HF_INT4 => "togethercomputer/Llama-3-8b-chat-hf-int4",
        LLAMA_3_2_90B_VISION_INSTRUCT_TURBO =>
            "meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo",
        LLAMA_3_2_11B_VISION_INSTRUCT_TURBO =>
            "meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo",
        LLAMA_3_2_3B_INSTRUCT_TURBO => "meta-llama/Llama-3.2-3B-Instruct-Turbo",
        LLAMA_3_8B_CHAT_HF_INT8 => "togethercomputer/Llama-3-8b-chat-hf-int8",
        LLAMA_3_1_70B_INSTRUCT_TURBO => "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        LLAMA_2_13B_CHAT => "meta-llama/Llama-2-13b-chat-hf",
        LLAMA_3_70B_INSTRUCT_LITE => "meta-llama/Meta-Llama-3-70B-Instruct-Lite",
        LLAMA_3_8B_CHAT_HF => "meta-llama/Llama-3-8b-chat-hf",
        LLAMA_3_70B_CHAT_HF => "meta-llama/Llama-3-70b-chat-hf",
        LLAMA_3_8B_INSTRUCT_TURBO => "meta-llama/Meta-Llama-3-8B-Instruct-Turbo",
        LLAMA_3_8B_INSTRUCT_LITE => "meta-llama/Meta-Llama-3-8B-Instruct-Lite",
        LLAMA_3_1_405B_INSTRUCT_LITE_PRO =>
            "meta-llama/Meta-Llama-3.1-405B-Instruct-Lite-Pro",
        LLAMA_2_7B_CHAT => "meta-llama/Llama-2-7b-chat-hf",
        LLAMA_3_1_405B_INSTRUCT_TURBO => "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
        LLAMA_VISION_FREE => "meta-llama/Llama-Vision-Free",
        LLAMA_3_70B_INSTRUCT_TURBO => "meta-llama/Meta-Llama-3-70B-Instruct-Turbo",
        LLAMA_3_1_8B_INSTRUCT_TURBO => "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        CODE_LLAMA_7B_INSTRUCT_TOGETHER => "togethercomputer/CodeLlama-7b-Instruct",
        CODE_LLAMA_34B_INSTRUCT_TOGETHER => "togethercomputer/CodeLlama-34b-Instruct",
        CODE_LLAMA_13B_INSTRUCT => "codellama/CodeLlama-13b-Instruct-hf",
        CODE_LLAMA_13B_INSTRUCT_TOGETHER => "togethercomputer/CodeLlama-13b-Instruct",
        LLAMA_2_13B_CHAT_TOGETHER => "togethercomputer/llama-2-13b-chat",
        LLAMA_2_7B_CHAT_TOGETHER => "togethercomputer/llama-2-7b-chat",
        LLAMA_3_8B_INSTRUCT => "meta-llama/Meta-Llama-3-8B-Instruct",
        LLAMA_3_70B_INSTRUCT => "meta-llama/Meta-Llama-3-70B-Instruct",
        CODE_LLAMA_70B_INSTRUCT => "codellama/CodeLlama-70b-Instruct-hf",
        LLAMA_2_70B_CHAT_TOGETHER => "togethercomputer/llama-2-70b-chat",
        LLAMA_3_1_8B_INSTRUCT_REFERENCE => "meta-llama/Meta-Llama-3.1-8B-Instruct-Reference",
        LLAMA_3_1_70B_INSTRUCT_REFERENCE =>
            "meta-llama/Meta-Llama-3.1-70B-Instruct-Reference",
        WIZARDLM_2_8X22B => "microsoft/WizardLM-2-8x22B",
        MISTRAL_7B_INSTRUCT_V0_1 => "mistralai/Mistral-7B-Instruct-v0.1",
        MISTRAL_7B_INSTRUCT_V0_2 => "mistralai/Mistral-7B-Instruct-v0.2",
        MISTRAL_7B_INSTRUCT_V0_3 => "mistralai/Mistral-7B-Instruct-v0.3",
        MIXTRAL_8X7B_INSTRUCT_V0_1 => "mistralai/Mixtral-8x7B-Instruct-v0.1",
        MIXTRAL_8X22B_INSTRUCT_V0_1 => "mistralai/Mixtral-8x22B-Instruct-v0.1",
        NOUS_HERMES_2_MIXTRAL_8X7B_DPO => "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",
        NOUS_HERMES_LLAMA2_70B => "NousResearch/Nous-Hermes-Llama2-70b",
        NOUS_HERMES_2_MIXTRAL_8X7B_SFT => "NousResearch/Nous-Hermes-2-Mixtral-8x7B-SFT",
        NOUS_HERMES_LLAMA2_13B => "NousResearch/Nous-Hermes-Llama2-13b",
        NOUS_HERMES_2_MISTRAL_DPO => "NousResearch/Nous-Hermes-2-Mistral-7B-DPO",
        NOUS_HERMES_LLAMA2_7B => "NousResearch/Nous-Hermes-llama-2-7b",
        NOUS_CAPYBARA_V1_9 => "NousResearch/Nous-Capybara-7B-V1p9",
        HERMES_2_THETA_LLAMA_3_70B => "NousResearch/Hermes-2-Theta-Llama-3-70B",
        OPENCHAT_3_5 => "openchat/openchat-3.5-1210",
        OPENORCA_MISTRAL_7B_8K => "Open-Orca/Mistral-7B-OpenOrca",
        QWEN_2_72B_INSTRUCT => "Qwen/Qwen2-72B-Instruct",
        QWEN2_5_72B_INSTRUCT_TURBO => "Qwen/Qwen2.5-72B-Instruct-Turbo",
        QWEN2_5_7B_INSTRUCT_TURBO => "Qwen/Qwen2.5-7B-Instruct-Turbo",
        QWEN1_5_110B_CHAT => "Qwen/Qwen1.5-110B-Chat",
        QWEN1_5_72B_CHAT => "Qwen/Qwen1.5-72B-Chat",
        QWEN_2_1_5B_INSTRUCT => "Qwen/Qwen2-1.5B-Instruct",
        QWEN_2_7B_INSTRUCT => "Qwen/Qwen2-7B-Instruct",
        QWEN1_5_14B_CHAT => "Qwen/Qwen1.5-14B-Chat",
        QWEN1_5_1_8B_CHAT => "Qwen/Qwen1.5-1.8B-Chat",
        QWEN1_5_32B_CHAT => "Qwen/Qwen1.5-32B-Chat",
        QWEN1_5_7B_CHAT => "Qwen/Qwen1.5-7B-Chat",
        QWEN1_5_0_5B_CHAT => "Qwen/Qwen1.5-0.5B-Chat",
        QWEN1_5_4B_CHAT => "Qwen/Qwen1.5-4B-Chat",
        SNORKEL_MISTRAL_PAIRRM_DPO => "snorkelai/Snorkel-Mistral-PairRM-DPO",
        SNOWFLAKE_ARCTIC_INSTRUCT => "Snowflake/snowflake-arctic-instruct",
        ALPACA_7B => "togethercomputer/alpaca-7b",
        OPENHERMES_2_MISTRAL_7B => "teknium/OpenHermes-2-Mistral-7B",
        OPENHERMES_2_5_MISTRAL_7B => "teknium/OpenHermes-2p5-Mistral-7B",
        GUANACO_65B => "togethercomputer/guanaco-65b",
        GUANACO_13B => "togethercomputer/guanaco-13b",
        GUANACO_33B => "togethercomputer/guanaco-33b",
        GUANACO_7B => "togethercomputer/guanaco-7b",
        REMM_SLERP_L2_13B => "Undi95/ReMM-SLERP-L2-13B",
        TOPPY_M_7B => "Undi95/Toppy-M-7B",
        SOLAR_10_7B_INSTRUCT_V1 => "upstage/SOLAR-10.7B-Instruct-v1.0",
        SOLAR_10_7B_INSTRUCT_V1_INT4 => "togethercomputer/SOLAR-10.7B-Instruct-v1.0-int4",
        WIZARDLM_13B_V1_2 => "WizardLM/WizardLM-13B-V1.2",
    }
}
pub use CompletionModels::*;
// =================================================================
// Rig Implementation Types
// =================================================================

#[derive(Clone)]
pub struct CompletionModel<T = reqwest::Client> {
    pub(crate) client: Client<T>,
    pub model: String,
}

impl<T> CompletionModel<T> {
    pub fn new(client: Client<T>, model: CompletionModels) -> Self {
        Self {
            client,
            model: model.to_string(),
        }
    }

    pub fn with_model(client: Client<T>, model: &str) -> Self {
        Self {
            client,
            model: model.into(),
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

        let tool_choice = completion_request
            .tool_choice
            .map(ToolChoice::try_from)
            .transpose()?;

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
                "tool_choice": tool_choice,
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

impl<T> completion::CompletionModel for CompletionModel<T>
where
    T: HttpClientExt + Clone + Default + std::fmt::Debug + Send + 'static,
{
    type Response = openai::CompletionResponse;
    type StreamingResponse = openai::StreamingCompletionResponse;

    type Client = Client<T>;
    type Models = CompletionModels;

    fn make(client: &Self::Client, model: impl Into<Self::Models>) -> Self {
        Self::new(client.clone(), model.into())
    }

    fn make_custom(client: &Self::Client, model: &str) -> Self {
        Self::with_model(client.clone(), model)
    }

    #[cfg_attr(feature = "worker", worker::send)]
    async fn completion(
        &self,
        completion_request: completion::CompletionRequest,
    ) -> Result<completion::CompletionResponse<openai::CompletionResponse>, CompletionError> {
        let preamble = completion_request.preamble.clone();
        let request = self.create_completion_request(completion_request)?;
        let messages_as_json_string =
            serde_json::to_string(request.get("messages").unwrap()).unwrap();

        let span = if tracing::Span::current().is_disabled() {
            info_span!(
                target: "rig::completions",
                "chat",
                gen_ai.operation.name = "chat",
                gen_ai.provider.name = "together",
                gen_ai.request.model = self.model,
                gen_ai.system_instructions = preamble,
                gen_ai.response.id = tracing::field::Empty,
                gen_ai.response.model = tracing::field::Empty,
                gen_ai.usage.output_tokens = tracing::field::Empty,
                gen_ai.usage.input_tokens = tracing::field::Empty,
                gen_ai.input.messages = &messages_as_json_string,
                gen_ai.output.messages = tracing::field::Empty,
            )
        } else {
            tracing::Span::current()
        };

        tracing::debug!(target: "rig::completions", "TogetherAI completion request: {messages_as_json_string}");

        let body = serde_json::to_vec(&request)?;

        let req = self
            .client
            .post("/v1/chat/completions")?
            .header("Content-Type", "application/json")
            .body(body)
            .map_err(|x| CompletionError::HttpError(x.into()))?;

        async move {
            let response = self.client.http_client().send::<_, Bytes>(req).await?;
            let status = response.status();
            let response_body = response.into_body().into_future().await?.to_vec();

            if status.is_success() {
                match serde_json::from_slice::<ApiResponse<openai::CompletionResponse>>(
                    &response_body,
                )? {
                    ApiResponse::Ok(response) => {
                        let span = tracing::Span::current();
                        span.record(
                            "gen_ai.output.messages",
                            serde_json::to_string(&response.choices).unwrap(),
                        );
                        span.record("gen_ai.response.id", &response.id);
                        span.record("gen_ai.response.model_name", &response.model);
                        if let Some(ref usage) = response.usage {
                            span.record("gen_ai.usage.input_tokens", usage.prompt_tokens);
                            span.record(
                                "gen_ai.usage.output_tokens",
                                usage.total_tokens - usage.prompt_tokens,
                            );
                        }
                        tracing::trace!(
                            target: "rig::completions",
                            "TogetherAI completion response: {}",
                            serde_json::to_string_pretty(&response)?
                        );
                        response.try_into()
                    }
                    ApiResponse::Error(err) => Err(CompletionError::ProviderError(err.error)),
                }
            } else {
                Err(CompletionError::ProviderError(
                    String::from_utf8_lossy(&response_body).to_string(),
                ))
            }
        }
        .instrument(span)
        .await
    }

    #[cfg_attr(feature = "worker", worker::send)]
    async fn stream(
        &self,
        request: CompletionRequest,
    ) -> Result<StreamingCompletionResponse<Self::StreamingResponse>, CompletionError> {
        CompletionModel::stream(self, request).await
    }
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(untagged, rename_all = "snake_case")]
pub enum ToolChoice {
    None,
    Auto,
    Function(Vec<ToolChoiceFunctionKind>),
}

impl TryFrom<crate::message::ToolChoice> for ToolChoice {
    type Error = CompletionError;

    fn try_from(value: crate::message::ToolChoice) -> Result<Self, Self::Error> {
        let res = match value {
            crate::message::ToolChoice::None => Self::None,
            crate::message::ToolChoice::Auto => Self::Auto,
            crate::message::ToolChoice::Specific { function_names } => {
                let vec: Vec<ToolChoiceFunctionKind> = function_names
                    .into_iter()
                    .map(|name| ToolChoiceFunctionKind::Function { name })
                    .collect();

                Self::Function(vec)
            }
            choice => {
                return Err(CompletionError::ProviderError(format!(
                    "Unsupported tool choice type: {choice:?}"
                )));
            }
        };

        Ok(res)
    }
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type", content = "function")]
pub enum ToolChoiceFunctionKind {
    Function { name: String },
}
