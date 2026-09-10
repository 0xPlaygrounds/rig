//! Hyperbolic Inference API client and Rig integration
//!
//! # Example
//! ```ignore
//! use rig_core::{client::CompletionClient, providers::hyperbolic};
//!
//! # fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let client = hyperbolic::Client::new("YOUR_API_KEY")?;
//!
//! let llama_3_1_8b = client.completion_model(hyperbolic::LLAMA_3_1_8B);
//! # Ok(())
//! # }
//! ```

use crate::client::BearerAuth;
#[cfg(feature = "audio")]
use crate::client::HasAudioGeneration;
#[cfg(feature = "image")]
use crate::client::HasImageGeneration;
use crate::client::{self, HasCompletion, ModelTransport, Provider, ProviderClientResult};
use crate::http_client::{self, HttpClientExt};

// ================================================================
// Main Hyperbolic Client
// ================================================================
const HYPERBOLIC_API_BASE_URL: &str = "https://api.hyperbolic.xyz";

#[derive(Debug, Default, Clone, Copy)]
pub struct Hyperbolic;
type HyperbolicApiKey = BearerAuth;

impl Provider for Hyperbolic {
    const NAME: &'static str = "hyperbolic";
    const BASE_URL: &'static str = HYPERBOLIC_API_BASE_URL;
    const VERIFY_PATH: &'static str = "/models";
    type ApiKey = HyperbolicApiKey;
    type Config = ();
    type EnvInput = HyperbolicApiKey;

    fn build(_: (), _: &HyperbolicApiKey) -> http_client::Result<Self> {
        Ok(Hyperbolic)
    }

    fn from_env<H: HttpClientExt>(http: H) -> ProviderClientResult<Client<H>> {
        Client::from_env_api_key("HYPERBOLIC_API_KEY", None, http)
    }

    fn from_val<H: HttpClientExt>(
        input: HyperbolicApiKey,
        http: H,
    ) -> ProviderClientResult<Client<H>> {
        Client::new_with(input, http)
    }
}

impl HasCompletion for Hyperbolic {
    type Model<H>
        = CompletionModel<H>
    where
        H: ModelTransport;

    fn completion_model<H: ModelTransport>(client: &Client<H>, model: String) -> Self::Model<H> {
        CompletionModel::new(client.clone(), model)
    }
}

#[cfg(feature = "image")]
impl HasImageGeneration for Hyperbolic {
    type Model<H>
        = ImageGenerationModel<H>
    where
        H: ModelTransport;

    fn image_generation_model<H: ModelTransport>(
        client: &Client<H>,
        model: String,
    ) -> Self::Model<H> {
        ImageGenerationModel::new(client.clone(), model)
    }
}

#[cfg(feature = "audio")]
impl HasAudioGeneration for Hyperbolic {
    type Model<H>
        = AudioGenerationModel<H>
    where
        H: ModelTransport;

    fn audio_generation_model<H: ModelTransport>(
        client: &Client<H>,
        model: String,
    ) -> Self::Model<H> {
        AudioGenerationModel::new(client.clone(), model)
    }
}

impl crate::providers::openai::completion::OpenAICompatibleProvider for Hyperbolic {
    const PROVIDER_NAME: &'static str = "hyperbolic";

    // Hyperbolic's structured-output support is unverified; keep the
    // pre-migration behavior of dropping `output_schema` with a warning.
    const SUPPORTS_RESPONSE_FORMAT: bool = false;

    // Hyperbolic does not support tool calling; `tools`/`tool_choice` are
    // dropped with a warning during request conversion.
    const SUPPORTS_TOOLS: bool = false;

    type StreamingUsage = crate::providers::openai::Usage;

    type Response = crate::providers::openai::CompletionResponse;

    fn finalize_request_body(
        &self,
        body: &mut serde_json::Value,
    ) -> Result<(), crate::completion::CompletionError> {
        // Strip tool-exchange remnants that shared chat histories may carry;
        // content-part arrays are kept as-is for Hyperbolic's vision models.
        if let Some(messages) = body
            .get_mut("messages")
            .and_then(serde_json::Value::as_array_mut)
        {
            crate::providers::openai::completion::sanitize_plain_text_history(
                messages, None, false, false,
            );
        }

        Ok(())
    }

    // The client base URL is the bare host; image/audio generation build
    // their own v1 paths.
    fn completion_path(&self, _model: &str) -> String {
        "/v1/chat/completions".to_string()
    }
}

pub type Client<H = crate::http_client::BoxedHttpClient> = client::Client<Hyperbolic, H>;
pub type ClientBuilder<H = crate::markers::Missing> = client::ClientBuilder<Hyperbolic, H>;

#[cfg(feature = "audio")]
use crate::providers::openai::client::ApiResponse;

// ================================================================
// Hyperbolic Completion API
// ================================================================

/// Meta Llama 3.1b Instruct model with 8B parameters.
pub const LLAMA_3_1_8B: &str = "meta-llama/Meta-Llama-3.1-8B-Instruct";
/// Meta Llama 3.3b Instruct model with 70B parameters.
pub const LLAMA_3_3_70B: &str = "meta-llama/Llama-3.3-70B-Instruct";
/// Meta Llama 3.1b Instruct model with 70B parameters.
pub const LLAMA_3_1_70B: &str = "meta-llama/Meta-Llama-3.1-70B-Instruct";
/// Meta Llama 3 Instruct model with 70B parameters.
pub const LLAMA_3_70B: &str = "meta-llama/Meta-Llama-3-70B-Instruct";
/// Hermes 3 Instruct model with 70B parameters.
pub const HERMES_3_70B: &str = "NousResearch/Hermes-3-Llama-3.1-70b";
/// Deepseek v2.5 model.
pub const DEEPSEEK_2_5: &str = "deepseek-ai/DeepSeek-V2.5";
/// Qwen 2.5 model with 72B parameters.
pub const QWEN_2_5_72B: &str = "Qwen/Qwen2.5-72B-Instruct";
/// Meta Llama 3.2b Instruct model with 3B parameters.
pub const LLAMA_3_2_3B: &str = "meta-llama/Llama-3.2-3B-Instruct";
/// Qwen 2.5 Coder Instruct model with 32B parameters.
pub const QWEN_2_5_CODER_32B: &str = "Qwen/Qwen2.5-Coder-32B-Instruct";
/// Preview (latest) version of Qwen model with 32B parameters.
pub const QWEN_QWQ_PREVIEW_32B: &str = "Qwen/QwQ-32B-Preview";
/// Deepseek R1 Zero model.
pub const DEEPSEEK_R1_ZERO: &str = "deepseek-ai/DeepSeek-R1-Zero";
/// Deepseek R1 model.
pub const DEEPSEEK_R1: &str = "deepseek-ai/DeepSeek-R1";

/// Hyperbolic completion model, driven by the shared OpenAI Chat Completions path.
pub type CompletionModel<H = crate::http_client::BoxedHttpClient> =
    crate::providers::openai::completion::GenericCompletionModel<Hyperbolic, H>;

/// Raw completion payload, shared with the OpenAI Chat Completions path.
pub type CompletionResponse = crate::providers::openai::CompletionResponse;

// =======================================
// Hyperbolic Image Generation API
// =======================================

#[cfg(feature = "image")]
pub use image_generation::*;

#[cfg(feature = "image")]
#[cfg_attr(docsrs, doc(cfg(feature = "image")))]
mod image_generation {
    use super::Hyperbolic;
    use crate::image_generation;
    use crate::image_generation::{
        ImageGenerationError, ImageGenerationRequest, NormalizeImageGenerationResponse,
    };
    use crate::json_utils::merge_inplace;
    use crate::providers::internal::image_generation::{
        GenericImageGenerationModel, JsonImageGenerationProvider, decode_base64_image,
    };
    use serde::{Deserialize, Serialize};
    use serde_json::json;

    pub const SDXL1_0_BASE: &str = "SDXL1.0-base";
    pub const SD2: &str = "SD2";
    pub const SD1_5: &str = "SD1.5";
    pub const SSD: &str = "SSD";
    pub const SDXL_TURBO: &str = "SDXL-turbo";
    pub const SDXL_CONTROLNET: &str = "SDXL-ControlNet";
    pub const SD1_5_CONTROLNET: &str = "SD1.5-ControlNet";

    /// Hyperbolic image generation model.
    pub type ImageGenerationModel<T = crate::http_client::BoxedHttpClient> =
        GenericImageGenerationModel<Hyperbolic, T>;

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct Image {
        pub image: String,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct ImageGenerationResponse {
        pub images: Vec<Image>,
    }

    impl NormalizeImageGenerationResponse for ImageGenerationResponse {
        fn normalize(
            self,
            provider: &str,
        ) -> Result<image_generation::ImageGenerationResponse, ImageGenerationError> {
            let image = decode_base64_image(
                &self,
                |response| response.images.first().map(|image| image.image.as_str()),
                "missing image data",
                None,
            )?;
            Ok(image_generation::ImageGenerationResponse::new(
                image, provider,
            ))
        }
    }

    impl JsonImageGenerationProvider for Hyperbolic {
        const IMAGE_GENERATION_PATH: &'static str = "/v1/image/generation";
        const PROVIDER_NAME: &'static str = "hyperbolic";
        type Response = ImageGenerationResponse;

        fn image_generation_request_body(
            model: &str,
            generation_request: ImageGenerationRequest,
        ) -> Result<serde_json::Value, ImageGenerationError> {
            let mut request = json!({
                "model_name": model,
                "prompt": generation_request.prompt,
                "height": generation_request.height,
                "width": generation_request.width,
            });

            if let Some(params) = generation_request.additional_params {
                merge_inplace(&mut request, params);
            }

            Ok(request)
        }
    }
}

// ======================================
// Hyperbolic Audio Generation API
// ======================================
#[cfg(feature = "audio")]
pub use audio_generation::*;

#[cfg(feature = "audio")]
#[cfg_attr(docsrs, doc(cfg(feature = "image")))]
mod audio_generation {
    use super::{ApiResponse, Client};
    use crate::audio_generation;
    use crate::audio_generation::{
        AudioGenerationError, AudioGenerationRequest, NormalizeAudioGenerationResponse,
    };
    use crate::http_client::{self, HttpClientExt};
    use base64::Engine;
    use base64::prelude::BASE64_STANDARD;
    use bytes::Bytes;
    use serde::{Deserialize, Serialize};
    use serde_json::json;

    #[derive(Clone)]
    pub struct AudioGenerationModel<T = crate::http_client::BoxedHttpClient> {
        client: Client<T>,
        pub language: String,
    }

    impl<T> AudioGenerationModel<T> {
        /// Hyperbolic addresses its TTS endpoint by language rather than by
        /// model, so the "model" identifier a client is asked for is the language.
        pub fn new(client: Client<T>, language: impl Into<String>) -> Self {
            Self {
                client,
                language: language.into(),
            }
        }
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct AudioGenerationResponse {
        pub audio: String,
    }

    impl NormalizeAudioGenerationResponse for AudioGenerationResponse {
        fn normalize(
            self,
            provider: &str,
        ) -> Result<audio_generation::AudioGenerationResponse, AudioGenerationError> {
            let data = BASE64_STANDARD
                .decode(&self.audio)
                .map_err(|err| AudioGenerationError::ResponseError(err.to_string()))?;

            Ok(audio_generation::AudioGenerationResponse::new(
                data, provider,
            ))
        }
    }

    impl<T> AudioGenerationModel<T>
    where
        T: HttpClientExt + Clone + crate::wasm_compat::WasmCompatSend + 'static,
    {
        /// Perform the generation and return Hyperbolic's native response
        /// (base64 audio) instead of the normalized
        /// [`audio_generation::AudioGenerationResponse`]. Same request,
        /// transport, parser, and error path as
        /// [`audio_generation::AudioGenerationModel::audio_generation`].
        pub async fn raw_audio_generation(
            &self,
            request: AudioGenerationRequest,
        ) -> Result<AudioGenerationResponse, AudioGenerationError> {
            let request = json!({
                "language": self.language,
                "speaker": request.voice,
                "text": request.text,
                "speed": request.speed
            });

            let body = serde_json::to_vec(&request)?;

            let req = self
                .client
                .post("/v1/audio/generation")?
                .body(body)
                .map_err(http_client::Error::from)?;

            let response = self.client.send::<_, Bytes>(req).await?;
            let status = response.status();
            let response_body = response.into_body().into_future().await?.to_vec();

            if !status.is_success() {
                return Err(AudioGenerationError::from_http_response(
                    status,
                    String::from_utf8_lossy(&response_body),
                ));
            }

            match serde_json::from_slice::<ApiResponse<AudioGenerationResponse>>(&response_body)? {
                ApiResponse::Ok(response) => Ok(response),
                ApiResponse::Err(err) => {
                    tracing::warn!(message = %err.message, "provider returned an error response");
                    Err(AudioGenerationError::from_http_response(
                        status,
                        String::from_utf8_lossy(&response_body),
                    ))
                }
            }
        }
    }

    impl<T> audio_generation::AudioGenerationModel for AudioGenerationModel<T>
    where
        T: HttpClientExt + Clone + crate::wasm_compat::WasmCompatSend + 'static,
    {
        async fn audio_generation(
            &self,
            request: AudioGenerationRequest,
        ) -> Result<audio_generation::AudioGenerationResponse, AudioGenerationError> {
            crate::telemetry::instrument_modality(
                "hyperbolic",
                &self.language,
                crate::telemetry::ModalityOperation::AudioGeneration,
                async {
                    let response = self.raw_audio_generation(request).await?;
                    let captured = serde_json::to_value(&response)?;
                    Ok(response.normalize("hyperbolic")?.with_raw(captured))
                },
            )
            .await
        }
    }
}

#[cfg(test)]
mod tests;
