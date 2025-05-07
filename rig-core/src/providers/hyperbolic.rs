//! Hyperbolic Inference API client and Rig integration
//!
//! # Example
//! ```
//! use rig::providers::hyperbolic;
//!
//! let client = hyperbolic::Client::new("YOUR_API_KEY");
//!
//! let llama_3_1_8b = client.completion_model(hyperbolic::LLAMA_3_1_8B);
//! ```

use super::openai::{send_compatible_streaming_request, AssistantContent};

use crate::json_utils::merge_inplace;
use crate::message;
use crate::streaming::{StreamingCompletionModel, StreamingCompletionResponse};
use crate::{
    agent::AgentBuilder,
    completion::{self, CompletionError, CompletionRequest},
    extractor::ExtractorBuilder,
    json_utils,
    providers::openai::Message,
    OneOrMany,
};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

// ================================================================
// Main Hyperbolic Client
// ================================================================
const HYPERBOLIC_API_BASE_URL: &str = "https://api.hyperbolic.xyz/v1";

#[derive(Clone)]
pub struct Client {
    base_url: String,
    http_client: reqwest::Client,
}

impl Client {
    /// Create a new Hyperbolic client with the given API key.
    pub fn new(api_key: &str) -> Self {
        Self::from_url(api_key, HYPERBOLIC_API_BASE_URL)
    }

    /// Create a new OpenAI client with the given API key and base API URL.
    pub fn from_url(api_key: &str, base_url: &str) -> Self {
        Self {
            base_url: base_url.to_string(),
            http_client: reqwest::Client::builder()
                .default_headers({
                    let mut headers = reqwest::header::HeaderMap::new();
                    headers.insert(
                        "Authorization",
                        format!("Bearer {}", api_key)
                            .parse()
                            .expect("Bearer token should parse"),
                    );
                    headers
                })
                .build()
                .expect("OpenAI reqwest client should build"),
        }
    }

    /// Create a new Hyperbolic client from the `HYPERBOLIC_API_KEY` environment variable.
    /// Panics if the environment variable is not set.
    pub fn from_env() -> Self {
        let api_key = std::env::var("HYPERBOLIC_API_KEY").expect("HYPERBOLIC_API_KEY not set");
        Self::new(&api_key)
    }

    fn post(&self, path: &str) -> reqwest::RequestBuilder {
        let url = format!("{}/{}", self.base_url, path).replace("//", "/");
        self.http_client.post(url)
    }

    /// Create a completion model with the given name.
    ///
    /// # Example
    /// ```
    /// use rig::providers::hyperbolic::{Client, self};
    ///
    /// // Initialize the Hyperbolic client
    /// let hyperbolic = Client::new("your-hyperbolic-api-key");
    ///
    /// let llama_3_1_8b = hyperbolic.completion_model(hyperbolic::LLAMA_3_1_8B);
    /// ```
    pub fn completion_model(&self, model: &str) -> CompletionModel {
        CompletionModel::new(self.clone(), model)
    }

    /// Create an image generation model with the given name.
    ///
    /// # Example
    /// ```
    /// use rig::providers::hyperbolic::{Client, self};
    ///
    /// // Initialize the Hyperbolic client
    /// let hyperbolic = Client::new("your-hyperbolic-api-key");
    ///
    /// let llama_3_1_8b = hyperbolic.image_generation_model(hyperbolic::SSD);
    /// ```
    #[cfg(feature = "image")]
    pub fn image_generation_model(&self, model: &str) -> ImageGenerationModel {
        ImageGenerationModel::new(self.clone(), model)
    }

    /// Create a completion model with the given name.
    ///
    /// # Example
    /// ```
    /// use rig::providers::hyperbolic::{Client, self};
    ///
    /// // Initialize the Hyperbolic client
    /// let hyperbolic = Client::new("your-hyperbolic-api-key");
    ///
    /// let tts = hyperbolic.audio_generation_model("EN");
    /// ```
    #[cfg(feature = "audio")]
    pub fn audio_generation_model(&self, language: &str) -> AudioGenerationModel {
        AudioGenerationModel::new(self.clone(), language)
    }

    /// Create an agent builder with the given completion model.
    ///
    /// # Example
    /// ```
    /// use rig::providers::hyperbolic::{Client, self};
    ///
    /// // Initialize the Eternal client
    /// let hyperbolic = Client::new("your-hyperbolic-api-key");
    ///
    /// let agent = hyperbolic.agent(hyperbolic::LLAMA_3_1_8B)
    ///    .preamble("You are comedian AI with a mission to make people laugh.")
    ///    .temperature(0.0)
    ///    .build();
    /// ```
    pub fn agent(&self, model: &str) -> AgentBuilder<CompletionModel> {
        AgentBuilder::new(self.completion_model(model))
    }

    /// Create an extractor builder with the given completion model.
    pub fn extractor<T: JsonSchema + for<'a> Deserialize<'a> + Serialize + Send + Sync>(
        &self,
        model: &str,
    ) -> ExtractorBuilder<T, CompletionModel> {
        ExtractorBuilder::new(self.completion_model(model))
    }
}

#[derive(Debug, Deserialize)]
struct ApiErrorResponse {
    message: String,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum ApiResponse<T> {
    Ok(T),
    Err(ApiErrorResponse),
}

#[derive(Debug, Deserialize)]
pub struct EmbeddingData {
    pub object: String,
    pub embedding: Vec<f64>,
    pub index: usize,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub total_tokens: usize,
}

impl std::fmt::Display for Usage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Prompt tokens: {} Total tokens: {}",
            self.prompt_tokens, self.total_tokens
        )
    }
}

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

/// A Hyperbolic completion object.
///
/// For more information, see this link: <https://docs.hyperbolic.xyz/reference/create_chat_completion_v1_chat_completions_post>
#[derive(Debug, Deserialize)]
pub struct CompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<Choice>,
    pub usage: Option<Usage>,
}

impl From<ApiErrorResponse> for CompletionError {
    fn from(err: ApiErrorResponse) -> Self {
        CompletionError::ProviderError(err.message)
    }
}

impl TryFrom<CompletionResponse> for completion::CompletionResponse<CompletionResponse> {
    type Error = CompletionError;

    fn try_from(response: CompletionResponse) -> Result<Self, Self::Error> {
        let choice = response.choices.first().ok_or_else(|| {
            CompletionError::ResponseError("Response contained no choices".to_owned())
        })?;

        let content = match &choice.message {
            Message::Assistant {
                content,
                tool_calls,
                ..
            } => {
                let mut content = content
                    .iter()
                    .map(|c| match c {
                        AssistantContent::Text { text } => completion::AssistantContent::text(text),
                        AssistantContent::Refusal { refusal } => {
                            completion::AssistantContent::text(refusal)
                        }
                    })
                    .collect::<Vec<_>>();

                content.extend(
                    tool_calls
                        .iter()
                        .map(|call| {
                            completion::AssistantContent::tool_call(
                                &call.id,
                                &call.function.name,
                                call.function.arguments.clone(),
                            )
                        })
                        .collect::<Vec<_>>(),
                );
                Ok(content)
            }
            _ => Err(CompletionError::ResponseError(
                "Response did not contain a valid message or tool call".into(),
            )),
        }?;

        let choice = OneOrMany::many(content).map_err(|_| {
            CompletionError::ResponseError(
                "Response contained no message or tool call (empty)".to_owned(),
            )
        })?;

        Ok(completion::CompletionResponse {
            choice,
            raw_response: response,
        })
    }
}

#[derive(Debug, Deserialize)]
pub struct Choice {
    pub index: usize,
    pub message: Message,
    pub finish_reason: String,
}

#[derive(Clone)]
pub struct CompletionModel {
    client: Client,
    /// Name of the model (e.g.: deepseek-ai/DeepSeek-R1)
    pub model: String,
}

impl CompletionModel {
    pub(crate) fn create_completion_request(
        &self,
        completion_request: CompletionRequest,
    ) -> Result<Value, CompletionError> {
        // Build up the order of messages (context, chat_history, prompt)
        let mut partial_history = vec![];
        if let Some(docs) = completion_request.normalized_documents() {
            partial_history.push(docs);
        }
        partial_history.extend(completion_request.chat_history);

        // Initialize full history with preamble (or empty if non-existent)
        let mut full_history: Vec<Message> = completion_request
            .preamble
            .map_or_else(Vec::new, |preamble| vec![Message::system(&preamble)]);

        // Convert and extend the rest of the history
        full_history.extend(
            partial_history
                .into_iter()
                .map(message::Message::try_into)
                .collect::<Result<Vec<Vec<Message>>, _>>()?
                .into_iter()
                .flatten()
                .collect::<Vec<_>>(),
        );

        let request = json!({
            "model": self.model,
            "messages": full_history,
            "temperature": completion_request.temperature,
        });

        let request = if let Some(params) = completion_request.additional_params {
            json_utils::merge(request, params)
        } else {
            request
        };

        Ok(request)
    }
}

impl CompletionModel {
    pub fn new(client: Client, model: &str) -> Self {
        Self {
            client,
            model: model.to_string(),
        }
    }
}

impl completion::CompletionModel for CompletionModel {
    type Response = CompletionResponse;

    #[cfg_attr(feature = "worker", worker::send)]
    async fn completion(
        &self,
        completion_request: CompletionRequest,
    ) -> Result<completion::CompletionResponse<CompletionResponse>, CompletionError> {
        let request = self.create_completion_request(completion_request)?;

        let response = self
            .client
            .post("/chat/completions")
            .json(&request)
            .send()
            .await?;

        if response.status().is_success() {
            match response.json::<ApiResponse<CompletionResponse>>().await? {
                ApiResponse::Ok(response) => {
                    tracing::info!(target: "rig",
                        "Hyperbolic completion token usage: {:?}",
                        response.usage.clone().map(|usage| format!("{usage}")).unwrap_or("N/A".to_string())
                    );

                    response.try_into()
                }
                ApiResponse::Err(err) => Err(CompletionError::ProviderError(err.message)),
            }
        } else {
            Err(CompletionError::ProviderError(response.text().await?))
        }
    }
}

impl StreamingCompletionModel for CompletionModel {
    type StreamingResponse = openai::StreamingCompletionResponse;
    async fn stream(
        &self,
        completion_request: CompletionRequest,
    ) -> Result<StreamingCompletionResponse<Self::StreamingResponse>, CompletionError> {
        let mut request = self.create_completion_request(completion_request)?;

        merge_inplace(
            &mut request,
            json!({"stream": true, "stream_options": {"include_usage": true}}),
        );

        let builder = self.client.post("/chat/completions").json(&request);

        send_compatible_streaming_request(builder).await
    }
}

// =======================================
// Hyperbolic Image Generation API
// =======================================

#[cfg(feature = "image")]
pub use image_generation::*;

#[cfg(feature = "image")]
mod image_generation {
    use super::{ApiResponse, Client};
    use crate::image_generation;
    use crate::image_generation::{ImageGenerationError, ImageGenerationRequest};
    use crate::json_utils::merge_inplace;
    use base64::prelude::BASE64_STANDARD;
    use base64::Engine;
    use serde::Deserialize;
    use serde_json::json;

    pub const SDXL1_0_BASE: &str = "SDXL1.0-base";
    pub const SD2: &str = "SD2";
    pub const SD1_5: &str = "SD1.5";
    pub const SSD: &str = "SSD";
    pub const SDXL_TURBO: &str = "SDXL-turbo";
    pub const SDXL_CONTROLNET: &str = "SDXL-ControlNet";
    pub const SD1_5_CONTROLNET: &str = "SD1.5-ControlNet";

    #[cfg(feature = "image")]
    #[derive(Clone)]
    pub struct ImageGenerationModel {
        client: Client,
        pub model: String,
    }

    #[cfg(feature = "image")]
    impl ImageGenerationModel {
        pub(crate) fn new(client: Client, model: &str) -> ImageGenerationModel {
            Self {
                client,
                model: model.to_string(),
            }
        }
    }

    #[cfg(feature = "image")]
    #[derive(Clone, Deserialize)]
    pub struct Image {
        image: String,
    }

    #[cfg(feature = "image")]
    #[derive(Clone, Deserialize)]
    pub struct ImageGenerationResponse {
        images: Vec<Image>,
    }

    #[cfg(feature = "image")]
    impl TryFrom<ImageGenerationResponse>
        for image_generation::ImageGenerationResponse<ImageGenerationResponse>
    {
        type Error = ImageGenerationError;

        fn try_from(value: ImageGenerationResponse) -> Result<Self, Self::Error> {
            let data = BASE64_STANDARD
                .decode(&value.images[0].image)
                .expect("Could not decode image.");

            Ok(Self {
                image: data,
                response: value,
            })
        }
    }

    #[cfg(feature = "image")]
    impl image_generation::ImageGenerationModel for ImageGenerationModel {
        type Response = ImageGenerationResponse;

        async fn image_generation(
            &self,
            generation_request: ImageGenerationRequest,
        ) -> Result<image_generation::ImageGenerationResponse<Self::Response>, ImageGenerationError>
        {
            let mut request = json!({
                "model_name": self.model,
                "prompt": generation_request.prompt,
                "height": generation_request.height,
                "width": generation_request.width,
            });

            if let Some(params) = generation_request.additional_params {
                merge_inplace(&mut request, params);
            }

            let response = self
                .client
                .post("/image/generation")
                .json(&request)
                .send()
                .await?;

            if !response.status().is_success() {
                return Err(ImageGenerationError::ProviderError(format!(
                    "{}: {}",
                    response.status().as_str(),
                    response.text().await?
                )));
            }

            match response
                .json::<ApiResponse<ImageGenerationResponse>>()
                .await?
            {
                ApiResponse::Ok(response) => response.try_into(),
                ApiResponse::Err(err) => Err(ImageGenerationError::ResponseError(err.message)),
            }
        }
    }
}

// ======================================
// Hyperbolic Audio Generation API
// ======================================
use crate::providers::openai;
#[cfg(feature = "audio")]
pub use audio_generation::*;

#[cfg(feature = "audio")]
mod audio_generation {
    use super::{ApiResponse, Client};
    use crate::audio_generation;
    use crate::audio_generation::{AudioGenerationError, AudioGenerationRequest};
    use base64::prelude::BASE64_STANDARD;
    use base64::Engine;
    use serde::Deserialize;
    use serde_json::json;

    #[derive(Clone)]
    pub struct AudioGenerationModel {
        client: Client,
        pub langauge: String,
    }

    impl AudioGenerationModel {
        pub(crate) fn new(client: Client, language: &str) -> AudioGenerationModel {
            Self {
                client,
                langauge: language.to_string(),
            }
        }
    }

    #[derive(Clone, Deserialize)]
    pub struct AudioGenerationResponse {
        audio: String,
    }

    impl TryFrom<AudioGenerationResponse>
        for audio_generation::AudioGenerationResponse<AudioGenerationResponse>
    {
        type Error = AudioGenerationError;

        fn try_from(value: AudioGenerationResponse) -> Result<Self, Self::Error> {
            let data = BASE64_STANDARD
                .decode(&value.audio)
                .expect("Could not decode audio.");

            Ok(Self {
                audio: data,
                response: value,
            })
        }
    }

    impl audio_generation::AudioGenerationModel for AudioGenerationModel {
        type Response = AudioGenerationResponse;

        async fn audio_generation(
            &self,
            request: AudioGenerationRequest,
        ) -> Result<audio_generation::AudioGenerationResponse<Self::Response>, AudioGenerationError>
        {
            let request = json!({
                "language": self.langauge,
                "speaker": request.voice,
                "text": request.text,
                "speed": request.speed
            });

            let response = self
                .client
                .post("/audio/generation")
                .json(&request)
                .send()
                .await?;

            if !response.status().is_success() {
                return Err(AudioGenerationError::ProviderError(format!(
                    "{}: {}",
                    response.status(),
                    response.text().await?
                )));
            }

            match serde_json::from_str::<ApiResponse<AudioGenerationResponse>>(
                &response.text().await?,
            )? {
                ApiResponse::Ok(response) => response.try_into(),
                ApiResponse::Err(err) => Err(AudioGenerationError::ProviderError(err.message)),
            }
        }
    }
}
