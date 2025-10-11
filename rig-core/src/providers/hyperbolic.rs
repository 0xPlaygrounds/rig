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
use super::openai::{AssistantContent, send_compatible_streaming_request};

use crate::client::{CompletionClient, ProviderClient, VerifyClient, VerifyError};
use crate::http_client::{self, HttpClientExt};
use crate::json_utils::merge_inplace;
use crate::message;
use crate::streaming::StreamingCompletionResponse;

use crate::impl_conversion_traits;
use crate::providers::openai;
use crate::{
    OneOrMany,
    completion::{self, CompletionError, CompletionRequest},
    json_utils,
    providers::openai::Message,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

// ================================================================
// Main Hyperbolic Client
// ================================================================
const HYPERBOLIC_API_BASE_URL: &str = "https://api.hyperbolic.xyz";

pub struct ClientBuilder<'a, T = reqwest::Client> {
    api_key: &'a str,
    base_url: &'a str,
    http_client: T,
}

impl<'a, T> ClientBuilder<'a, T>
where
    T: Default,
{
    pub fn new(api_key: &'a str) -> Self {
        Self {
            api_key,
            base_url: HYPERBOLIC_API_BASE_URL,
            http_client: Default::default(),
        }
    }
}

impl<'a, T> ClientBuilder<'a, T> {
    pub fn base_url(mut self, base_url: &'a str) -> Self {
        self.base_url = base_url;
        self
    }

    pub fn with_client<U>(self, http_client: U) -> ClientBuilder<'a, U> {
        ClientBuilder {
            api_key: self.api_key,
            base_url: self.base_url,
            http_client,
        }
    }

    pub fn build(self) -> Client<T> {
        Client {
            base_url: self.base_url.to_string(),
            api_key: self.api_key.to_string(),
            http_client: self.http_client,
        }
    }
}

#[derive(Clone)]
pub struct Client<T = reqwest::Client> {
    base_url: String,
    api_key: String,
    http_client: T,
}

impl<T> std::fmt::Debug for Client<T>
where
    T: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Client")
            .field("base_url", &self.base_url)
            .field("http_client", &self.http_client)
            .field("api_key", &"<REDACTED>")
            .finish()
    }
}

impl<T> Client<T>
where
    T: Default,
{
    /// Create a new Hyperbolic client builder.
    ///
    /// # Example
    /// ```
    /// use rig::providers::hyperbolic::{ClientBuilder, self};
    ///
    /// // Initialize the Hyperbolic client
    /// let hyperbolic = Client::builder("your-hyperbolic-api-key")
    ///    .build()
    /// ```
    pub fn builder(api_key: &str) -> ClientBuilder<'_, T> {
        ClientBuilder::new(api_key)
    }

    /// Create a new Hyperbolic client. For more control, use the `builder` method.
    ///
    /// # Panics
    /// - If the reqwest client cannot be built (if the TLS backend cannot be initialized).
    pub fn new(api_key: &str) -> Self {
        Self::builder(api_key).build()
    }
}

impl<T> Client<T>
where
    T: HttpClientExt,
{
    fn req(
        &self,
        method: http_client::Method,
        path: &str,
    ) -> http_client::Result<http_client::Builder> {
        let url = format!("{}/{}", self.base_url, path.trim_start_matches('/'));

        http_client::with_bearer_auth(
            http_client::Builder::new().method(method).uri(url),
            &self.api_key,
        )
    }

    fn get(&self, path: &str) -> http_client::Result<http_client::Builder> {
        self.req(http_client::Method::GET, path)
    }
}

impl Client<reqwest::Client> {
    fn reqwest_post(&self, path: &str) -> reqwest::RequestBuilder {
        let url = format!("{}/{}", self.base_url, path).replace("//", "/");

        self.http_client.post(url).bearer_auth(&self.api_key)
    }
}

impl ProviderClient for Client<reqwest::Client> {
    /// Create a new Hyperbolic client from the `HYPERBOLIC_API_KEY` environment variable.
    /// Panics if the environment variable is not set.
    fn from_env() -> Self {
        let api_key = std::env::var("HYPERBOLIC_API_KEY").expect("HYPERBOLIC_API_KEY not set");
        Self::new(&api_key)
    }

    fn from_val(input: crate::client::ProviderValue) -> Self {
        let crate::client::ProviderValue::Simple(api_key) = input else {
            panic!("Incorrect provider value type")
        };
        Self::new(&api_key)
    }
}

impl CompletionClient for Client<reqwest::Client> {
    type CompletionModel = CompletionModel<reqwest::Client>;

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
    fn completion_model(&self, model: &str) -> CompletionModel<reqwest::Client> {
        CompletionModel::new(self.clone(), model)
    }
}

impl VerifyClient for Client<reqwest::Client> {
    #[cfg_attr(feature = "worker", worker::send)]
    async fn verify(&self) -> Result<(), VerifyError> {
        let req = self
            .get("/models")?
            .body(http_client::NoBody)
            .map_err(http_client::Error::from)?;

        let response = HttpClientExt::send(&self.http_client, req).await?;

        match response.status() {
            reqwest::StatusCode::OK => Ok(()),
            reqwest::StatusCode::UNAUTHORIZED => Err(VerifyError::InvalidAuthentication),
            reqwest::StatusCode::INTERNAL_SERVER_ERROR
            | reqwest::StatusCode::SERVICE_UNAVAILABLE
            | reqwest::StatusCode::BAD_GATEWAY => {
                let text = http_client::text(response).await?;
                Err(VerifyError::ProviderError(text))
            }
            _ => {
                //response.error_for_status()?;
                Ok(())
            }
        }
    }
}

impl_conversion_traits!(
    AsEmbeddings,
    AsTranscription for Client<T>
);

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

#[derive(Clone, Debug, Deserialize, Serialize)]
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
#[derive(Debug, Deserialize, Serialize)]
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

        let usage = response
            .usage
            .as_ref()
            .map(|usage| completion::Usage {
                input_tokens: usage.prompt_tokens as u64,
                output_tokens: (usage.total_tokens - usage.prompt_tokens) as u64,
                total_tokens: usage.total_tokens as u64,
            })
            .unwrap_or_default();

        Ok(completion::CompletionResponse {
            choice,
            usage,
            raw_response: response,
        })
    }
}

#[derive(Debug, Deserialize, Serialize)]
pub struct Choice {
    pub index: usize,
    pub message: Message,
    pub finish_reason: String,
}

#[derive(Clone)]
pub struct CompletionModel<T> {
    client: Client<T>,
    /// Name of the model (e.g.: deepseek-ai/DeepSeek-R1)
    pub model: String,
}

impl<T> CompletionModel<T> {
    pub fn new(client: Client<T>, model: &str) -> Self {
        Self {
            client,
            model: model.to_string(),
        }
    }

    pub(crate) fn create_completion_request(
        &self,
        completion_request: CompletionRequest,
    ) -> Result<Value, CompletionError> {
        if completion_request.tool_choice.is_some() {
            tracing::warn!("WARNING: `tool_choice` not supported on Hyperbolic");
        }
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

impl completion::CompletionModel for CompletionModel<reqwest::Client> {
    type Response = CompletionResponse;
    type StreamingResponse = openai::StreamingCompletionResponse;

    #[cfg_attr(feature = "worker", worker::send)]
    async fn completion(
        &self,
        completion_request: CompletionRequest,
    ) -> Result<completion::CompletionResponse<CompletionResponse>, CompletionError> {
        let preamble = completion_request.preamble.clone();
        let request = self.create_completion_request(completion_request)?;

        let span = if tracing::Span::current().is_disabled() {
            info_span!(
                target: "rig::completions",
                "chat",
                gen_ai.operation.name = "chat",
                gen_ai.provider.name = "hyperbolic",
                gen_ai.request.model = self.model,
                gen_ai.system_instructions = preamble,
                gen_ai.response.id = tracing::field::Empty,
                gen_ai.response.model = tracing::field::Empty,
                gen_ai.usage.output_tokens = tracing::field::Empty,
                gen_ai.usage.input_tokens = tracing::field::Empty,
                gen_ai.input.messages = serde_json::to_string(&request.get("messages").unwrap()).unwrap(),
                gen_ai.output.messages = tracing::field::Empty,
            )
        } else {
            tracing::Span::current()
        };

        let async_block = async move {
            let response = self
                .client
                .reqwest_post("/v1/chat/completions")
                .json(&request)
                .send()
                .await
                .map_err(|e| http_client::Error::Instance(e.into()))?;

            if response.status().is_success() {
                match response
                    .json::<ApiResponse<CompletionResponse>>()
                    .await
                    .map_err(|e| http_client::Error::Instance(e.into()))?
                {
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
                Err(CompletionError::ProviderError(
                    response
                        .text()
                        .await
                        .map_err(|e| http_client::Error::Instance(e.into()))?,
                ))
            }
        };

        async_block.instrument(span).await
    }

    #[cfg_attr(feature = "worker", worker::send)]
    async fn stream(
        &self,
        completion_request: CompletionRequest,
    ) -> Result<StreamingCompletionResponse<Self::StreamingResponse>, CompletionError> {
        let preamble = completion_request.preamble.clone();
        let mut request = self.create_completion_request(completion_request)?;

        let span = if tracing::Span::current().is_disabled() {
            info_span!(
                target: "rig::completions",
                "chat_streaming",
                gen_ai.operation.name = "chat_streaming",
                gen_ai.provider.name = "hyperbolic",
                gen_ai.request.model = self.model,
                gen_ai.system_instructions = preamble,
                gen_ai.response.id = tracing::field::Empty,
                gen_ai.response.model = tracing::field::Empty,
                gen_ai.usage.output_tokens = tracing::field::Empty,
                gen_ai.usage.input_tokens = tracing::field::Empty,
                gen_ai.input.messages = serde_json::to_string(&request.get("messages").unwrap()).unwrap(),
                gen_ai.output.messages = tracing::field::Empty,
            )
        } else {
            tracing::Span::current()
        };

        merge_inplace(
            &mut request,
            json!({"stream": true, "stream_options": {"include_usage": true}}),
        );

        let builder = self
            .client
            .reqwest_post("/v1/chat/completions")
            .json(&request);

        send_compatible_streaming_request(builder)
            .instrument(span)
            .await
    }
}

// =======================================
// Hyperbolic Image Generation API
// =======================================

#[cfg(feature = "image")]
pub use image_generation::*;

#[cfg(feature = "image")]
#[cfg_attr(docsrs, doc(cfg(feature = "image")))]
mod image_generation {
    use super::{ApiResponse, Client};
    use crate::client::ImageGenerationClient;
    use crate::image_generation::{ImageGenerationError, ImageGenerationRequest};
    use crate::json_utils::merge_inplace;
    use crate::{http_client, image_generation};
    use base64::Engine;
    use base64::prelude::BASE64_STANDARD;
    use serde::Deserialize;
    use serde_json::json;

    pub const SDXL1_0_BASE: &str = "SDXL1.0-base";
    pub const SD2: &str = "SD2";
    pub const SD1_5: &str = "SD1.5";
    pub const SSD: &str = "SSD";
    pub const SDXL_TURBO: &str = "SDXL-turbo";
    pub const SDXL_CONTROLNET: &str = "SDXL-ControlNet";
    pub const SD1_5_CONTROLNET: &str = "SD1.5-ControlNet";

    #[derive(Clone)]
    pub struct ImageGenerationModel<T> {
        client: Client<T>,
        pub model: String,
    }

    impl<T> ImageGenerationModel<T> {
        pub(crate) fn new(client: Client<T>, model: &str) -> ImageGenerationModel<T> {
            Self {
                client,
                model: model.to_string(),
            }
        }
    }

    #[derive(Clone, Deserialize)]
    pub struct Image {
        image: String,
    }

    #[derive(Clone, Deserialize)]
    pub struct ImageGenerationResponse {
        images: Vec<Image>,
    }

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

    impl image_generation::ImageGenerationModel for ImageGenerationModel<reqwest::Client> {
        type Response = ImageGenerationResponse;

        #[cfg_attr(feature = "worker", worker::send)]
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
                .reqwest_post("/v1/image/generation")
                .json(&request)
                .send()
                .await
                .map_err(|e| {
                    ImageGenerationError::HttpError(http_client::Error::Instance(e.into()))
                })?;

            if !response.status().is_success() {
                return Err(ImageGenerationError::ProviderError(format!(
                    "{}: {}",
                    response.status().as_str(),
                    response.text().await.map_err(|e| {
                        ImageGenerationError::HttpError(http_client::Error::Instance(e.into()))
                    })?
                )));
            }

            match response
                .json::<ApiResponse<ImageGenerationResponse>>()
                .await
                .map_err(|e| {
                    ImageGenerationError::HttpError(http_client::Error::Instance(e.into()))
                })? {
                ApiResponse::Ok(response) => response.try_into(),
                ApiResponse::Err(err) => Err(ImageGenerationError::ResponseError(err.message)),
            }
        }
    }

    impl ImageGenerationClient for Client<reqwest::Client> {
        type ImageGenerationModel = ImageGenerationModel<reqwest::Client>;

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
        fn image_generation_model(&self, model: &str) -> ImageGenerationModel<reqwest::Client> {
            ImageGenerationModel::new(self.clone(), model)
        }
    }
}

// ======================================
// Hyperbolic Audio Generation API
// ======================================
#[cfg(feature = "audio")]
pub use audio_generation::*;
use tracing::{Instrument, info_span};

#[cfg(feature = "audio")]
#[cfg_attr(docsrs, doc(cfg(feature = "image")))]
mod audio_generation {
    use super::{ApiResponse, Client};
    use crate::audio_generation;
    use crate::audio_generation::{AudioGenerationError, AudioGenerationRequest};
    use crate::client::AudioGenerationClient;
    use crate::http_client;
    use base64::Engine;
    use base64::prelude::BASE64_STANDARD;
    use serde::Deserialize;
    use serde_json::json;

    #[derive(Clone)]
    pub struct AudioGenerationModel<T> {
        client: Client<T>,
        pub language: String,
    }

    impl<T> AudioGenerationModel<T> {
        pub(crate) fn new(client: Client<T>, language: &str) -> AudioGenerationModel<T> {
            Self {
                client,
                language: language.to_string(),
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

    impl audio_generation::AudioGenerationModel for AudioGenerationModel<reqwest::Client> {
        type Response = AudioGenerationResponse;

        #[cfg_attr(feature = "worker", worker::send)]
        async fn audio_generation(
            &self,
            request: AudioGenerationRequest,
        ) -> Result<audio_generation::AudioGenerationResponse<Self::Response>, AudioGenerationError>
        {
            let request = json!({
                "language": self.language,
                "speaker": request.voice,
                "text": request.text,
                "speed": request.speed
            });

            let response = self
                .client
                .reqwest_post("/v1/audio/generation")
                .json(&request)
                .send()
                .await
                .map_err(|e| {
                    AudioGenerationError::HttpError(http_client::Error::Instance(e.into()))
                })?;

            if !response.status().is_success() {
                return Err(AudioGenerationError::ProviderError(format!(
                    "{}: {}",
                    response.status(),
                    response.text().await.map_err(|e| {
                        AudioGenerationError::HttpError(http_client::Error::Instance(e.into()))
                    })?
                )));
            }

            match serde_json::from_str::<ApiResponse<AudioGenerationResponse>>(
                &response.text().await.map_err(|e| {
                    AudioGenerationError::HttpError(http_client::Error::Instance(e.into()))
                })?,
            )? {
                ApiResponse::Ok(response) => response.try_into(),
                ApiResponse::Err(err) => Err(AudioGenerationError::ProviderError(err.message)),
            }
        }
    }
    impl AudioGenerationClient for Client<reqwest::Client> {
        type AudioGenerationModel = AudioGenerationModel<reqwest::Client>;

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
        fn audio_generation_model(&self, language: &str) -> AudioGenerationModel<reqwest::Client> {
            AudioGenerationModel::new(self.clone(), language)
        }
    }
}
