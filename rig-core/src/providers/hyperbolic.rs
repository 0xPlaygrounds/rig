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
use http::Method;
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
    pub fn new_with_client(api_key: &'a str, http_client: T) -> Self {
        Self {
            api_key,
            base_url: HYPERBOLIC_API_BASE_URL,
            http_client,
        }
    }

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

impl Client<reqwest::Client> {
    pub fn builder(api_key: &str) -> ClientBuilder<'_, reqwest::Client> {
        ClientBuilder::new(api_key)
    }

    pub fn new(api_key: &str) -> Self {
        Self::builder(api_key).build()
    }

    pub fn from_env() -> Self {
        <Self as ProviderClient>::from_env()
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

impl<T> ProviderClient for Client<T>
where
    T: HttpClientExt + Clone + Default + std::fmt::Debug + Send + 'static,
{
    /// Create a new Hyperbolic client from the `HYPERBOLIC_API_KEY` environment variable.
    /// Panics if the environment variable is not set.
    fn from_env() -> Self {
        let api_key = std::env::var("HYPERBOLIC_API_KEY").expect("HYPERBOLIC_API_KEY not set");
        ClientBuilder::<T>::new(&api_key).build()
    }

    fn from_val(input: crate::client::ProviderValue) -> Self {
        let crate::client::ProviderValue::Simple(api_key) = input else {
            panic!("Incorrect provider value type")
        };
        ClientBuilder::<T>::new(&api_key).build()
    }
}

impl<T> CompletionClient for Client<T>
where
    T: HttpClientExt + Clone + Default + std::fmt::Debug + Send + 'static,
{
    type CompletionModel = CompletionModel<T>;

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
    fn completion_model(&self, model: &str) -> Self::CompletionModel {
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

impl<T> completion::CompletionModel for CompletionModel<T>
where
    T: HttpClientExt + Clone + Default + std::fmt::Debug + Send + 'static,
{
    type Response = CompletionResponse;
    type StreamingResponse = openai::StreamingCompletionResponse;

    #[cfg_attr(feature = "worker", worker::send)]
    async fn completion(
        &self,
        completion_request: CompletionRequest,
    ) -> Result<completion::CompletionResponse<CompletionResponse>, CompletionError> {
        let preamble = completion_request.preamble.clone();
        let request = self.create_completion_request(completion_request)?;
        let body = serde_json::to_vec(&request)?;

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

        let req = self
            .client
            .req(Method::POST, "/v1/chat/completions")?
            .header("Content-Type", "application/json")
            .body(body)
            .map_err(http_client::Error::from)?;

        let async_block = async move {
            let response = self.client.http_client.send::<_, bytes::Bytes>(req).await?;

            let status = response.status();
            let response_body = response.into_body().into_future().await?.to_vec();

            if status.is_success() {
                match serde_json::from_slice::<ApiResponse<CompletionResponse>>(&response_body)? {
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
                    String::from_utf8_lossy(&response_body).to_string(),
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

        let body = serde_json::to_vec(&request)?;

        let req = self
            .client
            .req(Method::POST, "/v1/chat/completions")?
            .header("Content-Type", "application/json")
            .body(body)
            .map_err(http_client::Error::from)?;

        send_compatible_streaming_request(self.client.http_client.clone(), req)
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
    use crate::http_client::HttpClientExt;
    use crate::image_generation;
    use crate::image_generation::{ImageGenerationError, ImageGenerationRequest};
    use crate::json_utils::merge_inplace;
    use base64::Engine;
    use base64::prelude::BASE64_STANDARD;
    use http::Method;
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

    impl<T> image_generation::ImageGenerationModel for ImageGenerationModel<T>
    where
        T: HttpClientExt + Clone + Default + std::fmt::Debug + Send + 'static,
    {
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

            let body = serde_json::to_vec(&request)?;

            let request = self
                .client
                .req(Method::POST, "/v1/image/generation")?
                .header("Content-Type", "application/json")
                .body(body)
                .map_err(|e| ImageGenerationError::HttpError(e.into()))?;

            let response = self
                .client
                .http_client
                .send::<_, bytes::Bytes>(request)
                .await?;

            let status = response.status();
            let response_body = response.into_body().into_future().await?.to_vec();

            if !status.is_success() {
                return Err(ImageGenerationError::ProviderError(format!(
                    "{status}: {}",
                    String::from_utf8_lossy(&response_body)
                )));
            }

            match serde_json::from_slice::<ApiResponse<ImageGenerationResponse>>(&response_body)? {
                ApiResponse::Ok(response) => response.try_into(),
                ApiResponse::Err(err) => Err(ImageGenerationError::ResponseError(err.message)),
            }
        }
    }

    impl<T> ImageGenerationClient for Client<T>
    where
        T: HttpClientExt + Clone + Default + std::fmt::Debug + Send + 'static,
    {
        type ImageGenerationModel = ImageGenerationModel<T>;

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
        fn image_generation_model(&self, model: &str) -> Self::ImageGenerationModel {
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
    use crate::http_client::{self, HttpClientExt};
    use base64::Engine;
    use base64::prelude::BASE64_STANDARD;
    use bytes::Bytes;
    use http::Method;
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

    impl<T> audio_generation::AudioGenerationModel for AudioGenerationModel<T>
    where
        T: HttpClientExt + Clone + Default + std::fmt::Debug + Send + 'static,
    {
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

            let body = serde_json::to_vec(&request)?;

            let req = self
                .client
                .req(Method::POST, "/v1/audio/generation")?
                .header("Content-Type", "application/json")
                .body(body)
                .map_err(http_client::Error::from)?;

            let response = self.client.http_client.send::<_, Bytes>(req).await?;
            let status = response.status();
            let response_body = response.into_body().into_future().await?.to_vec();

            if !status.is_success() {
                return Err(AudioGenerationError::ProviderError(format!(
                    "{status}: {}",
                    String::from_utf8_lossy(&response_body)
                )));
            }

            match serde_json::from_slice::<ApiResponse<AudioGenerationResponse>>(&response_body)? {
                ApiResponse::Ok(response) => response.try_into(),
                ApiResponse::Err(err) => Err(AudioGenerationError::ProviderError(err.message)),
            }
        }
    }
    impl<T> AudioGenerationClient for Client<T>
    where
        T: HttpClientExt + Clone + Default + std::fmt::Debug + Send + 'static,
    {
        type AudioGenerationModel = AudioGenerationModel<T>;

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
        fn audio_generation_model(&self, language: &str) -> Self::AudioGenerationModel {
            AudioGenerationModel::new(self.clone(), language)
        }
    }
}
