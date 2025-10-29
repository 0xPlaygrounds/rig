#[cfg(feature = "audio")]
use super::audio_generation::AudioGenerationModel;
use super::embedding::{
    EmbeddingModel, TEXT_EMBEDDING_3_LARGE, TEXT_EMBEDDING_3_SMALL, TEXT_EMBEDDING_ADA_002,
};
use std::fmt::Debug;

#[cfg(feature = "image")]
use super::image_generation::ImageGenerationModel;
use super::transcription::TranscriptionModel;

use crate::{
    client::{
        CompletionClient, EmbeddingsClient, ProviderClient, TranscriptionClient, VerifyClient,
        VerifyError,
    },
    extractor::ExtractorBuilder,
    http_client::{self, HttpClientExt},
    providers::openai::CompletionModel,
};

#[cfg(feature = "audio")]
use crate::client::AudioGenerationClient;
#[cfg(feature = "image")]
use crate::client::ImageGenerationClient;

use bytes::Bytes;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

// ================================================================
// Main OpenAI Client
// ================================================================
const OPENAI_API_BASE_URL: &str = "https://api.openai.com/v1";

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
            base_url: OPENAI_API_BASE_URL,
            http_client: Default::default(),
        }
    }
}

impl<'a, T> ClientBuilder<'a, T> {
    pub fn new_with_client(api_key: &'a str, http_client: T) -> Self {
        ClientBuilder {
            api_key,
            base_url: OPENAI_API_BASE_URL,
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
    pub(crate) http_client: T,
}

impl<T> Debug for Client<T>
where
    T: Debug,
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
    /// Create a new OpenAI client builder.
    ///
    /// # Example
    /// ```
    /// use rig::providers::openai::{ClientBuilder, self};
    ///
    /// // Initialize the OpenAI client
    /// let openai_client = Client::builder("your-open-ai-api-key")
    ///    .build()
    /// ```
    pub fn builder(api_key: &str) -> ClientBuilder<'_, reqwest::Client> {
        ClientBuilder::new(api_key)
    }

    /// Create a new OpenAI client. For more control, use the `builder` method.
    ///
    pub fn new(api_key: &str) -> Self {
        Self::builder(api_key).build()
    }

    pub fn from_env() -> Self {
        <Self as ProviderClient>::from_env()
    }
}

impl<T> Client<T>
where
    T: HttpClientExt + Clone + std::fmt::Debug + Default + Send + 'static,
{
    pub(crate) fn post(&self, path: &str) -> http_client::Result<http_client::Builder> {
        let url = format!("{}/{}", self.base_url, path.trim_start_matches('/'));

        http_client::with_bearer_auth(http_client::Request::post(url), &self.api_key)
    }

    pub(crate) fn get(&self, path: &str) -> http_client::Result<http_client::Builder> {
        let url = format!("{}/{}", self.base_url, path.trim_start_matches('/'));

        http_client::with_bearer_auth(http_client::Request::get(url), &self.api_key)
    }

    pub(crate) async fn send<U, R>(
        &self,
        req: http_client::Request<U>,
    ) -> http_client::Result<http_client::Response<http_client::LazyBody<R>>>
    where
        U: Into<Bytes> + Send,
        R: From<Bytes> + Send + 'static,
    {
        self.http_client.send(req).await
    }

    /// Create an extractor builder with the given completion model.
    /// Intended for use exclusively with the Chat Completions API.
    /// Useful for using extractors with Chat Completion compliant APIs.
    pub fn extractor_completions_api<U>(
        &self,
        model: &str,
    ) -> ExtractorBuilder<CompletionModel<T>, U>
    where
        U: JsonSchema + for<'a> Deserialize<'a> + Serialize + Send + Sync,
        CompletionModel<T>: crate::completion::CompletionModel,
    {
        ExtractorBuilder::new(self.completion_model(model).completions_api())
    }
}

impl<T> ProviderClient for Client<T>
where
    T: HttpClientExt + Clone + std::fmt::Debug + Default + Send + 'static,
{
    /// Create a new OpenAI client from the `OPENAI_API_KEY` environment variable.
    /// Panics if the environment variable is not set.
    fn from_env() -> Self {
        let base_url: Option<String> = std::env::var("OPENAI_BASE_URL").ok();
        let api_key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY not set");

        match base_url {
            Some(url) => ClientBuilder::<T>::new(&api_key).base_url(&url).build(),
            None => ClientBuilder::<T>::new(&api_key).build(),
        }
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
    T: HttpClientExt + std::fmt::Debug + Clone + Default + Send + 'static,
{
    type CompletionModel = super::responses_api::ResponsesCompletionModel<T>;
    /// Create a completion model with the given name.
    ///
    /// # Example
    /// ```
    /// use rig::providers::openai::{Client, self};
    ///
    /// // Initialize the OpenAI client
    /// let openai = Client::new("your-open-ai-api-key");
    ///
    /// let gpt4 = openai.completion_model(openai::GPT_4);
    /// ```
    fn completion_model(&self, model: &str) -> Self::CompletionModel {
        super::responses_api::ResponsesCompletionModel::new(self.clone(), model)
    }
}

impl<T> EmbeddingsClient for Client<T>
where
    T: HttpClientExt + std::fmt::Debug + Clone + Default + Send + 'static,
{
    type EmbeddingModel = EmbeddingModel<T>;
    fn embedding_model(&self, model: &str) -> Self::EmbeddingModel {
        let ndims = match model {
            TEXT_EMBEDDING_3_LARGE => 3072,
            TEXT_EMBEDDING_3_SMALL | TEXT_EMBEDDING_ADA_002 => 1536,
            _ => 0,
        };
        EmbeddingModel::new(self.clone(), model, ndims)
    }

    fn embedding_model_with_ndims(&self, model: &str, ndims: usize) -> Self::EmbeddingModel {
        EmbeddingModel::new(self.clone(), model, ndims)
    }
}

impl<T> TranscriptionClient for Client<T>
where
    T: HttpClientExt + Clone + std::fmt::Debug + Default + Send + 'static,
{
    type TranscriptionModel = TranscriptionModel<T>;
    /// Create a transcription model with the given name.
    ///
    /// # Example
    /// ```
    /// use rig::providers::openai::{Client, self};
    ///
    /// // Initialize the OpenAI client
    /// let openai = Client::new("your-open-ai-api-key");
    ///
    /// let gpt4 = openai.transcription_model(openai::WHISPER_1);
    /// ```
    fn transcription_model(&self, model: &str) -> Self::TranscriptionModel {
        TranscriptionModel::new(self.clone(), model)
    }
}

#[cfg(feature = "image")]
impl<T> ImageGenerationClient for Client<T>
where
    T: HttpClientExt + Clone + std::fmt::Debug + Default + Send + 'static,
{
    type ImageGenerationModel = ImageGenerationModel<T>;
    /// Create an image generation model with the given name.
    ///
    /// # Example
    /// ```
    /// use rig::providers::openai::{Client, self};
    ///
    /// // Initialize the OpenAI client
    /// let openai = Client::new("your-open-ai-api-key");
    ///
    /// let gpt4 = openai.image_generation_model(openai::DALL_E_3);
    /// ```
    fn image_generation_model(&self, model: &str) -> Self::ImageGenerationModel {
        ImageGenerationModel::new(self.clone(), model)
    }
}

#[cfg(feature = "audio")]
impl<T> AudioGenerationClient for Client<T>
where
    T: HttpClientExt + Clone + std::fmt::Debug + Send + Default + 'static,
{
    type AudioGenerationModel = AudioGenerationModel<T>;
    /// Create an audio generation model with the given name.
    ///
    /// # Example
    /// ```
    /// use rig::providers::openai::{Client, self};
    ///
    /// // Initialize the OpenAI client
    /// let openai = Client::new("your-open-ai-api-key");
    ///
    /// let gpt4 = openai.audio_generation_model(openai::TTS_1);
    /// ```
    fn audio_generation_model(&self, model: &str) -> Self::AudioGenerationModel {
        AudioGenerationModel::new(self.clone(), model)
    }
}

impl<T> VerifyClient for Client<T>
where
    T: HttpClientExt + Clone + std::fmt::Debug + Send + Default + 'static,
{
    #[cfg_attr(feature = "worker", worker::send)]
    async fn verify(&self) -> Result<(), VerifyError> {
        let req = self
            .get("/models")?
            .body(http_client::NoBody)
            .map_err(|e| VerifyError::HttpError(e.into()))?;

        let response = self.send(req).await?;

        match response.status() {
            reqwest::StatusCode::OK => Ok(()),
            reqwest::StatusCode::UNAUTHORIZED => Err(VerifyError::InvalidAuthentication),
            reqwest::StatusCode::INTERNAL_SERVER_ERROR => {
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

#[derive(Debug, Deserialize)]
pub struct ApiErrorResponse {
    pub(crate) message: String,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub(crate) enum ApiResponse<T> {
    Ok(T),
    Err(ApiErrorResponse),
}

#[cfg(test)]
mod tests {
    use crate::message::ImageDetail;
    use crate::providers::openai::{
        AssistantContent, Function, ImageUrl, Message, ToolCall, ToolType, UserContent,
    };
    use crate::{OneOrMany, message};
    use serde_path_to_error::deserialize;

    #[test]
    fn test_deserialize_message() {
        let assistant_message_json = r#"
        {
            "role": "assistant",
            "content": "\n\nHello there, how may I assist you today?"
        }
        "#;

        let assistant_message_json2 = r#"
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "\n\nHello there, how may I assist you today?"
                }
            ],
            "tool_calls": null
        }
        "#;

        let assistant_message_json3 = r#"
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_h89ipqYUjEpCPI6SxspMnoUU",
                    "type": "function",
                    "function": {
                        "name": "subtract",
                        "arguments": "{\"x\": 2, \"y\": 5}"
                    }
                }
            ],
            "content": null,
            "refusal": null
        }
        "#;

        let user_message_json = r#"
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What's in this image?"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
                    }
                },
                {
                    "type": "audio",
                    "input_audio": {
                        "data": "...",
                        "format": "mp3"
                    }
                }
            ]
        }
        "#;

        let assistant_message: Message = {
            let jd = &mut serde_json::Deserializer::from_str(assistant_message_json);
            deserialize(jd).unwrap_or_else(|err| {
                panic!(
                    "Deserialization error at {} ({}:{}): {}",
                    err.path(),
                    err.inner().line(),
                    err.inner().column(),
                    err
                );
            })
        };

        let assistant_message2: Message = {
            let jd = &mut serde_json::Deserializer::from_str(assistant_message_json2);
            deserialize(jd).unwrap_or_else(|err| {
                panic!(
                    "Deserialization error at {} ({}:{}): {}",
                    err.path(),
                    err.inner().line(),
                    err.inner().column(),
                    err
                );
            })
        };

        let assistant_message3: Message = {
            let jd: &mut serde_json::Deserializer<serde_json::de::StrRead<'_>> =
                &mut serde_json::Deserializer::from_str(assistant_message_json3);
            deserialize(jd).unwrap_or_else(|err| {
                panic!(
                    "Deserialization error at {} ({}:{}): {}",
                    err.path(),
                    err.inner().line(),
                    err.inner().column(),
                    err
                );
            })
        };

        let user_message: Message = {
            let jd = &mut serde_json::Deserializer::from_str(user_message_json);
            deserialize(jd).unwrap_or_else(|err| {
                panic!(
                    "Deserialization error at {} ({}:{}): {}",
                    err.path(),
                    err.inner().line(),
                    err.inner().column(),
                    err
                );
            })
        };

        match assistant_message {
            Message::Assistant { content, .. } => {
                assert_eq!(
                    content[0],
                    AssistantContent::Text {
                        text: "\n\nHello there, how may I assist you today?".to_string()
                    }
                );
            }
            _ => panic!("Expected assistant message"),
        }

        match assistant_message2 {
            Message::Assistant {
                content,
                tool_calls,
                ..
            } => {
                assert_eq!(
                    content[0],
                    AssistantContent::Text {
                        text: "\n\nHello there, how may I assist you today?".to_string()
                    }
                );

                assert_eq!(tool_calls, vec![]);
            }
            _ => panic!("Expected assistant message"),
        }

        match assistant_message3 {
            Message::Assistant {
                content,
                tool_calls,
                refusal,
                ..
            } => {
                assert!(content.is_empty());
                assert!(refusal.is_none());
                assert_eq!(
                    tool_calls[0],
                    ToolCall {
                        id: "call_h89ipqYUjEpCPI6SxspMnoUU".to_string(),
                        r#type: ToolType::Function,
                        function: Function {
                            name: "subtract".to_string(),
                            arguments: serde_json::json!({"x": 2, "y": 5}),
                        },
                    }
                );
            }
            _ => panic!("Expected assistant message"),
        }

        match user_message {
            Message::User { content, .. } => {
                let (first, second) = {
                    let mut iter = content.into_iter();
                    (iter.next().unwrap(), iter.next().unwrap())
                };
                assert_eq!(
                    first,
                    UserContent::Text {
                        text: "What's in this image?".to_string()
                    }
                );
                assert_eq!(second, UserContent::Image { image_url: ImageUrl { url: "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg".to_string(), detail: ImageDetail::default() } });
            }
            _ => panic!("Expected user message"),
        }
    }

    #[test]
    fn test_message_to_message_conversion() {
        let user_message = message::Message::User {
            content: OneOrMany::one(message::UserContent::text("Hello")),
        };

        let assistant_message = message::Message::Assistant {
            id: None,
            content: OneOrMany::one(message::AssistantContent::text("Hi there!")),
        };

        let converted_user_message: Vec<Message> = user_message.clone().try_into().unwrap();
        let converted_assistant_message: Vec<Message> =
            assistant_message.clone().try_into().unwrap();

        match converted_user_message[0].clone() {
            Message::User { content, .. } => {
                assert_eq!(
                    content.first(),
                    UserContent::Text {
                        text: "Hello".to_string()
                    }
                );
            }
            _ => panic!("Expected user message"),
        }

        match converted_assistant_message[0].clone() {
            Message::Assistant { content, .. } => {
                assert_eq!(
                    content[0].clone(),
                    AssistantContent::Text {
                        text: "Hi there!".to_string()
                    }
                );
            }
            _ => panic!("Expected assistant message"),
        }

        let original_user_message: message::Message =
            converted_user_message[0].clone().try_into().unwrap();
        let original_assistant_message: message::Message =
            converted_assistant_message[0].clone().try_into().unwrap();

        assert_eq!(original_user_message, user_message);
        assert_eq!(original_assistant_message, assistant_message);
    }

    #[test]
    fn test_message_from_message_conversion() {
        let user_message = Message::User {
            content: OneOrMany::one(UserContent::Text {
                text: "Hello".to_string(),
            }),
            name: None,
        };

        let assistant_message = Message::Assistant {
            content: vec![AssistantContent::Text {
                text: "Hi there!".to_string(),
            }],
            refusal: None,
            audio: None,
            name: None,
            tool_calls: vec![],
        };

        let converted_user_message: message::Message = user_message.clone().try_into().unwrap();
        let converted_assistant_message: message::Message =
            assistant_message.clone().try_into().unwrap();

        match converted_user_message.clone() {
            message::Message::User { content } => {
                assert_eq!(content.first(), message::UserContent::text("Hello"));
            }
            _ => panic!("Expected user message"),
        }

        match converted_assistant_message.clone() {
            message::Message::Assistant { content, .. } => {
                assert_eq!(
                    content.first(),
                    message::AssistantContent::text("Hi there!")
                );
            }
            _ => panic!("Expected assistant message"),
        }

        let original_user_message: Vec<Message> = converted_user_message.try_into().unwrap();
        let original_assistant_message: Vec<Message> =
            converted_assistant_message.try_into().unwrap();

        assert_eq!(original_user_message[0], user_message);
        assert_eq!(original_assistant_message[0], assistant_message);
    }
}
