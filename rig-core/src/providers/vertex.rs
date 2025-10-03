//! Bindings to the Google Vertex AI API.
//! This API primarily builds on Gemini, so there is very little change required.
//!
//! Relevant documentation: <https://cloud.google.com/vertex-ai/generative-ai/docs/start/quickstart?usertype=apikey>

use crate::{providers::gemini::completion::gemini_api_types::PartKind, streaming};
use futures::StreamExt;
use reqwest::RequestBuilder;
use reqwest_eventsource::Event;
use reqwest_eventsource::RequestBuilderExt;
use tracing::info_span;

use crate::{
    client::{ClientBuilderError, CompletionClient, ProviderClient},
    completion::{CompletionError, CompletionRequest},
    impl_conversion_traits,
    providers::gemini::{
        completion::gemini_api_types::{GenerateContentResponse, Part},
        streaming::{StreamGenerateContentResponse, StreamingCompletionResponse},
    },
    telemetry::SpanCombinator,
};

/// The Google Vertex AI client.
#[derive(Clone)]
pub struct Client {
    api_key: String,
    client: reqwest::Client,
    api_endpoint: String,
}

impl Client {
    /// Create a fluent builder for this struct.
    pub fn builder<'a>(api_key: &'a str) -> ClientBuilder<'a> {
        ClientBuilder::new(api_key)
    }

    pub(crate) fn post(&self, model: &str) -> RequestBuilder {
        let url = format!(
            "https://{api_endpoint}/v1/publishers/google/models/{model}:generateContent?key={api_key}",
            api_endpoint = self.api_endpoint,
            api_key = self.api_key
        );

        self.client.post(url).bearer_auth(&self.api_key)
    }

    pub(crate) fn post_sse(&self, model: &str) -> RequestBuilder {
        let url = format!(
            "https://{api_endpoint}/v1/publishers/google/models/{model}:streamGenerateContent?alt=sse&key={api_key}",
            api_endpoint = self.api_endpoint,
            api_key = self.api_key
        );

        self.client.post(url)
    }
}

impl std::fmt::Debug for Client {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Client")
            .field("api_key", &"<REDACTED>")
            .field("client", &self.client)
            .field("project_id", &"<REDACTED>")
            .field("location", &"<REDACTED>")
            .field("api_endpoint", &"<REDACTED>")
            .finish()
    }
}

/// A fluent client builder to build [`Client`].
pub struct ClientBuilder<'a> {
    api_key: &'a str,
    api_endpoint: Option<&'a str>,
    client: Option<reqwest::Client>,
}

impl<'a> ClientBuilder<'a> {
    /// Create a new instance of `ClientBuilder`.
    pub fn new(api_key: &'a str) -> Self {
        Self {
            api_key,
            client: None,
            api_endpoint: None,
        }
    }

    /// Add a custom HTTP client.
    /// If not set, the client builder will attempt to use a default `reqwest::Client` on building.
    pub fn http_client(mut self, client: reqwest::Client) -> Self {
        self.client = Some(client);
        self
    }

    pub fn api_endpoint(mut self, api_endpoint: &'a str) -> Self {
        self.api_endpoint = Some(api_endpoint);
        self
    }

    pub fn build(self) -> Result<Client, ClientBuilderError> {
        let api_endpoint = if let Some(endpoint) = self.api_endpoint {
            endpoint.to_string()
        } else {
            return Err(ClientBuilderError::InvalidProperty(
                "API endpoint is required",
            ));
        };

        let client = if let Some(client) = self.client {
            client
        } else {
            reqwest::Client::new()
        };

        let client = Client {
            api_key: self.api_key.to_string(),
            api_endpoint,
            client,
        };

        Ok(client)
    }
}

impl_conversion_traits!(
    AsEmbeddings,
    AsAudioGeneration,
    AsImageGeneration,
    AsTranscription,
    AsVerify for Client
);

impl ProviderClient for Client {
    /// Using this method requires the following environment variables:
    /// - `VERTEX_API_KEY` (your google cloud auth token)
    /// - `VERTEX_API_ENDPOINT` (your Vertex API endpoint)
    fn from_env() -> Self {
        let api_key =
            std::env::var("VERTEX_API_KEY").expect("VERTEX_API_KEY to an existing env var");
        let api_endpoint = std::env::var("GOOGLE_CLOUD_API_ENDPOINT")
            .expect("GOOGLE_CLOUD_API_ENDPOINT to an existing env var");
        let client = reqwest::Client::new();

        Self {
            api_key,
            api_endpoint,
            client,
        }
    }

    // Listed as TODO because DynClientBuilder needs a bit of an overhaul, which should be done soon
    fn from_val(_input: crate::client::ProviderValue) -> Self {
        todo!("Implement DynClientBuilder stuff for Vertex")
    }
}

impl CompletionClient for Client {
    type CompletionModel = CompletionModel;

    fn completion_model(&self, model: &str) -> Self::CompletionModel {
        Self::CompletionModel::new(self.clone(), model)
    }
}

/// A completion model for the Google Vertex AI API.
#[derive(Clone, Debug)]
pub struct CompletionModel {
    client: Client,
    model_name: String,
}

impl CompletionModel {
    pub fn new(client: Client, model_name: &str) -> Self {
        Self {
            client,
            model_name: model_name.to_string(),
        }
    }
}

impl crate::completion::CompletionModel for CompletionModel {
    type Response = GenerateContentResponse;
    type StreamingResponse = StreamingCompletionResponse;

    #[cfg_attr(feature = "worker", worker::send)]
    async fn completion(
        &self,
        completion_request: CompletionRequest,
    ) -> Result<crate::completion::CompletionResponse<GenerateContentResponse>, CompletionError>
    {
        let span = if tracing::Span::current().is_disabled() {
            info_span!(
                target: "rig::completions",
                "generate_content",
                gen_ai.operation.name = "generate_content",
                gen_ai.provider.name = "gcp.vertex",
                gen_ai.request.model = self.model_name,
                gen_ai.system_instructions = &completion_request.preamble,
                gen_ai.response.id = tracing::field::Empty,
                gen_ai.response.model = tracing::field::Empty,
                gen_ai.usage.output_tokens = tracing::field::Empty,
                gen_ai.usage.input_tokens = tracing::field::Empty,
                gen_ai.input.messages = tracing::field::Empty,
                gen_ai.output.messages = tracing::field::Empty,
            )
        } else {
            tracing::Span::current()
        };

        let request =
            crate::providers::gemini::completion::create_request_body(completion_request)?;
        span.record_model_input(&request.contents);

        tracing::debug!(
            "Sending completion request to Gemini API {}",
            serde_json::to_string_pretty(&request)?
        );

        let response = self
            .client
            .post(&self.model_name)
            .json(&request)
            .send()
            .await?;

        if response.status().is_success() {
            let response = response.json::<GenerateContentResponse>().await?;

            let span = tracing::Span::current();
            span.record_model_output(&response.candidates);
            span.record_response_metadata(&response);
            span.record_token_usage(&response.usage_metadata);

            tracing::debug!(
                "Received response from Gemini API: {}",
                serde_json::to_string_pretty(&request)?
            );

            Ok(crate::completion::CompletionResponse::try_from(response))
        } else {
            Err(CompletionError::ProviderError(response.text().await?))
        }?
    }

    #[cfg_attr(feature = "worker", worker::send)]
    async fn stream(
        &self,
        request: CompletionRequest,
    ) -> Result<
        crate::streaming::StreamingCompletionResponse<Self::StreamingResponse>,
        CompletionError,
    > {
        CompletionModel::stream(self, request).await
    }
}

impl CompletionModel {
    pub(crate) async fn stream(
        &self,
        completion_request: CompletionRequest,
    ) -> Result<
        crate::streaming::StreamingCompletionResponse<StreamingCompletionResponse>,
        CompletionError,
    > {
        let span = if tracing::Span::current().is_disabled() {
            info_span!(
                target: "rig::completions",
                "chat_streaming",
                gen_ai.operation.name = "chat_streaming",
                gen_ai.provider.name = "gcp.vertex",
                gen_ai.request.model = &self.model_name,
                gen_ai.system_instructions = &completion_request.preamble,
                gen_ai.response.id = tracing::field::Empty,
                gen_ai.response.model = &self.model_name,
                gen_ai.usage.output_tokens = tracing::field::Empty,
                gen_ai.usage.input_tokens = tracing::field::Empty,
                gen_ai.input.messages = tracing::field::Empty,
                gen_ai.output.messages = tracing::field::Empty,
            )
        } else {
            tracing::Span::current()
        };

        let request =
            crate::providers::gemini::completion::create_request_body(completion_request)?;

        span.record_model_input(&request.contents);

        tracing::debug!(
            "Sending completion request to Gemini API {}",
            serde_json::to_string_pretty(&request)?
        );

        // Build the request with proper headers for SSE
        let mut event_source = self
            .client
            .post_sse(&self.model_name)
            .json(&request)
            .eventsource()
            .expect("Cloning request must always succeed");

        let stream = Box::pin(async_stream::stream! {
            let mut text_response = String::new();
            let mut model_outputs: Vec<Part> = Vec::new();
            while let Some(event_result) = event_source.next().await {
                match event_result {
                    Ok(Event::Open) => {
                        tracing::trace!("SSE connection opened");
                        continue;
                    }
                    Ok(Event::Message(message)) => {
                        // Skip heartbeat messages or empty data
                        if message.data.trim().is_empty() {
                            continue;
                        }

                        let data = match serde_json::from_str::<StreamGenerateContentResponse>(&message.data) {
                            Ok(d) => d,
                            Err(error) => {
                                tracing::error!(?error, message = message.data, "Failed to parse SSE message");
                                continue;
                            }
                        };

                        // Process the response data
                        let Some(choice) = data.candidates.first() else {
                            tracing::debug!("There is no content candidate");
                            continue;
                        };

                        match choice.content.parts.first() {
                            Some(Part {
                                part: PartKind::Text(text),
                                thought: Some(true),
                                ..
                            }) => {
                                yield Ok(streaming::RawStreamingChoice::Reasoning { reasoning: text.clone(), id: None });
                            },
                            Some(Part {
                                part: PartKind::Text(text),
                                ..
                            }) => {
                                text_response += text;
                                yield Ok(streaming::RawStreamingChoice::Message(text.clone()));
                            },
                            Some(Part {
                                part: PartKind::FunctionCall(function_call),
                                ..
                            }) => {
                                model_outputs.push(choice.content.parts.first().cloned().expect("This should never fail"));
                                yield Ok(streaming::RawStreamingChoice::ToolCall {
                                    name: function_call.name.clone(),
                                    id: function_call.name.clone(),
                                    arguments: function_call.args.clone(),
                                    call_id: None
                                });
                            },
                            Some(part) => {
                                tracing::warn!(?part, "Unsupported response type with streaming");
                            }
                            None => tracing::trace!(reason = ?choice.finish_reason, "There is no part in the streaming content"),
                        }

                        // Check if this is the final response
                        if choice.finish_reason.is_some() {
                            if !text_response.is_empty() {
                                model_outputs.push(Part { thought: None, thought_signature: None, part: PartKind::Text(text_response), additional_params: None });
                            }
                            let span = tracing::Span::current();
                            span.record_model_output(&model_outputs);
                            span.record_token_usage(&data.usage_metadata);
                            yield Ok(streaming::RawStreamingChoice::FinalResponse(StreamingCompletionResponse {
                                usage_metadata: data.usage_metadata.unwrap_or_default()
                            }));
                            break;
                        }
                    }
                    Err(reqwest_eventsource::Error::StreamEnded) => {
                        break;
                    }
                    Err(error) => {
                        tracing::error!(?error, "SSE error");
                        yield Err(CompletionError::ResponseError(error.to_string()));
                        break;
                    }
                }
            }

            // Ensure event source is closed when stream ends
            event_source.close();
        });

        Ok(streaming::StreamingCompletionResponse::stream(stream))
    }
}
