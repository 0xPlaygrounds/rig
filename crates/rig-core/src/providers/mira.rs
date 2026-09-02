//! Mira API client and Rig integration
//!
//! # Example
//! ```ignore
//! use rig_core::providers::mira;
//!
//! let client = mira::Client::new("YOUR_API_KEY");
//!
//! ```
use crate::client::{self, BearerAuth, DebugExt, Provider};
use crate::completion::{self, CompletionError};
use serde::{Deserialize, Serialize};
use tracing::{self};

#[derive(Debug, Default, Clone, Copy)]
pub struct MiraExt;
#[derive(Debug, Default, Clone, Copy)]
pub struct MiraBuilder;

type MiraApiKey = BearerAuth;

impl Provider for MiraExt {
    type Builder = MiraBuilder;

    const VERIFY_PATH: &'static str = "/user-credits";
}

client::impl_capabilities!(
    MiraExt,
    completion = CompletionModel<H>,
    model_listing = MiraModelLister<H>,
);

crate::providers::internal::model_listing::impl_model_lister!(
    /// [`ModelLister`](crate::client::ModelLister) implementation for the
    /// Mira API (`GET /v1/models`).
    MiraModelLister,
    Client<H>,
    crate::providers::internal::model_listing::ListModelEntry,
    "Mira",
    "/v1/models"
);

impl DebugExt for MiraExt {}

impl crate::providers::openai::completion::OpenAICompatibleProvider for MiraExt {
    const PROVIDER_NAME: &'static str = "mira";

    // Mira's gateway rejects tool parameters.
    const SUPPORTS_TOOLS: bool = false;

    type StreamingUsage = crate::providers::openai::Usage;

    // Mira's gateway does not accept OpenAI structured-output parameters.
    const SUPPORTS_RESPONSE_FORMAT: bool = false;

    // The gateway also rejects unknown parameters like `stream_options`.
    const STREAM_INCLUDE_USAGE: bool = false;

    type Response = CompletionResponse;

    // The client base URL is the bare host; `list_models` builds its own v1 path.
    fn completion_path(&self, _model: &str) -> String {
        "/v1/chat/completions".to_string()
    }

    fn prepare_request(
        &self,
        request: &mut crate::providers::openai::completion::CompletionRequest,
    ) -> Result<(), CompletionError> {
        // Mira's gateway rejects pass-through parameters (tools are dropped
        // via `SUPPORTS_TOOLS = false` during conversion).
        if request.additional_params.take().is_some() {
            tracing::warn!("Additional parameters are not supported by Mira and will be ignored");
        }

        Ok(())
    }

    fn finalize_request_body(&self, body: &mut serde_json::Value) -> Result<(), CompletionError> {
        let Some(map) = body.as_object_mut() else {
            return Ok(());
        };

        // Mira only understands plain `{role, content}` string messages;
        // strip tool-exchange remnants and message names, and flatten
        // content-part arrays.
        if let Some(messages) = map
            .get_mut("messages")
            .and_then(serde_json::Value::as_array_mut)
        {
            crate::providers::openai::completion::sanitize_plain_text_history(
                messages,
                Some(("\n", false)),
                true,
                false,
            );
        }

        Ok(())
    }
}

client::impl_default_provider_builder!(
    MiraBuilder => MiraExt,
    api_key = MiraApiKey,
    base_url = MIRA_API_BASE_URL,
);

pub type Client<H = crate::http_client::BoxedHttpClient> = client::Client<MiraExt, H>;
pub type ClientBuilder<H = crate::markers::Missing> =
    client::ClientBuilder<MiraBuilder, MiraApiKey, H>;

#[derive(Debug, Deserialize, Clone, Serialize)]
pub struct RawMessage {
    pub role: String,
    pub content: String,
}

const MIRA_API_BASE_URL: &str = "https://api.mira.network";

#[derive(Debug, Deserialize, Serialize)]
#[serde(untagged)]
pub enum CompletionResponse {
    Structured {
        id: String,
        object: String,
        created: u64,
        model: String,
        choices: Vec<ChatChoice>,
        #[serde(skip_serializing_if = "Option::is_none")]
        usage: Option<Usage>,
    },
    Simple(String),
}

#[derive(Debug, Deserialize, Serialize)]
pub struct ChatChoice {
    pub message: RawMessage,
    #[serde(default)]
    pub finish_reason: Option<String>,
    #[serde(default)]
    pub index: Option<usize>,
}

client::impl_provider_from_env!(MiraExt, input = String, api_key_env = "MIRA_API_KEY");

/// Mira completion model, driven by the shared OpenAI Chat Completions path.
pub type CompletionModel<H = crate::http_client::BoxedHttpClient> =
    crate::providers::openai::completion::GenericCompletionModel<MiraExt, H>;

impl crate::telemetry::ProviderResponseExt for CompletionResponse {
    type Usage = Usage;

    fn response_id(&self) -> Option<&str> {
        match self {
            Self::Structured { id, .. } => Some(id.as_str()),
            Self::Simple(_) => None,
        }
    }

    fn response_model_name(&self) -> Option<&str> {
        match self {
            Self::Structured { model, .. } => Some(model.as_str()),
            Self::Simple(_) => None,
        }
    }

    fn text_response(&self) -> Option<String> {
        match self {
            Self::Structured { choices, .. } => choices
                .iter()
                .find(|choice| choice.message.role == "assistant")
                .map(|choice| choice.message.content.clone()),
            Self::Simple(text) => Some(text.clone()),
        }
    }

    fn usage(&self) -> Option<Self::Usage> {
        match self {
            Self::Structured { usage, .. } => usage.clone(),
            Self::Simple(_) => None,
        }
    }
}

impl From<&Usage> for completion::Usage {
    fn from(usage: &Usage) -> Self {
        crate::providers::internal::completion_usage(
            usage.prompt_tokens as u64,
            // Mira reports only prompt and total counts; the completion count
            // is the remainder.
            usage.total_tokens.saturating_sub(usage.prompt_tokens) as u64,
            usage.total_tokens as u64,
            0,
        )
    }
}

impl From<Usage> for completion::Usage {
    fn from(usage: Usage) -> Self {
        Self::from(&usage)
    }
}

/// Normalize a Mira chat completion response.
///
/// The provider descriptor name is an *input* rather than a constant so the
/// shared OpenAI-compatible completion path labels the response with the
/// descriptor that actually produced it.
impl crate::completion::NormalizeCompletionResponse for CompletionResponse {
    fn normalize(self, provider: &str) -> Result<completion::CompletionResponse, CompletionError> {
        use crate::providers::internal::openai_chat_completions_compatible as compat;

        let (id, model, choices, usage) = match self {
            CompletionResponse::Structured {
                id,
                model,
                choices,
                usage,
                ..
            } => (id, model, choices, usage),
            // The bare-string variant carries no metadata at all — not even a
            // terminal reason, so the normalized reason stays `None`.
            CompletionResponse::Simple(text) => {
                let choice = crate::message::require_non_empty_response(vec![
                    completion::AssistantContent::text(&text),
                ])?;
                return Ok(completion::CompletionResponse::new(
                    choice,
                    completion::Usage::new(),
                    provider,
                ));
            }
        };

        // Preserve Mira's role-specific error messages: the shared helper
        // folds every non-assistant message into one generic error. Mira's
        // wire messages are plain `{role, content}` strings, so an assistant
        // message can never carry unsupported content types.
        if let Some(choice) = choices.first() {
            match choice.message.role.as_str() {
                "assistant" => {}
                "user" => {
                    tracing::warn!(target: "rig", "Received user message in response where assistant message was expected");
                    return Err(CompletionError::ResponseError(
                        "Received user message in response where assistant message was expected"
                            .to_owned(),
                    ));
                }
                "system" => {
                    tracing::warn!(target: "rig", "Received system message in response where assistant message was expected");
                    return Err(CompletionError::ResponseError(
                        "Received system message in response where assistant message was expected"
                            .to_owned(),
                    ));
                }
                other => {
                    return Err(CompletionError::ResponseError(format!(
                        "Unsupported message role: {other}"
                    )));
                }
            }
        }

        let usage = usage
            .as_ref()
            .map(completion::Usage::from)
            .unwrap_or_default();

        compat::normalize_openai_response(
            provider,
            &choices,
            Some(id.as_str()).filter(|id| !id.is_empty()),
            Some(model.as_str()).filter(|model| !model.is_empty()),
            usage,
            |choice| choice.finish_reason.as_deref().unwrap_or(""),
            |choice| {
                Some(vec![completion::AssistantContent::text(
                    &choice.message.content,
                )])
            },
        )
    }
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

#[cfg(test)]
mod tests;
