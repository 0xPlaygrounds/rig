use crate::{
    model::Model,
    providers::{internal, openrouter::Client},
};
use serde::Deserialize;

/// An OpenRouter listing entry.
///
/// No `rename_all`: OpenRouter's listing is snake_case. A `camelCase` rule
/// here made serde look for `contextLength`, which the wire never sends, so
/// every model's context window decoded as `None` while the response said
/// otherwise (the recorded fixture reports 262144 for the first entry).
#[derive(Debug, Deserialize)]
struct ModelEntry {
    id: String,
    name: String,
    description: Option<String>,
    created: u64,
    context_length: Option<u32>,
    /// OpenRouter reports the output ceiling one level down.
    #[serde(default)]
    top_provider: Option<TopProvider>,
}

#[derive(Debug, Deserialize)]
struct TopProvider {
    #[serde(default)]
    max_completion_tokens: Option<u32>,
}

impl From<ModelEntry> for Model {
    fn from(value: ModelEntry) -> Self {
        Model {
            id: value.id,
            name: Some(value.name),
            description: value.description,
            r#type: None,
            created_at: Some(value.created),
            owned_by: None,
            context_length: value.context_length,
            // OpenRouter reports the output ceiling under
            // `top_provider.max_completion_tokens`. Reading it is not a guess,
            // and `max_output_tokens` exists precisely so a provider-reported
            // ceiling is not dropped (rig#2322). `None` stays `None`: some
            // entries genuinely omit it.
            max_output_tokens: value
                .top_provider
                .and_then(|provider| provider.max_completion_tokens),
        }
    }
}

internal::model_listing::impl_model_lister!(
    /// [`ModelLister`](crate::client::ModelLister) implementation for the
    /// OpenRouter API (`GET /models`).
    OpenRouterModelLister,
    Client<H>,
    ModelEntry,
    "OpenRouter",
    "/models"
);
