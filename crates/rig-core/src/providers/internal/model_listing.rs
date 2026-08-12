//! Shared plumbing for provider `GET /models` listings.
//!
//! Every provider's model listing is the same HTTP conversation — GET a
//! path, triage the status, decode a `{ "data": [...] }` envelope, convert
//! entries into [`Model`]s — differing only in the path, the provider label
//! used in error context, and the entry DTO. The DTOs and their
//! `From<Entry> for Model` impls stay in each provider module (that mapping
//! is genuinely provider-specific); the conversation lives here once.

use crate::{
    client::{Client, Provider},
    http_client::{self, HttpClientExt},
    model::{Model, ModelList, ModelListingError},
    wasm_compat::{WasmCompatSend, WasmCompatSync},
};

/// The standard `{ "data": [...] }` list envelope shared by OpenAI-style
/// listing endpoints.
#[derive(Debug, serde::Deserialize)]
pub(crate) struct DataEnvelope<Entry> {
    pub(crate) data: Vec<Entry>,
}

/// The standard OpenAI-style listing entry (OpenAI, Mistral, DeepSeek,
/// Xiaomi MiMo). `id` is the one field every listing carries; the rest are
/// optional so providers that omit them still decode. Providers whose
/// entries genuinely diverge keep their own DTO.
#[derive(Debug, serde::Deserialize)]
pub(crate) struct ListModelEntry {
    pub(crate) id: String,
    pub(crate) name: Option<String>,
    pub(crate) created: Option<u64>,
    pub(crate) owned_by: Option<String>,
}

impl From<ListModelEntry> for Model {
    fn from(value: ListModelEntry) -> Self {
        let mut model = Model::from_id(value.id);
        model.name = value.name;
        model.created_at = value.created;
        model.owned_by = value.owned_by;
        model
    }
}

/// Map a transport-level send error into listing-flavored context: an
/// [`http_client::Error::InvalidStatusCodeWithMessage`] (backends that reject
/// non-2xx before handing back a response) keeps the provider label, path,
/// status, and body preview, exactly like a non-2xx status on a returned
/// response. Shared with listings that build their own request (copilot's
/// auth-derived base URL cannot go through [`get_json`]).
pub(crate) fn map_transport_error(
    provider_name: &str,
    path: &str,
    error: http_client::Error,
) -> ModelListingError {
    match error {
        http_client::Error::InvalidStatusCodeWithMessage(status, message) => {
            ModelListingError::api_error_with_context(
                provider_name,
                path,
                status.as_u16(),
                message.as_bytes(),
            )
        }
        other => ModelListingError::from(other),
    }
}

/// GET `path` and decode the response body as `T`, with listing-flavored
/// error context.
///
/// Error triage is standardized on the most informative behavior: an
/// [`http_client::Error::InvalidStatusCodeWithMessage`] surfaced by the
/// transport (backends that reject non-2xx before handing back a response)
/// is mapped into [`ModelListingError::api_error_with_context`] so the
/// provider label, path, status, and body preview survive, exactly like a
/// non-2xx status on a returned response.
pub(crate) async fn get_json<T, Ext, H>(
    client: &Client<Ext, H>,
    provider_name: &str,
    path: &str,
) -> Result<T, ModelListingError>
where
    T: serde::de::DeserializeOwned,
    Ext: Provider + WasmCompatSend + WasmCompatSync + 'static,
    H: HttpClientExt + WasmCompatSend + WasmCompatSync + 'static,
{
    let req = client.get(path)?.body(http_client::NoBody)?;
    let response = client
        .send::<_, Vec<u8>>(req)
        .await
        .map_err(|error| map_transport_error(provider_name, path, error))?;

    if !response.status().is_success() {
        let status_code = response.status().as_u16();
        let body = response.into_body().await?;
        return Err(ModelListingError::api_error_with_context(
            provider_name,
            path,
            status_code,
            &body,
        ));
    }

    let body = response.into_body().await?;
    serde_json::from_slice(&body).map_err(|error| {
        ModelListingError::parse_error_with_context(provider_name, path, &error, &body)
    })
}

/// List models from an OpenAI-style `{ "data": [Entry, ...] }` endpoint,
/// converting each entry via `Entry: Into<Model>`.
pub(crate) async fn list_models<Entry, Ext, H>(
    client: &Client<Ext, H>,
    provider_name: &str,
    path: &str,
) -> Result<ModelList, ModelListingError>
where
    Entry: serde::de::DeserializeOwned + Into<Model>,
    Ext: Provider + WasmCompatSend + WasmCompatSync + 'static,
    H: HttpClientExt + WasmCompatSend + WasmCompatSync + 'static,
{
    let envelope: DataEnvelope<Entry> = get_json(client, provider_name, path).await?;
    let models = envelope.data.into_iter().map(Into::into).collect();
    Ok(ModelList::new(models))
}

#[cfg(test)]
mod tests {
    use super::ListModelEntry;
    use crate::model::Model;

    /// `id` is the one required field: an entry that carries nothing else
    /// (some OpenAI-compatible gateways omit `created`/`owned_by`) still
    /// decodes, mapping the absent fields to `None` on the `Model`.
    #[test]
    fn minimal_entry_decodes_with_id_alone() {
        let entry: ListModelEntry =
            serde_json::from_str(r#"{"id":"gpt-test"}"#).expect("minimal entry should decode");
        let model = Model::from(entry);
        assert_eq!(model.id, "gpt-test");
        assert_eq!(model.name, None);
        assert_eq!(model.created_at, None);
        assert_eq!(model.owned_by, None);
    }
}
