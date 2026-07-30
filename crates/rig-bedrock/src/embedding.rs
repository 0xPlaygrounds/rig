//! Bedrock embedding model identifiers and the single-document
//! `InvokeModel` wire types.
//!
//! Embedding calls are the free functions [`crate::functions::embed`] /
//! [`crate::functions::embed_batches`], which drive
//! [`invoke_embedding`] once per document.

use aws_smithy_types::Blob;
use rig_core::embeddings::EmbeddingError;
use serde::{Deserialize, Serialize};

use crate::types::errors::AwsSdkInvokeModelError;

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct EmbeddingRequest {
    pub input_text: String,
    pub dimensions: usize,
    pub normalize: bool,
}

#[derive(Deserialize, Debug)]
#[serde(rename_all = "camelCase")]
pub struct EmbeddingResponse {
    pub embedding: Vec<f64>,
    pub input_text_token_count: usize,
}

/// `amazon.titan-embed-text-v1`
pub const AMAZON_TITAN_EMBED_TEXT_V1: &str = "amazon.titan-embed-text-v1";
/// `amazon.titan-embed-text-v2:0`
pub const AMAZON_TITAN_EMBED_TEXT_V2_0: &str = "amazon.titan-embed-text-v2:0";
/// `amazon.titan-embed-image-v1`
pub const AMAZON_TITAN_EMBED_IMAGE_V1: &str = "amazon.titan-embed-image-v1";
/// `cohere.embed-english-v3`
pub const COHERE_EMBED_ENGLISH_V3: &str = "cohere.embed-english-v3";
/// `cohere.embed-multilingual-v3`
pub const COHERE_EMBED_MULTILINGUAL_V3: &str = "cohere.embed-multilingual-v3";

/// Invoke `model` for one embedding document over the AWS SDK.
///
/// Drives [`crate::functions::embed`], which calls it once per document
/// (Bedrock's embedding API is single-document).
pub(crate) async fn invoke_embedding(
    client: &aws_sdk_bedrockruntime::Client,
    model: &str,
    request: EmbeddingRequest,
) -> Result<EmbeddingResponse, EmbeddingError> {
    let input_document = serde_json::to_string(&request).map_err(EmbeddingError::JsonError)?;

    let model_response = client
        .invoke_model()
        .model_id(model)
        .content_type("application/json")
        .accept("application/json")
        .body(Blob::new(input_document))
        .send()
        .await;

    let response = model_response
        .map_err(|sdk_error| AwsSdkInvokeModelError(sdk_error).into())
        .map_err(|e: EmbeddingError| e)?;

    let response_str = String::from_utf8(response.body.into_inner())
        .map_err(|e| EmbeddingError::ResponseError(e.to_string()))?;

    let result: EmbeddingResponse =
        serde_json::from_str(&response_str).map_err(EmbeddingError::JsonError)?;

    Ok(result)
}
