// Tests assert on filter shapes, where a failed conversion should panic loudly
// rather than be handled; matches the other vector-store crates (e.g. rig-lancedb).
#![cfg_attr(test, allow(clippy::expect_used))]
//! AWS S3Vectors vector store integration for Rig.
//!
//! This crate provides [`S3VectorsVectorStore`], a Rig vector store backed by
//! AWS S3Vectors indexes. It uses the AWS SDK client supplied by the caller and
//! maps Rig search filters to S3Vectors filter documents through
//! [`S3SearchFilter`].
//!
//! The root `rig` facade re-exports this crate as `rig::s3vectors` when the
//! `s3vectors` feature is enabled.

use aws_sdk_s3vectors::{
    Client,
    types::{PutInputVector, VectorData},
};
use aws_smithy_types::Document;
use rig_core::{
    embeddings::EmbeddingModel,
    vector_store::{
        InsertDocuments, VectorStoreError, VectorStoreIndex,
        request::{DynamicSearchFilter, Filter, FilterError, SearchFilter, VectorSearchRequest},
    },
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use uuid::Uuid;

#[derive(Debug, Serialize, Deserialize)]
pub struct CreateRecord {
    document: serde_json::Value,
    embedded_text: String,
}

/// S3Vectors filter backed by the AWS SDK's native document type.
#[derive(Clone, Debug)]
pub struct S3SearchFilter(aws_smithy_types::Document);

impl SearchFilter for S3SearchFilter {
    type Value = aws_smithy_types::Document;

    fn eq(key: impl AsRef<str>, value: Self::Value) -> Self {
        Self(document_comparison(key, "$eq", value))
    }

    fn gt(key: impl AsRef<str>, value: Self::Value) -> Self {
        Self(document_comparison(key, "$gt", value))
    }

    fn lt(key: impl AsRef<str>, value: Self::Value) -> Self {
        Self(document_comparison(key, "$lt", value))
    }

    fn and(self, rhs: Self) -> Self {
        Self(document_object([(
            "$and",
            Document::Array(vec![self.0, rhs.0]),
        )]))
    }

    fn or(self, rhs: Self) -> Self {
        Self(document_object([(
            "$or",
            Document::Array(vec![self.0, rhs.0]),
        )]))
    }
}

/// Builds a `Document::Object` from the given entries.
fn document_object<K>(entries: impl IntoIterator<Item = (K, Document)>) -> Document
where
    K: Into<String>,
{
    Document::Object(
        entries
            .into_iter()
            .map(|(key, value)| (key.into(), value))
            .collect(),
    )
}

/// Builds the `{ key: { op: value } }` shape S3Vectors uses for comparison
/// operators such as `$eq` and `$gt`.
fn document_comparison(key: impl AsRef<str>, op: &str, value: Document) -> Document {
    document_object([(key.as_ref(), document_object([(op, value)]))])
}

impl DynamicSearchFilter for S3SearchFilter {
    fn from_dynamic_filter(filter: Filter<serde_json::Value>) -> Result<Self, FilterError> {
        Ok(filter.interpret_with(|value| json_value_to_document(&value)))
    }
}

impl S3SearchFilter {
    pub fn inner(&self) -> &aws_smithy_types::Document {
        &self.0
    }

    pub fn into_inner(self) -> aws_smithy_types::Document {
        self.0
    }

    pub fn gte(key: String, value: <Self as SearchFilter>::Value) -> Self {
        Self(document_comparison(key, "$gte", value))
    }

    pub fn lte(key: String, value: <Self as SearchFilter>::Value) -> Self {
        Self(document_comparison(key, "$lte", value))
    }

    pub fn exists(key: String) -> Self {
        Self(document_object([(
            "$exists",
            document_object([(key, Document::Bool(true))]),
        )]))
    }

    #[allow(clippy::should_implement_trait)]
    pub fn not(self) -> Self {
        Self(document_object([("$not", self.0)]))
    }
}

pub struct S3VectorsVectorStore<M> {
    embedding_model: M,
    client: Client,
    bucket_name: String,
    index_name: String,
}

impl<M> S3VectorsVectorStore<M>
where
    M: EmbeddingModel,
{
    pub fn new(
        embedding_model: M,
        client: aws_sdk_s3vectors::Client,
        bucket_name: &str,
        index_name: &str,
    ) -> Self {
        Self {
            embedding_model,
            client,
            bucket_name: bucket_name.to_string(),
            index_name: index_name.to_string(),
        }
    }

    pub fn bucket_name(&self) -> &str {
        &self.bucket_name
    }

    pub fn set_bucket_name(&mut self, bucket_name: &str) {
        self.bucket_name = bucket_name.to_string();
    }

    pub fn index_name(&self) -> &str {
        &self.index_name
    }

    pub fn set_index_name(&mut self, index_name: &str) {
        self.index_name = index_name.to_string();
    }

    pub fn client(&self) -> &Client {
        &self.client
    }

    /// Validates the sample count, embeds the query, and runs the S3Vectors
    /// query, returning the `(distance, vector)` pairs passing the threshold.
    async fn run_query(
        &self,
        req: &VectorSearchRequest<S3SearchFilter>,
        return_metadata: bool,
    ) -> Result<Vec<(f64, aws_sdk_s3vectors::types::QueryOutputVector)>, VectorStoreError> {
        if req.samples() > i32::MAX as u64 {
            return Err(VectorStoreError::DatastoreError(format!("The number of samples to return with the `rig` AWS S3Vectors integration cannot be higher than {}", i32::MAX).into()));
        }

        let embedding = self
            .embedding_model
            .embed_text(req.query())
            .await?
            .vec
            .into_iter()
            .map(|x| x as f32)
            .collect();

        let mut query_builder = self
            .client
            .query_vectors()
            .query_vector(VectorData::Float32(embedding))
            .top_k(req.samples() as i32)
            .return_distance(true)
            .vector_bucket_name(self.bucket_name())
            .index_name(self.index_name());

        if return_metadata {
            query_builder = query_builder.return_metadata(true);
        }

        if let Some(filter) = req.filter() {
            query_builder = query_builder.filter(filter.inner().clone())
        }

        let query = query_builder
            .send()
            .await
            .map_err(VectorStoreError::datastore)?;

        Ok(query
            .vectors
            .into_iter()
            .map(|x| {
                let distance = x.distance.ok_or_else(|| {
                    VectorStoreError::DatastoreError("S3Vectors response missing distance".into())
                })? as f64;

                Ok((distance, x))
            })
            .collect::<Result<Vec<_>, VectorStoreError>>()?
            .into_iter()
            .filter(|(distance, _)| {
                !req.threshold()
                    .is_some_and(|threshold| *distance < threshold)
            })
            .collect())
    }
}

impl<M> InsertDocuments for S3VectorsVectorStore<M>
where
    M: EmbeddingModel,
{
    async fn insert_documents<Doc: serde::Serialize + rig_core::Embed + Send>(
        &self,
        documents: Vec<(Doc, Vec<rig_core::embeddings::Embedding>)>,
    ) -> Result<(), rig_core::vector_store::VectorStoreError> {
        let docs: Vec<PutInputVector> =
            rig_core::vector_store::flatten_embedded(documents, |json_value, y| {
                let document = CreateRecord {
                    document: json_value.clone(),
                    embedded_text: y.document,
                };
                let document =
                    serde_json::to_value(&document).map_err(VectorStoreError::JsonError)?;
                let document = json_value_to_document(&document);
                let vec = y.vec.into_iter().map(|item| item as f32).collect();
                PutInputVector::builder()
                    .metadata(document.clone())
                    .data(VectorData::Float32(vec))
                    .key(Uuid::new_v4())
                    .build()
                    .map_err(|x| {
                        VectorStoreError::DatastoreError(
                            format!("Couldn't build vector input: {x}").into(),
                        )
                    })
            })
            .map_err(|x| {
                VectorStoreError::DatastoreError(
                    format!("Could not build vector store data: {x}").into(),
                )
            })?;

        self.client
            .put_vectors()
            .vector_bucket_name(self.bucket_name())
            .set_vectors(Some(docs))
            .set_index_name(Some(self.index_name.clone()))
            .send()
            .await
            .map_err(|x| {
                VectorStoreError::DatastoreError(
                    format!("Error while submitting document insertion request: {x}").into(),
                )
            })?;

        Ok(())
    }
}

fn json_value_to_document(value: &Value) -> Document {
    match value {
        Value::Null => Document::Null,
        Value::Bool(b) => Document::Bool(*b),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Document::Number(aws_smithy_types::Number::NegInt(i))
            } else if let Some(u) = n.as_u64() {
                Document::Number(aws_smithy_types::Number::PosInt(u))
            } else if let Some(f) = n.as_f64() {
                Document::Number(aws_smithy_types::Number::Float(f))
            } else {
                Document::Null // fallback, should never happen
            }
        }
        Value::String(s) => Document::String(s.clone()),
        Value::Array(arr) => Document::Array(arr.iter().map(json_value_to_document).collect()),
        Value::Object(obj) => Document::Object(
            obj.iter()
                .map(|(k, v)| (k.clone(), json_value_to_document(v)))
                .collect::<HashMap<_, _>>(),
        ),
    }
}

fn document_to_json_value(value: &Document) -> Value {
    match value {
        Document::Null => Value::Null,
        Document::Bool(b) => Value::Bool(*b),
        Document::Number(n) => match n {
            aws_smithy_types::Number::Float(f) => serde_json::Number::from_f64(*f)
                .map(Value::Number)
                .unwrap_or_else(|| Value::String(f.to_string())),
            aws_smithy_types::Number::NegInt(i) => {
                serde_json::Value::Number(serde_json::Number::from(*i))
            }
            aws_smithy_types::Number::PosInt(u) => {
                serde_json::Value::Number(serde_json::Number::from(*u))
            }
        },
        Document::String(s) => Value::String(s.clone()),
        Document::Array(arr) => Value::Array(arr.iter().map(document_to_json_value).collect()),
        Document::Object(obj) => {
            let res = obj
                .iter()
                .map(|(k, v)| (k.clone(), document_to_json_value(v)))
                .collect::<serde_json::Map<String, serde_json::Value>>();

            serde_json::Value::Object(res)
        }
    }
}

impl<M> VectorStoreIndex for S3VectorsVectorStore<M>
where
    M: EmbeddingModel,
{
    type Filter = S3SearchFilter;

    async fn top_n<T: for<'a> serde::Deserialize<'a> + Send>(
        &self,
        req: VectorSearchRequest<S3SearchFilter>,
    ) -> Result<Vec<(f64, String, T)>, VectorStoreError> {
        self.run_query(&req, true)
            .await?
            .into_iter()
            .map(|(distance, x)| {
                let metadata_document = x.metadata.ok_or_else(|| {
                    VectorStoreError::DatastoreError("S3Vectors response missing metadata".into())
                })?;
                let val = document_to_json_value(&metadata_document);
                let metadata: T = serde_json::from_value(val)?;

                Ok((distance, x.key, metadata))
            })
            .collect()
    }

    async fn top_n_ids(
        &self,
        req: VectorSearchRequest<S3SearchFilter>,
    ) -> Result<Vec<(f64, String)>, VectorStoreError> {
        Ok(self
            .run_query(&req, false)
            .await?
            .into_iter()
            .map(|(distance, x)| (distance, x.key))
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dynamic_filter_compiles_to_native_aws_documents() {
        let filter = Filter::eq("status", serde_json::json!("ready"))
            .and(Filter::eq("tags", serde_json::json!(["rust", "ai"])))
            .and(Filter::gt("score", serde_json::json!(4.5)));

        let compiled = S3SearchFilter::from_dynamic_filter(filter)
            .expect("JSON values should compile to AWS documents");

        assert_eq!(
            document_to_json_value(compiled.inner()),
            serde_json::json!({
                "$and": [{
                    "$and": [
                        { "status": { "$eq": "ready" } },
                        { "tags": { "$eq": ["rust", "ai"] } }
                    ]
                }, {
                    "score": { "$gt": 4.5 }
                }]
            })
        );
    }

    #[test]
    fn extension_operators_build_the_documented_filter_shapes() {
        let number = |n| Document::Number(aws_smithy_types::Number::PosInt(n));
        let filter = S3SearchFilter::gte("score".into(), number(5))
            .or(S3SearchFilter::lte("score".into(), number(1)))
            .or(S3SearchFilter::exists("status".into()))
            .not();

        assert_eq!(
            document_to_json_value(filter.inner()),
            serde_json::json!({
                "$not": {
                    "$or": [
                        { "$or": [
                            { "score": { "$gte": 5 } },
                            { "score": { "$lte": 1 } }
                        ]},
                        { "$exists": { "status": true } }
                    ]
                }
            })
        );
    }
}
