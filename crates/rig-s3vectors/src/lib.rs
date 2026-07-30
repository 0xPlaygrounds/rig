//! AWS S3Vectors vector store integration for Rig.
//!
//! This crate provides [`S3VectorsVectorStore`], a Rig vector store backed by
//! AWS S3Vectors indexes. It uses the AWS SDK client supplied by the caller and
//! maps Rig search filters to S3Vectors filter documents through
//! [`S3SearchFilter`]. Queries arrive pre-embedded via [`VectorSearchRequest`];
//! the store never embeds text itself.
//!
//! The root `rig` facade re-exports this crate as `rig::s3vectors` when the
//! `s3vectors` feature is enabled.

#[macro_use]
mod document;

use aws_sdk_s3vectors::{
    Client,
    types::{PutInputVector, VectorData},
};
use aws_smithy_types::Document;
use rig_core::{
    OneOrMany,
    embeddings::Embedding,
    vector_store::{
        SearchHit, StoreRecord, VectorStoreError,
        request::{Filter as CoreFilter, VectorSearchRequest},
    },
};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use serde_json::Value;
use std::collections::HashMap;

#[derive(Debug, Serialize, Deserialize)]
pub struct CreateRecord {
    document: serde_json::Value,
    embedded_text: String,
}

// NOTE: Cannot be used with `Filter<serde_json::Value>` requests due to
// aws_smithy_types::Document not impl'ing Serialize or Deserialize
/// Converts a canonical JSON filter operand into a Smithy [`Document`].
///
/// Total: every `serde_json::Value` shape has a `Document` counterpart.
fn json_to_document(value: serde_json::Value) -> aws_smithy_types::Document {
    use aws_smithy_types::{Document, Number};
    match value {
        serde_json::Value::Null => Document::Null,
        serde_json::Value::Bool(b) => Document::Bool(b),
        serde_json::Value::Number(n) => Document::Number(if let Some(u) = n.as_u64() {
            Number::PosInt(u)
        } else if let Some(i) = n.as_i64() {
            Number::NegInt(i)
        } else {
            Number::Float(n.as_f64().unwrap_or_default())
        }),
        serde_json::Value::String(s) => Document::String(s),
        serde_json::Value::Array(xs) => {
            Document::Array(xs.into_iter().map(json_to_document).collect())
        }
        serde_json::Value::Object(map) => Document::Object(
            map.into_iter()
                .map(|(k, v)| (k, json_to_document(v)))
                .collect(),
        ),
    }
}

#[derive(Clone, Debug)]
pub struct S3SearchFilter(aws_smithy_types::Document);

impl S3SearchFilter {
    /// Translates the canonical [`CoreFilter`] into this backend's filter type.
    pub fn from_filter(filter: CoreFilter<serde_json::Value>) -> Self {
        match filter {
            CoreFilter::Eq(key, value) => Self::eq(key, json_to_document(value)),
            CoreFilter::Gt(key, value) => Self::gt(key, json_to_document(value)),
            CoreFilter::Lt(key, value) => Self::lt(key, json_to_document(value)),
            CoreFilter::And(lhs, rhs) => Self::from_filter(*lhs).and(Self::from_filter(*rhs)),
            CoreFilter::Or(lhs, rhs) => Self::from_filter(*lhs).or(Self::from_filter(*rhs)),
        }
    }

    #[allow(clippy::should_implement_trait)]
    pub fn eq(key: impl AsRef<str>, value: aws_smithy_types::Document) -> Self {
        let key = key.as_ref().to_owned();
        Self(document!({ key: { "$eq": value } }))
    }

    pub fn gt(key: impl AsRef<str>, value: aws_smithy_types::Document) -> Self {
        let key = key.as_ref().to_owned();
        Self(document!({ key: { "$gt": value } }))
    }

    pub fn lt(key: impl AsRef<str>, value: aws_smithy_types::Document) -> Self {
        let key = key.as_ref().to_owned();
        Self(document!({ key: { "$lt": value } }))
    }

    pub fn and(self, rhs: Self) -> Self {
        Self(document!({ "$and": [ self.0, rhs.0 ]}))
    }

    pub fn or(self, rhs: Self) -> Self {
        Self(document!({ "$or": [ self.0, rhs.0 ]}))
    }
}

impl S3SearchFilter {
    pub fn inner(&self) -> &aws_smithy_types::Document {
        &self.0
    }

    pub fn into_inner(self) -> aws_smithy_types::Document {
        self.0
    }

    pub fn gte(key: String, value: aws_smithy_types::Document) -> Self {
        Self(document!({ key: { "$gte": value } }))
    }

    pub fn lte(key: String, value: aws_smithy_types::Document) -> Self {
        Self(document!({ key: { "$lte": value } }))
    }

    pub fn exists(key: String) -> Self {
        Self(document!({ "$exists": { key: true } }))
    }

    #[allow(clippy::should_implement_trait)]
    pub fn not(self) -> Self {
        Self(document!({ "$not": self.0 }))
    }
}

/// A vector store backed by an AWS S3Vectors index.
///
/// Queries and records arrive pre-embedded, so the store holds no embedding
/// model.
pub struct S3VectorsVectorStore {
    client: Client,
    bucket_name: String,
    index_name: String,
}

impl S3VectorsVectorStore {
    pub fn new(client: aws_sdk_s3vectors::Client, bucket_name: &str, index_name: &str) -> Self {
        Self {
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

    /// Insert precomputed records into the S3Vectors index.
    ///
    /// Each embedding of a record becomes one stored vector whose metadata is
    /// a [`CreateRecord`] wrapping the record payload and embedded text. A
    /// record with a single embedding is keyed by [`StoreRecord::id`];
    /// additional embeddings get `"{id}#{n}"` keys so they don't overwrite
    /// each other.
    pub async fn insert(&self, records: Vec<StoreRecord>) -> Result<(), VectorStoreError> {
        let docs: Vec<PutInputVector> = records
            .into_iter()
            .map(|record| {
                let single = record.embeddings.len() == 1;

                record
                    .embeddings
                    .into_iter()
                    .enumerate()
                    .map(|(n, embedding)| {
                        let key = if single {
                            record.id.clone()
                        } else {
                            format!("{id}#{n}", id = record.id)
                        };

                        let document = CreateRecord {
                            document: record.payload.clone(),
                            embedded_text: embedding.document,
                        };
                        let document =
                            serde_json::to_value(&document).map_err(VectorStoreError::JsonError)?;
                        let document = json_value_to_document(&document);
                        let vec = embedding.vec.into_iter().map(|item| item as f32).collect();
                        PutInputVector::builder()
                            .metadata(document)
                            .data(VectorData::Float32(vec))
                            .key(key)
                            .build()
                            .map_err(|x| {
                                VectorStoreError::DatastoreError(
                                    format!("Couldn't build vector input: {x}").into(),
                                )
                            })
                    })
                    .collect()
            })
            .collect::<Result<Vec<Vec<PutInputVector>>, VectorStoreError>>()
            .map_err(|x| {
                VectorStoreError::DatastoreError(
                    format!("Could not build vector store data: {x}").into(),
                )
            })?
            .into_iter()
            .flatten()
            .collect();

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

    /// Serializes each document and inserts it. Sugar over [`Self::insert`].
    pub async fn insert_as<T: Serialize>(
        &self,
        docs: Vec<(String, T, OneOrMany<Embedding>)>,
    ) -> Result<(), VectorStoreError> {
        let records = docs
            .into_iter()
            .map(|(id, doc, embeddings)| StoreRecord::new(id, &doc, embeddings))
            .collect::<Result<Vec<_>, _>>()?;
        self.insert(records).await
    }

    /// Returns the top N most similar documents for a pre-embedded query.
    ///
    /// The [`SearchHit::payload`] is the stored metadata document (a
    /// [`CreateRecord`]: the original payload under `document` plus the
    /// `embedded_text`).
    pub async fn top_n(
        &self,
        req: VectorSearchRequest<S3SearchFilter>,
    ) -> Result<Vec<SearchHit>, VectorStoreError> {
        let query = self.query_vectors(&req, true).await?;

        let res = query
            .vectors
            .into_iter()
            .map(|x| {
                let Some(distance) = Self::qualifying_distance(x.distance, req.threshold())? else {
                    return Ok(None);
                };

                let metadata_document = x.metadata.ok_or_else(|| {
                    VectorStoreError::DatastoreError(Box::new(std::io::Error::other(
                        "S3Vectors response missing metadata",
                    )))
                })?;
                let payload = document_to_json_value(&metadata_document);

                Ok(Some(SearchHit {
                    id: x.key,
                    score: distance,
                    payload,
                }))
            })
            .collect::<Result<Vec<_>, VectorStoreError>>()?
            .into_iter()
            .flatten()
            .collect();

        Ok(res)
    }

    /// Returns the top N most similar document keys as `(score, key)` tuples.
    pub async fn top_n_ids(
        &self,
        req: VectorSearchRequest<S3SearchFilter>,
    ) -> Result<Vec<(f64, String)>, VectorStoreError> {
        let query = self.query_vectors(&req, false).await?;

        let res = query
            .vectors
            .into_iter()
            .map(|x| {
                let Some(distance) = Self::qualifying_distance(x.distance, req.threshold())? else {
                    return Ok(None);
                };

                Ok(Some((distance, x.key)))
            })
            .collect::<Result<Vec<_>, VectorStoreError>>()?
            .into_iter()
            .flatten()
            .collect();

        Ok(res)
    }

    /// Returns the top N most similar documents deserialized into `T` as
    /// `(score, key, metadata)` tuples. Sugar over [`Self::top_n`].
    pub async fn top_n_as<T: DeserializeOwned>(
        &self,
        req: VectorSearchRequest<S3SearchFilter>,
    ) -> Result<Vec<(f64, String, T)>, VectorStoreError> {
        self.top_n(req)
            .await?
            .into_iter()
            .map(|hit| {
                let doc = serde_json::from_value(hit.payload)?;
                Ok((hit.score, hit.id, doc))
            })
            .collect()
    }

    /// Runs a `query_vectors` call for the pre-embedded request.
    async fn query_vectors(
        &self,
        req: &VectorSearchRequest<S3SearchFilter>,
        return_metadata: bool,
    ) -> Result<aws_sdk_s3vectors::operation::query_vectors::QueryVectorsOutput, VectorStoreError>
    {
        if req.samples() > i32::MAX as u64 {
            return Err(VectorStoreError::DatastoreError(format!("The number of samples to return with the `rig` AWS S3Vectors integration cannot be higher than {}", i32::MAX).into()));
        }

        let embedding = req
            .query()
            .first()
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
            .return_metadata(return_metadata)
            .vector_bucket_name(self.bucket_name())
            .index_name(self.index_name());

        if let Some(filter) = req.filter() {
            query_builder = query_builder.filter(filter.inner().clone())
        }

        query_builder
            .send()
            .await
            .map_err(|e| VectorStoreError::DatastoreError(Box::new(e)))
    }

    /// Extracts the distance from a response vector, applying the threshold.
    fn qualifying_distance(
        distance: Option<f32>,
        threshold: Option<f64>,
    ) -> Result<Option<f64>, VectorStoreError> {
        let distance = distance.ok_or_else(|| {
            VectorStoreError::DatastoreError(Box::new(std::io::Error::other(
                "S3Vectors response missing distance",
            )))
        })? as f64;

        if threshold.is_some_and(|threshold| distance < threshold) {
            return Ok(None);
        }

        Ok(Some(distance))
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
