use std::collections::HashMap;

use aws_sdk_s3vectors::{
    Client,
    types::{PutInputVector, VectorData},
};
use aws_smithy_types::Document;
use rig::{
    embeddings::EmbeddingModel,
    vector_store::{InsertDocuments, VectorStoreError, VectorStoreIndex},
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use uuid::Uuid;

#[derive(Debug, Serialize, Deserialize)]
pub struct CreateRecord {
    document: serde_json::Value,
    embedded_text: String,
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
}

impl<M> InsertDocuments for S3VectorsVectorStore<M>
where
    M: EmbeddingModel,
{
    async fn insert_documents<Doc: serde::Serialize + rig::Embed + Send>(
        &self,
        documents: Vec<(Doc, rig::OneOrMany<rig::embeddings::Embedding>)>,
    ) -> Result<(), rig::vector_store::VectorStoreError> {
        let docs: Vec<PutInputVector> = documents
            .into_iter()
            .map(|x| {
                let json_value = serde_json::to_value(&x.0).map_err(VectorStoreError::JsonError)?;

                x.1.into_iter()
                    .map(|y| {
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
        Document::Number(n) => {
            let res = match n {
                aws_smithy_types::Number::Float(f) => {
                    serde_json::Number::from_f64(f.to_owned()).unwrap()
                }
                aws_smithy_types::Number::NegInt(i) => {
                    serde_json::Number::from_i128(*i as i128).unwrap()
                }
                aws_smithy_types::Number::PosInt(u) => {
                    serde_json::Number::from_u128(*u as u128).unwrap()
                }
            };

            serde_json::Value::Number(res)
        }
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
    async fn top_n<T: for<'a> serde::Deserialize<'a> + Send>(
        &self,
        query: &str,
        n: usize,
    ) -> Result<Vec<(f64, String, T)>, VectorStoreError> {
        let embedding = self
            .embedding_model
            .embed_text(query)
            .await?
            .vec
            .into_iter()
            .map(|x| x as f32)
            .collect();

        let query = self
            .client
            .query_vectors()
            .query_vector(VectorData::Float32(embedding))
            .top_k(n as i32)
            .return_distance(true)
            .return_metadata(true)
            .vector_bucket_name(self.bucket_name())
            .index_name(self.index_name())
            .send()
            .await
            .unwrap();

        let res: Vec<(f64, String, T)> = query
            .vectors
            .into_iter()
            .map(|x| {
                let distance = x.distance.expect("vector distance should always exist") as f64;
                let val =
                    document_to_json_value(&x.metadata.expect("metadata should always exist"));

                let metadata: T = serde_json::from_value(val)
                    .expect("converting JSON from S3Vectors to valid T should always work");

                (distance, x.key, metadata)
            })
            .collect();

        Ok(res)
    }
    async fn top_n_ids(
        &self,
        query: &str,
        n: usize,
    ) -> Result<Vec<(f64, String)>, VectorStoreError> {
        let embedding = self
            .embedding_model
            .embed_text(query)
            .await?
            .vec
            .into_iter()
            .map(|x| x as f32)
            .collect();

        let query = self
            .client
            .query_vectors()
            .query_vector(VectorData::Float32(embedding))
            .top_k(n as i32)
            .return_distance(true)
            .vector_bucket_name(self.bucket_name())
            .index_name(self.index_name())
            .send()
            .await
            .unwrap();

        let res: Vec<(f64, String)> = query
            .vectors
            .into_iter()
            .map(|x| {
                let distance = x.distance.expect("vector distance should always exist") as f64;

                (distance, x.key)
            })
            .collect();

        Ok(res)
    }
}
