use std::collections::BTreeMap;

use pinecone_sdk::models::{Kind, Metadata, Namespace, QueryResponse, Value, Vector};
use pinecone_sdk::pinecone::data::Index;
use pinecone_sdk::pinecone::PineconeClient;
use prost_types::ListValue;
use rig::embeddings::EmbeddingModel;
use rig::vector_store::{VectorStoreError, VectorStoreIndex};
use rig::{embeddings::Embedding, Embed, OneOrMany};
use serde::Serialize;
use serde_json::Value as JsonValue;

pub struct PineconeVectorStore<M> {
    model: M,
    client: PineconeClient,
    index_name: String,
    namespace: Namespace,
}

impl<M> PineconeVectorStore<M>
where
    M: EmbeddingModel,
{
    pub fn new<S, N>(client: PineconeClient, index_name: S, model: M, namespace: N) -> Self
    where
        S: Into<String>,
        N: Into<Namespace>,
    {
        let index_name: String = index_name.into();
        let namespace: Namespace = namespace.into();
        Self {
            client,
            model,
            index_name,
            namespace,
        }
    }

    pub fn update_index_name(&mut self, index_name: &str) {
        self.index_name = index_name.to_string();
    }

    pub fn namespace(&self) -> &Namespace {
        &self.namespace
    }

    pub fn update_namespace(&mut self, namespace: Namespace) {
        self.namespace = namespace;
    }

    pub async fn insert_documents<Doc: Serialize + Embed + Send>(
        &mut self,
        documents: Vec<(Doc, OneOrMany<Embedding>)>,
        namespace: &Namespace,
    ) -> Result<(), VectorStoreError> {
        let vectors: Vec<Vector> = documents
            .into_iter()
            .map(|(doc, embedding)| {
                let metadata = {
                    let json_value: JsonValue = serde_json::to_value(&doc).unwrap();
                    json_to_metadata(&json_value)
                };

                let values = embedding.first().vec.iter().map(|&x| x as f32).collect();

                Vector {
                    id: uuid::Uuid::new_v4().to_string(),
                    values,
                    sparse_values: None,
                    metadata: Some(metadata),
                }
            })
            .collect();

        let mut idx = self
            .client
            .index(&self.index_name)
            .await
            .map_err(|x| VectorStoreError::DatastoreError(x.into()))?;

        idx.upsert(&vectors, namespace)
            .await
            .map_err(|x| VectorStoreError::DatastoreError(x.into()))?;

        Ok(())
    }

    /// Embed query based on `QdrantVectorStore` model and modify the vector in the required format.
    pub async fn generate_query_vector(&self, query: &str) -> Result<Vec<f32>, VectorStoreError> {
        let embedding = self.model.embed_text(query).await?;
        Ok(embedding.vec.iter().map(|&x| x as f32).collect())
    }
}

impl<M> VectorStoreIndex for PineconeVectorStore<M>
where
    M: EmbeddingModel,
{
    async fn top_n<T: for<'a> serde::Deserialize<'a> + Send>(
        &self,
        query: &str,
        n: usize,
    ) -> Result<Vec<(f64, String, T)>, VectorStoreError> {
        let vector = self.generate_query_vector(query).await?;
        let mut index = self
            .client
            .index(&self.index_name)
            .await
            .map_err(|x| VectorStoreError::DatastoreError(x.into()))?;

        let res: QueryResponse = index
            .query_by_value(
                vector,
                None,
                n as u32,
                self.namespace(),
                None,
                Some(true),
                Some(true),
            )
            .await
            .map_err(|x| VectorStoreError::DatastoreError(x.into()))?;

        todo!()
    }

    async fn top_n_ids(
        &self,
        query: &str,
        n: usize,
    ) -> Result<Vec<(f64, String)>, VectorStoreError> {
        todo!()
    }
}

pub fn json_to_metadata(json: &JsonValue) -> Metadata {
    match json {
        JsonValue::Object(map) => {
            let fields = map
                .iter()
                .map(|(k, v)| (k.clone(), json_to_kind_value(v)))
                .collect();
            Metadata { fields }
        }
        _ => {
            // Not a JSON object — return empty metadata or panic based on your needs
            Metadata {
                fields: BTreeMap::new(),
            }
        }
    }
}

fn json_to_kind_value(json: &JsonValue) -> Value {
    let kind = match json {
        JsonValue::Null => Some(Kind::NullValue(0)),
        JsonValue::Bool(b) => Some(Kind::BoolValue(*b)),
        JsonValue::Number(n) => n
            .as_f64()
            .map(Kind::NumberValue)
            .or_else(|| n.as_i64().map(|i| Kind::NumberValue(i as f64)))
            .or_else(|| n.as_u64().map(|u| Kind::NumberValue(u as f64))),
        JsonValue::String(s) => Some(Kind::StringValue(s.clone())),
        JsonValue::Array(arr) => Some(Kind::ListValue(ListValue {
            values: arr.iter().map(json_to_kind_value).collect(),
        })),
        JsonValue::Object(map) => Some(Kind::StructValue(Metadata {
            fields: map
                .iter()
                .map(|(k, v)| (k.clone(), json_to_kind_value(v)))
                .collect(),
        })),
    };

    Value { kind }
}

pub fn metadata_to_json_value(metadata: &Metadata) -> serde_json::Value {
    let mut map = serde_json::Map::new();
    for (k, v) in &metadata.fields {
        map.insert(k.clone(), convert_value_to_json(v));
    }
    serde_json::Value::Object(map)
}

fn convert_value_to_json(value: &Value) -> serde_json::Value {
    match &value.kind {
        Some(Kind::NullValue(_)) => serde_json::Value::Null,
        Some(Kind::BoolValue(b)) => serde_json::Value::Bool(*b),
        Some(Kind::NumberValue(n)) => serde_json::Value::Number(
            serde_json::Number::from_f64(*n).expect("Invalid f64 for JSON number"),
        ),
        Some(Kind::StringValue(s)) => serde_json::Value::String(s.clone()),
        Some(Kind::ListValue(list)) => {
            let arr = list.values.iter().map(convert_value_to_json).collect();
            serde_json::Value::Array(arr)
        }
        Some(Kind::StructValue(struct_val)) => metadata_to_json_value(struct_val),
        None => serde_json::Value::Null,
    }
}
