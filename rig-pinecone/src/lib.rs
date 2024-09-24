use std::{collections::BTreeMap, fmt::format};

use pinecone_sdk::{models::{Vector, Metadata}, pinecone::data::Index};
use prost_types::Struct;
use rig::{
    embeddings::{DocumentEmbeddings, Embedding},
    vector_store::VectorStore,
};

pub struct PineconeVectorStore {
    index: Index,
}

impl PineconeVectorStore {
    pub async fn new(index: Index) -> Self {
        PineconeVectorStore { index }
    }
}

impl VectorStore for PineconeVectorStore {
    type Q;

    async fn add_documents(
        &mut self,
        documents: Vec<rig::embeddings::DocumentEmbeddings>,
    ) -> Result<(), rig::vector_store::VectorStoreError> {
        documents.iter().for_each(
            |DocumentEmbeddings {
                 id,
                 document,
                 embeddings,
             }| {
                embeddings.clone().into_iter().enumerate().map(
                    |(
                        i,
                        Embedding {
                            document: embedding_document,
                            vec,
                        },
                    )| {
                        let mut fields = BTreeMap::new();
                        fields.insert("document_id".to_string(), id.to_string());
                        fields.insert("document".to_string(), serde_json::to_string(document)?);
                        fields.insert("embedding_document".to_string(), embedding_document);
                        
                        Vector {
                            id: format!("{}-{i}", id),
                            values: vec.into_iter().map(|float_val| float_val as f32).collect(),
                            metadata: Some(Metadata { fields }),
                            sparse_values: None,
                        }
                    },
                );
            },
        );

        todo!()
    }

    async fn get_document_embeddings(
        &self,
        id: &str,
    ) -> Result<Option<rig::embeddings::DocumentEmbeddings>, rig::vector_store::VectorStoreError>
    {
        todo!()
    }

    async fn get_document<T: for<'a> Deserialize<'a>>(
        &self,
        id: &str,
    ) -> Result<Option<T>, rig::vector_store::VectorStoreError> {
        todo!()
    }

    async fn get_document_by_query(
        &self,
        query: Self::Q,
    ) -> Result<Option<rig::embeddings::DocumentEmbeddings>, rig::vector_store::VectorStoreError>
    {
        todo!()
    }
}
