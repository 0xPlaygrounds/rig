use pinecone_sdk::pinecone::{data::Index};
use rig::vector_store::VectorStore;

pub struct PineconeVectorStore {
    index: Index
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
        todo!()
    }

    async fn get_document_embeddings(
        &self,
        id: &str,
    ) ->Result<Option<rig::embeddings::DocumentEmbeddings>, rig::vector_store::VectorStoreError> {
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
    ) -> Result<Option<rig::embeddings::DocumentEmbeddings>, rig::vector_store::VectorStoreError> {
        todo!()
    }
}