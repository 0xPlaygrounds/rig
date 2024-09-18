use document::{DocumentRecord, DocumentRecords};
use embedding::{EmbeddingRecord, EmbeddingRecordsBatch};
use rig::{
    embeddings::{DocumentEmbeddings, Embedding},
    vector_store::VectorStoreError,
};

use crate::serde_to_rig_error;

pub mod document;
pub mod embedding;

pub fn merge(
    documents: DocumentRecords,
    embeddings: EmbeddingRecordsBatch,
) -> Result<Vec<DocumentEmbeddings>, VectorStoreError> {
    documents
        .as_iter()
        .map(|DocumentRecord { id, document }| {
            let emebedding_records = embeddings.get_by_id(id);

            Ok::<_, VectorStoreError>(DocumentEmbeddings {
                id: id.to_string(),
                document: serde_json::from_str(document).map_err(serde_to_rig_error)?,
                embeddings: match emebedding_records {
                    Some(records) => records
                        .as_iter()
                        .map(
                            |EmbeddingRecord {
                                 content, embedding, ..
                             }| Embedding {
                                document: content.to_string(),
                                vec: embedding.to_vec(),
                            },
                        )
                        .collect::<Vec<_>>(),
                    None => vec![],
                },
            })
        })
        .collect::<Result<Vec<_>, _>>()
}
