use document::{DocumentRecord, DocumentRecords};
use embedding::{EmbeddingRecord, EmbeddingRecordsBatch};
use rig::embeddings::{DocumentEmbeddings, Embedding};

pub mod document;
pub mod embedding;

/// Merge an `DocumentRecords` object with an `EmbeddingRecordsBatch` object.
/// These objects contain document and embedding data, respectively, read from LanceDB.
/// For each document in `DocumentRecords` find the embeddings from `EmbeddingRecordsBatch` that correspond to that document,
/// using the document_id as reference.
pub fn merge(
    documents: &DocumentRecords,
    embeddings: &EmbeddingRecordsBatch,
) -> Result<Vec<DocumentEmbeddings>, serde_json::Error> {
    documents
        .as_iter()
        .map(|DocumentRecord { id, document }| {
            let emebedding_records = embeddings.get_by_id(id);

            Ok(DocumentEmbeddings {
                id: id.to_string(),
                document: serde_json::from_str(document)?,
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
