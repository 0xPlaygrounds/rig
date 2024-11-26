use rig::embeddings::{DocumentEmbeddings, Embedding, EmbeddingModel};
use rig::vector_store::{VectorStore, VectorStoreError, VectorStoreIndex};
use rusqlite::OptionalExtension;
use serde::Deserialize;
use std::marker::PhantomData;
use tokio_rusqlite::Connection;
use tracing::{debug, info};
use zerocopy::IntoBytes;

#[derive(Debug)]
pub enum SqliteError {
    DatabaseError(Box<dyn std::error::Error + Send + Sync>),
    SerializationError(Box<dyn std::error::Error + Send + Sync>),
}

#[derive(Clone)]
pub struct SqliteVectorStore<E: EmbeddingModel> {
    conn: Connection,
    _phantom: PhantomData<E>,
}

impl<E: EmbeddingModel> SqliteVectorStore<E> {
    pub async fn new(conn: Connection, embedding_model: &E) -> Result<Self, VectorStoreError> {
        // Run migrations or create tables if they don't exist
        let dims = embedding_model.ndims();
        conn.call(move |conn| {
            conn.execute_batch(&format!(
                "BEGIN;
                -- Document tables
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id TEXT UNIQUE NOT NULL,
                    document TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_document_id ON documents(document_id);
                CREATE VIRTUAL TABLE IF NOT EXISTS embeddings USING vec0(embedding float[{}]);

                COMMIT;",
                dims
            ))
            .map_err(tokio_rusqlite::Error::from)
        })
        .await
        .map_err(|e| VectorStoreError::DatastoreError(Box::new(e)))?;

        Ok(Self {
            conn,
            _phantom: PhantomData,
        })
    }

    fn serialize_embedding(embedding: &Embedding) -> Vec<f32> {
        embedding.vec.iter().map(|x| *x as f32).collect()
    }

    /// Create a new `SqliteVectorIndex` from an existing `SqliteVectorStore`.
    pub async fn index(&self, model: E) -> Result<SqliteVectorIndex<E>, VectorStoreError> {
        Ok(SqliteVectorIndex::new(model, self.clone()))
    }
}

impl<E: EmbeddingModel> VectorStore for SqliteVectorStore<E> {
    type Q = String;

    async fn add_documents(
        &mut self,
        documents: Vec<DocumentEmbeddings>,
    ) -> Result<(), VectorStoreError> {
        info!("Adding {} documents to store", documents.len());
        self.conn
            .call(|conn| {
                let tx = conn.transaction().map_err(tokio_rusqlite::Error::from)?;

                for doc in documents {
                    debug!("Storing document with id {}", doc.id);
                    // Store document and get auto-incremented ID
                    tx.execute(
                        "INSERT OR REPLACE INTO documents (document_id, document) VALUES (?1, ?2)",
                        [&doc.id, &doc.document.to_string()],
                    )
                    .map_err(tokio_rusqlite::Error::from)?;

                    let document_id = tx.last_insert_rowid();

                    // Store embeddings
                    let mut stmt = tx
                        .prepare("INSERT INTO embeddings (rowid, embedding) VALUES (?1, ?2)")
                        .map_err(tokio_rusqlite::Error::from)?;

                    debug!(
                        "Storing {} embeddings for document {}",
                        doc.embeddings.len(),
                        doc.id
                    );
                    for embedding in doc.embeddings {
                        let vec = Self::serialize_embedding(&embedding);
                        let blob = rusqlite::types::Value::Blob(vec.as_bytes().to_vec());
                        stmt.execute(rusqlite::params![document_id, blob])
                            .map_err(tokio_rusqlite::Error::from)?;
                    }
                }

                tx.commit().map_err(tokio_rusqlite::Error::from)?;
                Ok(())
            })
            .await
            .map_err(|e| VectorStoreError::DatastoreError(Box::new(e)))?;

        Ok(())
    }

    async fn get_document<T: for<'a> Deserialize<'a>>(
        &self,
        id: &str,
    ) -> Result<Option<T>, VectorStoreError> {
        debug!("Fetching document with id {}", id);
        let id_clone = id.to_string();
        let doc_str = self
            .conn
            .call(move |conn| {
                conn.query_row(
                    "SELECT document FROM documents WHERE document_id = ?1",
                    rusqlite::params![id_clone],
                    |row| row.get::<_, String>(0),
                )
                .optional()
                .map_err(tokio_rusqlite::Error::from)
            })
            .await
            .map_err(|e| VectorStoreError::DatastoreError(Box::new(e)))?;

        match doc_str {
            Some(doc_str) => {
                let doc: T = serde_json::from_str(&doc_str)
                    .map_err(|e| VectorStoreError::DatastoreError(Box::new(e)))?;
                Ok(Some(doc))
            }
            None => {
                debug!("No document found with id {}", id);
                Ok(None)
            }
        }
    }

    async fn get_document_embeddings(
        &self,
        id: &str,
    ) -> Result<Option<DocumentEmbeddings>, VectorStoreError> {
        debug!("Fetching embeddings for document {}", id);
        let id_clone = id.to_string();
        let result = self
            .conn
            .call(move |conn| {
                let mut stmt = conn.prepare(
                    "SELECT e.embedding, d.document 
                     FROM embeddings e
                     JOIN documents d ON e.rowid = d.id
                     WHERE d.document_id = ?1",
                )?;

                let result = stmt
                    .query_map(rusqlite::params![id_clone], |row| {
                        let bytes: Vec<u8> = row.get(0)?;
                        let doc_str: String = row.get(1)?;
                        let doc: serde_json::Value =
                            serde_json::from_str(&doc_str).map_err(|e| {
                                rusqlite::Error::FromSqlConversionFailure(
                                    0,
                                    rusqlite::types::Type::Text,
                                    Box::new(e),
                                )
                            })?;
                        let vec = bytes
                            .chunks(4)
                            .map(|chunk| {
                                f32::from_le_bytes(
                                    chunk
                                        .try_into()
                                        .expect("Invalid chunk length - must be 4 bytes"),
                                ) as f64
                            })
                            .collect();
                        Ok((
                            rig::embeddings::Embedding {
                                vec,
                                document: "".to_string(),
                            },
                            doc,
                        ))
                    })?
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(result)
            })
            .await
            .map_err(|e| VectorStoreError::DatastoreError(Box::new(e)))?;

        if let Some((_, doc)) = result.first() {
            let embeddings: Vec<Embedding> = result.iter().map(|(e, _)| e.clone()).collect();
            debug!("Found {} embeddings for document {}", embeddings.len(), id);
            Ok(Some(DocumentEmbeddings {
                id: id.to_string(),
                document: doc.clone(),
                embeddings,
            }))
        } else {
            debug!("No embeddings found for document {}", id);
            Ok(None)
        }
    }

    async fn get_document_by_query(
        &self,
        query: Self::Q,
    ) -> Result<Option<DocumentEmbeddings>, VectorStoreError> {
        debug!("Searching for document matching query");
        let result = self
            .conn
            .call(move |conn| {
                let mut stmt = conn.prepare(
                    "SELECT d.document_id, e.distance 
                     FROM embeddings e
                     JOIN documents d ON e.rowid = d.id
                     WHERE e.embedding MATCH ?1  AND k = ?2
                     ORDER BY e.distance",
                )?;

                let result = stmt
                    .query_row(rusqlite::params![query.as_bytes(), 1], |row| {
                        Ok((row.get::<_, String>(0)?, row.get::<_, f64>(1)?))
                    })
                    .optional()?;
                Ok(result)
            })
            .await
            .map_err(|e| VectorStoreError::DatastoreError(Box::new(e)))?;

        match result {
            Some((id, distance)) => {
                debug!("Found matching document {} with distance {}", id, distance);
                self.get_document_embeddings(&id).await
            }
            None => {
                debug!("No matching documents found");
                Ok(None)
            }
        }
    }
}

pub struct SqliteVectorIndex<E: EmbeddingModel> {
    store: SqliteVectorStore<E>,
    embedding_model: E,
}

impl<E: EmbeddingModel> SqliteVectorIndex<E> {
    pub fn new(embedding_model: E, store: SqliteVectorStore<E>) -> Self {
        Self {
            store,
            embedding_model,
        }
    }
}

impl<E: EmbeddingModel + std::marker::Sync> VectorStoreIndex for SqliteVectorIndex<E> {
    async fn top_n<T: for<'a> Deserialize<'a>>(
        &self,
        query: &str,
        n: usize,
    ) -> Result<Vec<(f64, String, T)>, VectorStoreError> {
        debug!("Finding top {} matches for query", n);
        let embedding = self.embedding_model.embed_document(query).await?;
        let query_vec = SqliteVectorStore::<E>::serialize_embedding(&embedding);

        let rows = self
            .store
            .conn
            .call(move |conn| {
                let mut stmt = conn.prepare(
                    "SELECT d.document_id, d.document, e.distance 
                    FROM embeddings e
                    JOIN documents d ON e.rowid = d.id
                    WHERE e.embedding MATCH ?1 AND k = ?2
                    ORDER BY e.distance",
                )?;

                let rows = stmt
                    .query_map(rusqlite::params![query_vec.as_bytes().to_vec(), n], |row| {
                        Ok((
                            row.get::<_, String>(0)?,
                            row.get::<_, String>(1)?,
                            row.get::<_, f64>(2)?,
                        ))
                    })?
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(rows)
            })
            .await
            .map_err(|e| VectorStoreError::DatastoreError(Box::new(e)))?;

        debug!("Found {} potential matches", rows.len());
        let mut top_n = Vec::new();
        for (id, doc_str, distance) in rows {
            match serde_json::from_str::<T>(&doc_str) {
                Ok(doc) => {
                    top_n.push((distance, id, doc));
                }
                Err(e) => {
                    debug!("Failed to deserialize document {}: {}", id, e);
                    continue;
                }
            }
        }

        debug!("Returning {} matches", top_n.len());
        Ok(top_n)
    }

    async fn top_n_ids(
        &self,
        query: &str,
        n: usize,
    ) -> Result<Vec<(f64, String)>, VectorStoreError> {
        debug!("Finding top {} document IDs for query", n);
        let embedding = self.embedding_model.embed_document(query).await?;
        let query_vec = SqliteVectorStore::<E>::serialize_embedding(&embedding);

        let results = self
            .store
            .conn
            .call(move |conn| {
                let mut stmt = conn.prepare(
                    "SELECT d.document_id, e.distance 
                     FROM embeddings e
                     JOIN documents d ON e.rowid = d.id
                     WHERE e.embedding MATCH ?1 AND k = ?2
                     ORDER BY e.distance",
                )?;

                let results = stmt
                    .query_map(
                        rusqlite::params![
                            query_vec
                                .iter()
                                .flat_map(|x| x.to_le_bytes())
                                .collect::<Vec<u8>>(),
                            n
                        ],
                        |row| Ok((row.get::<_, f64>(1)?, row.get::<_, String>(0)?)),
                    )?
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(results)
            })
            .await
            .map_err(|e| VectorStoreError::DatastoreError(Box::new(e)))?;

        debug!("Found {} matching document IDs", results.len());
        Ok(results)
    }
}
