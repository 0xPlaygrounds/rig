use rig::embeddings::{Embedding, EmbeddingModel};
use rig::vector_store::{VectorStoreError, VectorStoreIndex};
use rig::OneOrMany;
use serde::Deserialize;
use std::marker::PhantomData;
use tokio_rusqlite::Connection;
use tracing::{debug, info};
use zerocopy::IntoBytes;

#[derive(Debug)]
pub enum SqliteError {
    DatabaseError(Box<dyn std::error::Error + Send + Sync>),
    SerializationError(Box<dyn std::error::Error + Send + Sync>),
    InvalidColumnType(String),
}

pub trait ColumnValue: Send + Sync {
    fn to_sql_string(&self) -> String;
    fn column_type(&self) -> &'static str;
}

pub struct Column {
    name: &'static str,
    col_type: &'static str,
    indexed: bool,
}

impl Column {
    pub fn new(name: &'static str, col_type: &'static str) -> Self {
        Self {
            name,
            col_type,
            indexed: false,
        }
    }

    pub fn indexed(mut self) -> Self {
        self.indexed = true;
        self
    }
}

/// Example of a document type that can be used with SqliteVectorStore
/// ```rust
/// use rig::Embed;
/// use serde::Deserialize;
/// use rig_sqlite::{Column, ColumnValue, SqliteVectorStoreTable};
///
/// #[derive(Embed, Clone, Debug, Deserialize)]
/// struct Document {
///     id: String,
///     #[embed]
///     content: String,
/// }
///
/// impl SqliteVectorStoreTable for Document {
///     fn name() -> &'static str {
///         "documents"
///     }
///
///     fn schema() -> Vec<Column> {
///         vec![
///             Column::new("id", "TEXT PRIMARY KEY"),
///             Column::new("content", "TEXT"),
///         ]
///     }
///
///     fn id(&self) -> String {
///         self.id.clone()
///     }
///
///     fn column_values(&self) -> Vec<(&'static str, Box<dyn ColumnValue>)> {
///         vec![
///             ("id", Box::new(self.id.clone())),
///             ("content", Box::new(self.content.clone())),
///         ]
///     }
/// }
/// ```
pub trait SqliteVectorStoreTable: Send + Sync + Clone {
    fn name() -> &'static str;
    fn schema() -> Vec<Column>;
    fn id(&self) -> String;
    fn column_values(&self) -> Vec<(&'static str, Box<dyn ColumnValue>)>;
}

#[derive(Clone)]
pub struct SqliteVectorStore<E: EmbeddingModel + 'static, T: SqliteVectorStoreTable + 'static> {
    conn: Connection,
    _phantom: PhantomData<(E, T)>,
}

impl<E: EmbeddingModel + 'static, T: SqliteVectorStoreTable + 'static> SqliteVectorStore<E, T> {
    pub async fn new(conn: Connection, embedding_model: &E) -> Result<Self, VectorStoreError> {
        let dims = embedding_model.ndims();
        let table_name = T::name();
        let schema = T::schema();

        // Build the table schema
        let mut create_table = format!("CREATE TABLE IF NOT EXISTS {} (", table_name);

        // Add columns
        let mut first = true;
        for column in &schema {
            if !first {
                create_table.push(',');
            }
            create_table.push_str(&format!("\n    {} {}", column.name, column.col_type));
            first = false;
        }

        create_table.push_str("\n)");

        // Build index creation statements
        let mut create_indexes = vec![format!(
            "CREATE INDEX IF NOT EXISTS idx_{}_id ON {}(id)",
            table_name, table_name
        )];

        // Add indexes for marked columns
        for column in schema {
            if column.indexed {
                create_indexes.push(format!(
                    "CREATE INDEX IF NOT EXISTS idx_{}_{} ON {}({})",
                    table_name, column.name, table_name, column.name
                ));
            }
        }

        conn.call(move |conn| {
            conn.execute_batch("BEGIN")?;

            // Create document table
            conn.execute_batch(&create_table)?;

            // Create indexes
            for index_stmt in create_indexes {
                conn.execute_batch(&index_stmt)?;
            }

            // Create embeddings table
            conn.execute_batch(&format!(
                "CREATE VIRTUAL TABLE IF NOT EXISTS {}_embeddings USING vec0(embedding float[{}])",
                table_name, dims
            ))?;

            conn.execute_batch("COMMIT")?;
            Ok(())
        })
        .await
        .map_err(|e| VectorStoreError::DatastoreError(Box::new(e)))?;

        Ok(Self {
            conn,
            _phantom: PhantomData,
        })
    }

    pub fn index(self, model: E) -> SqliteVectorIndex<E, T> {
        SqliteVectorIndex::new(model, self)
    }

    pub fn add_rows_with_txn(
        &self,
        txn: &rusqlite::Transaction<'_>,
        documents: Vec<(T, OneOrMany<Embedding>)>,
    ) -> Result<i64, tokio_rusqlite::Error> {
        info!("Adding {} documents to store", documents.len());
        let table_name = T::name();
        let mut last_id = 0;

        for (doc, embeddings) in &documents {
            debug!("Storing document with id {}", doc.id());

            let values = doc.column_values();
            let columns = values.iter().map(|(col, _)| *col).collect::<Vec<_>>();

            let placeholders = (1..=values.len())
                .map(|i| format!("?{}", i))
                .collect::<Vec<_>>();

            let insert_sql = format!(
                "INSERT OR REPLACE INTO {} ({}) VALUES ({})",
                table_name,
                columns.join(", "),
                placeholders.join(", ")
            );

            txn.execute(
                &insert_sql,
                rusqlite::params_from_iter(values.iter().map(|(_, val)| val.to_sql_string())),
            )?;
            last_id = txn.last_insert_rowid();

            let embeddings_sql = format!(
                "INSERT INTO {}_embeddings (rowid, embedding) VALUES (?1, ?2)",
                table_name
            );

            let mut stmt = txn.prepare(&embeddings_sql)?;
            for (i, embedding) in embeddings.iter().enumerate() {
                let vec = serialize_embedding(embedding);
                debug!(
                    "Storing embedding {} of {} (size: {} bytes)",
                    i + 1,
                    embeddings.len(),
                    vec.len() * 4
                );
                let blob = rusqlite::types::Value::Blob(vec.as_bytes().to_vec());
                stmt.execute(rusqlite::params![last_id, blob])?;
            }
        }

        Ok(last_id)
    }

    pub async fn add_rows(
        &self,
        documents: Vec<(T, OneOrMany<Embedding>)>,
    ) -> Result<i64, VectorStoreError> {
        let documents = documents.clone();
        let this = self.clone();

        self.conn
            .call(move |conn| {
                let tx = conn.transaction().map_err(tokio_rusqlite::Error::from)?;
                let result = this.add_rows_with_txn(&tx, documents)?;
                tx.commit().map_err(tokio_rusqlite::Error::from)?;
                Ok(result)
            })
            .await
            .map_err(|e| VectorStoreError::DatastoreError(Box::new(e)))
    }
}

/// SQLite vector store implementation for Rig.
///
/// This crate provides a SQLite-based vector store implementation that can be used with Rig.
/// It uses the `sqlite-vec` extension to enable vector similarity search capabilities.
///
/// # Example
/// ```rust
/// use rig::{
///     embeddings::EmbeddingsBuilder,
///     providers::openai::{Client, TEXT_EMBEDDING_ADA_002},
///     vector_store::VectorStoreIndex,
///     Embed,
/// };
/// use rig_sqlite::{Column, ColumnValue, SqliteVectorStore, SqliteVectorStoreTable};
/// use serde::Deserialize;
/// use tokio_rusqlite::Connection;
///
/// #[derive(Embed, Clone, Debug, Deserialize)]
/// struct Document {
///     id: String,
///     #[embed]
///     content: String,
/// }
///
/// impl SqliteVectorStoreTable for Document {
///     fn name() -> &'static str {
///         "documents"
///     }
///
///     fn schema() -> Vec<Column> {
///         vec![
///             Column::new("id", "TEXT PRIMARY KEY"),
///             Column::new("content", "TEXT"),
///         ]
///     }
///
///     fn id(&self) -> String {
///         self.id.clone()
///     }
///
///     fn column_values(&self) -> Vec<(&'static str, Box<dyn ColumnValue>)> {
///         vec![
///             ("id", Box::new(self.id.clone())),
///             ("content", Box::new(self.content.clone())),
///         ]
///     }
/// }
///
/// let conn = Connection::open("vector_store.db").await?;
/// let openai_client = Client::new("YOUR_API_KEY");
/// let model = openai_client.embedding_model(TEXT_EMBEDDING_ADA_002);
///
/// // Initialize vector store
/// let vector_store = SqliteVectorStore::new(conn, &model).await?;
///
/// // Create documents
/// let documents = vec![
///     Document {
///         id: "doc1".to_string(),
///         content: "Example document 1".to_string(),
///     },
///     Document {
///         id: "doc2".to_string(),
///         content: "Example document 2".to_string(),
///     },
/// ];
///
/// // Generate embeddings
/// let embeddings = EmbeddingsBuilder::new(model.clone())
///     .documents(documents)?
///     .build()
///     .await?;
///
/// // Add to vector store
/// vector_store.add_rows(embeddings).await?;
///
/// // Create index and search
/// let index = vector_store.index(model);
/// let results = index
///     .top_n::<Document>("Example query", 2)
///     .await?;
/// ```
pub struct SqliteVectorIndex<E: EmbeddingModel + 'static, T: SqliteVectorStoreTable + 'static> {
    store: SqliteVectorStore<E, T>,
    embedding_model: E,
}

impl<E: EmbeddingModel + 'static, T: SqliteVectorStoreTable> SqliteVectorIndex<E, T> {
    pub fn new(embedding_model: E, store: SqliteVectorStore<E, T>) -> Self {
        Self {
            store,
            embedding_model,
        }
    }
}

impl<E: EmbeddingModel + std::marker::Sync, T: SqliteVectorStoreTable> VectorStoreIndex
    for SqliteVectorIndex<E, T>
{
    async fn top_n<D: for<'a> Deserialize<'a>>(
        &self,
        query: &str,
        n: usize,
    ) -> Result<Vec<(f64, String, D)>, VectorStoreError> {
        debug!("Finding top {} matches for query", n);
        let embedding = self.embedding_model.embed_text(query).await?;
        let query_vec: Vec<f32> = serialize_embedding(&embedding);
        let table_name = T::name();

        // Get all column names from SqliteVectorStoreTable
        let columns = T::schema();
        let column_names: Vec<&str> = columns.iter().map(|column| column.name).collect();

        let rows = self
            .store
            .conn
            .call(move |conn| {
                // Build SELECT statement with all columns
                let select_cols = column_names.join(", ");
                let mut stmt = conn.prepare(&format!(
                    "SELECT d.{}, e.distance 
                    FROM {}_embeddings e
                    JOIN {} d ON e.rowid = d.rowid
                    WHERE e.embedding MATCH ?1 AND k = ?2
                    ORDER BY e.distance",
                    select_cols, table_name, table_name
                ))?;

                let rows = stmt
                    .query_map(rusqlite::params![query_vec.as_bytes().to_vec(), n], |row| {
                        // Create a map of column names to values
                        let mut map = serde_json::Map::new();
                        for (i, col_name) in column_names.iter().enumerate() {
                            let value: String = row.get(i)?;
                            map.insert(col_name.to_string(), serde_json::Value::String(value));
                        }
                        let distance: f64 = row.get(column_names.len())?;
                        let id: String = row.get(0)?; // Assuming id is always first column

                        Ok((id, serde_json::Value::Object(map), distance))
                    })?
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(rows)
            })
            .await
            .map_err(|e| VectorStoreError::DatastoreError(Box::new(e)))?;

        debug!("Found {} potential matches", rows.len());
        let mut top_n = Vec::new();
        for (id, doc_value, distance) in rows {
            match serde_json::from_value::<D>(doc_value) {
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
        let embedding = self.embedding_model.embed_text(query).await?;
        let query_vec = serialize_embedding(&embedding);
        let table_name = T::name();

        let results = self
            .store
            .conn
            .call(move |conn| {
                let mut stmt = conn.prepare(&format!(
                    "SELECT d.id, e.distance 
                     FROM {0}_embeddings e
                     JOIN {0} d ON e.rowid = d.rowid
                     WHERE e.embedding MATCH ?1 AND k = ?2
                     ORDER BY e.distance",
                    table_name
                ))?;

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

fn serialize_embedding(embedding: &Embedding) -> Vec<f32> {
    embedding.vec.iter().map(|x| *x as f32).collect()
}

impl ColumnValue for String {
    fn to_sql_string(&self) -> String {
        self.clone()
    }

    fn column_type(&self) -> &'static str {
        "TEXT"
    }
}
