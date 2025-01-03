use pgvector::Vector;
use rig::{
    embeddings::embedding::{Embedding, EmbeddingModel},
    vector_store::{VectorStoreError, VectorStoreIndex},
    OneOrMany,
};
use serde::Deserialize;
use std::marker::PhantomData;
use tokio_postgres::{types::ToSql, Client};
use tracing::debug;

pub struct Column {
    name: &'static str,
    col_type: &'static str,
}

impl Column {
    pub fn new(name: &'static str, col_type: &'static str) -> Self {
        Self { name, col_type }
    }
}

/// Example of a document type taht can be used with PostgresVectoreStore
/// ```rust
///
/// use rig::Embed;
/// use rig_postgres::{Column, PostgresVectorStoreTable};
/// use tokio_postgres::types::ToSql;
///
/// #[derive(Clone, Debug, Embed)]
/// pub struct Document {
///     id: String,
///     #[embed]
///     content: String,
/// }
///
/// impl PostgresVectorStoreTable for Document {
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
///     fn column_values(&self) -> Vec<(&'static str, Box<dyn ToSql + Sync>)> {
///         vec![
///             ("id", Box::new(self.id.clone())),
///             ("content", Box::new(self.content.clone())),
///         ]
///     }
/// }
/// ```
pub trait PostgresVectorStoreTable: Send + Sync + Clone {
    fn name() -> &'static str;
    fn schema() -> Vec<Column>;
    fn column_values(&self) -> Vec<(&'static str, Box<dyn ToSql + Sync>)>;
}

pub struct PostgresVectorStore<E: EmbeddingModel + 'static, T: PostgresVectorStoreTable + 'static> {
    client: Client,
    _phantom: PhantomData<(E, T)>,
}

impl<E: EmbeddingModel + 'static, T: PostgresVectorStoreTable + 'static> PostgresVectorStore<E, T> {
    pub async fn new(client: Client, embedding_model: &E) -> Result<Self, VectorStoreError> {
        let dims = embedding_model.ndims();
        let table_name = T::name();
        let schema = T::schema();

        async {
            // ensure extension is installed
            client
                .execute("CREATE EXTENSION IF NOT EXISTS vector;", &[])
                .await?;

            // create the table according to schema, with an extra `embeddings vector(...)` column
            let columns = schema
                .iter()
                .map(|col| format!("{} {}", col.name, col.col_type))
                .collect::<Vec<_>>()
                .join(", ");
            let embeddings_col = format!("embeddings vector({})", dims);

            debug!("Creating table: {}", table_name);
            let create_table = format!("CREATE TABLE IF NOT EXISTS {} ({}, {})", table_name, columns, embeddings_col);
            client.execute(&create_table, &[]).await?;

            // create the index on the `embeddings` column
            debug!("Creating index on embeddings column");
            client
                .execute(
                    &format!(
                        "CREATE INDEX IF NOT EXISTS {}_embeddings_idx ON {} USING hnsw(embeddings vector_cosine_ops)",
                        table_name, table_name
                    ),
                    &[],
                )
                .await?;

            Ok::<_, tokio_postgres::Error>(())
        }
        .await
        .map_err(|e| VectorStoreError::DatastoreError(Box::new(e)))?;

        Ok(Self {
            client,
            _phantom: PhantomData,
        })
    }

    pub async fn add_rows(
        &self,
        documents: Vec<(T, OneOrMany<Embedding>)>,
    ) -> Result<(), VectorStoreError> {
        let table_name = T::name();
        let (mut columns, mut placeholders): (Vec<&str>, Vec<String>) = T::schema()
            .iter()
            .enumerate()
            .map(|(index, col)| (col.name, format!("${}", index + 1)))
            .unzip();
        columns.push("embeddings");
        placeholders.push(format!("${}", placeholders.len() + 1));
        let columns = columns.join(", ");
        let placeholders = placeholders.join(", ");

        let query_string = &format!(
            "INSERT INTO {} ({}) VALUES ({})",
            table_name, columns, placeholders
        );
        let query = self
            .client
            .prepare(query_string)
            .await
            .map_err(|e| VectorStoreError::DatastoreError(Box::new(e)))?;

        debug!(
            "Inserting {} rows into table: {}",
            documents.len(),
            table_name
        );
        for (doc, embeddings) in &documents {
            let embedding_vector: Vector = embeddings
                .iter()
                .flat_map(|e| e.vec.iter().map(|e| *e as f32))
                .collect::<Vec<_>>()
                .into();

            // building the parameters we use in the query:
            // first, we select only the values from the column_values() tuples
            // and then append the embedding vector as the last parameter for the insert query
            let column_values = doc.column_values();
            let params: Vec<&(dyn ToSql + Sync)> = column_values
                .iter()
                .map(|(_, v)| &**v)
                .chain(std::iter::once(&embedding_vector as &(dyn ToSql + Sync)))
                .collect();

            self.client
                .execute(&query, &params)
                .await
                .map_err(|e| VectorStoreError::DatastoreError(Box::new(e)))?;
        }

        Ok(())
    }
}

/// PostgreSQL vector store implementation for Rig.
///
/// This crate provides a PostgreSQL vector store implementation for Rig. It uses the pgvector extension
/// to store embeddings and perform similarity searches.
///
/// # Example
/// ```rust,ignore
/// use rig::{
///     embeddings::EmbeddingsBuilder,
///     providers::openai::{Client, TEXT_EMBEDDING_3_SMALL},
///     vector_store::VectorStoreIndex,
///     Embed,
/// };
/// use rig_postgres::{Column, PostgresVectorIndex, PostgresVectorStore, PostgresVectorStoreTable};
/// use serde::Deserialize;
/// use tokio_postgres::types::ToSql;
///
/// #[derive(Clone, Debug, Deserialize, Embed)]
/// pub struct Document {
///     id: String,
///     #[embed]
///     content: String,
/// }
///
/// impl PostgresVectorStoreTable for Document {
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
///     fn column_values(&self) -> Vec<(&'static str, Box<dyn ToSql + Sync>)> {
///         vec![
///             ("id", Box::new(self.id.clone())),
///             ("content", Box::new(self.content.clone())),
///         ]
///     }
/// }
///
/// # tokio_test::block_on(async {
/// # Result::<(), Box<dyn std::error::Error>>::Ok({
/// // set up postgres connection
/// let database_url = std::env::var("DATABASE_URL").expect("DATABASE_URL not set");
/// let db_config: tokio_postgres::Config = database_url.parse()?;
/// let (psql, connection) = db_config.connect(tokio_postgres::NoTls).await?;
///
/// tokio::spawn(async move {
///    if let Err(e) = connection.await {
///        tracing::error!("Connection error: {}", e);
///    }
/// });
///
/// // set up embedding model
/// let openai_api_key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY not set");
/// let openai = Client::new(&openai_api_key);
/// let model = openai.embedding_model(TEXT_EMBEDDING_3_SMALL);
///
/// // generate embeddings
/// let documents: Vec<Document> = vec![
///     "The Mediterranean diet emphasizes fish, olive oil, and vegetables, believed to reduce chronic diseases.",
///     "Photosynthesis in plants converts light energy into glucose and produces essential oxygen.",
///     "20th-century innovations, from radios to smartphones, centered on electronic advancements.",
/// ].into_iter().map(|content| Document {
///     id: uuid::Uuid::new_v4().to_string(),
///     content: content.to_string(),
/// }).collect();
/// let embeddings = EmbeddingsBuilder::new(model.clone())
///     .documents(documents)?
///     .build()
///     .await?;
///
/// // add embeddings to store
/// let store = PostgresVectorStore::new(psql, &model).await?;
/// store.add_rows(embeddings).await?;
///
/// // query the index
/// let index = PostgresVectorIndex::new(model, store);
/// let results = index.top_n::<Document>("What is photosynthesis", 1).await?;
/// # })
/// # });
/// ```
pub struct PostgresVectorIndex<E: EmbeddingModel + 'static, T: PostgresVectorStoreTable + 'static> {
    store: PostgresVectorStore<E, T>,
    embedding_model: E,
}

impl<E: EmbeddingModel + 'static, T: PostgresVectorStoreTable> PostgresVectorIndex<E, T> {
    pub fn new(embedding_model: E, store: PostgresVectorStore<E, T>) -> Self {
        Self {
            store,
            embedding_model,
        }
    }
}

impl<E: EmbeddingModel + Sync, T: PostgresVectorStoreTable> VectorStoreIndex
    for PostgresVectorIndex<E, T>
{
    async fn top_n<D: for<'a> Deserialize<'a> + Send>(
        &self,
        query: &str,
        n: usize,
    ) -> Result<Vec<(f64, String, D)>, VectorStoreError> {
        let embedding = self.embedding_model.embed_text(query).await?;
        let vector: Vector = embedding
            .vec
            .iter()
            .map(|e| *e as f32)
            .collect::<Vec<f32>>()
            .into();
        let table_name = T::name();
        let column_names = T::schema()
            .iter()
            .map(|col| col.name)
            .collect::<Vec<_>>()
            .join(", ");

        let query_string = format!(
            "SELECT {}, embeddings <=> $1 AS distance FROM {} ORDER BY distance LIMIT $2",
            column_names, table_name
        );
        let rows = self
            .store
            .client
            .query(&query_string, &[&vector, &(n as i64)])
            .await
            .map_err(|e| VectorStoreError::DatastoreError(Box::new(e)))?;

        let res = rows
            .into_iter()
            .map(|row| {
                let rlen = row.len();
                let id: String = row.get(0); // assuming id is first column
                let distance: f64 = row.get(rlen - 1); // distance is last column

                let mut map = serde_json::Map::new();
                for (i, column) in row.columns().iter().enumerate().take(rlen - 1) {
                    let name = column.name();
                    let value = serde_json::Value::String(row.get(i));
                    map.insert(name.to_string(), value);
                }

                let doc_value = serde_json::Value::Object(map);
                match serde_json::from_value::<D>(doc_value) {
                    Ok(doc) => Ok((distance, id, doc)),
                    Err(e) => Err(VectorStoreError::DatastoreError(Box::new(e))),
                }
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(res)
    }

    async fn top_n_ids(
        &self,
        query: &str,
        n: usize,
    ) -> Result<Vec<(f64, String)>, VectorStoreError> {
        let embedding = self.embedding_model.embed_text(query).await?;
        let vector: Vector = embedding
            .vec
            .iter()
            .map(|e| *e as f32)
            .collect::<Vec<f32>>()
            .into();
        let table_name = T::name();
        let id_col_name = T::schema()[0].name;

        let query = format!(
            "SELECT {}, embeddings <=> $1 AS distance FROM {} ORDER BY embeddings <=> $1 LIMIT $2",
            id_col_name, table_name
        );
        let rows = self
            .store
            .client
            .query(&query, &[&vector, &(n as i64)])
            .await
            .map_err(|e| VectorStoreError::DatastoreError(Box::new(e)))?;

        let res = rows
            .into_iter()
            .map(|row| {
                let id: String = row.get(0);
                let distance: f64 = row.get(1);
                Ok::<(f64, String), VectorStoreError>((distance, id))
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(res)
    }
}
