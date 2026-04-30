<div style="display: flex; align-items: center; justify-content: center;">
    <picture>
        <source media="(prefers-color-scheme: dark)" srcset="../img/rig_logo_dark.svg">
        <source media="(prefers-color-scheme: light)" srcset="../img/rig_logo.svg">
        <img src="../img/rig_logo.svg" width="200" alt="Rig logo">
    </picture>
    <span style="font-size: 48px; margin: 0 20px; font-weight: regular; font-family: Open Sans, sans-serif;"> + </span>
    <picture>
        <source srcset="https://www.postgresql.org/media/img/about/press/elephant.png">
        <img src="https://www.postgresql.org/media/img/about/press/elephant.png" width="200" alt="Postgres logo">
    </picture>
</div>

<br><br>

## Rig-postgres

This companion crate implements a Rig vector store based on PostgreSQL.

## Usage

Add the companion crate to your `Cargo.toml`, along with the rig-core crate:

```toml
[dependencies]
rig-core = "0.4.0"
rig-postgres = "0.1.0"
```

You can also run `cargo add rig-core rig-postgres` to add the most recent versions of the dependencies to your project.

## PostgreSQL setup

The crate utilizes [pgvector](https://github.com/pgvector/pgvector) extension, which is available for PostgreSQL version 13 and later. Use any of the [official](https://www.postgresql.org/download/) or alternative methods to install psql.

You can install Postgres using Docker:

```sh
docker pull pgvector/pgvector:pg17

docker run -e POSTGRES_USER=myuser \
           -e POSTGRES_PASSWORD=mypassword \
           -e POSTGRES_DB=mydatabase \
           --name my_postgres \
           -p 5432:5432 \
           -d ankane/pgvector
```

Now you can configure Postgres, the recommended way is using sqlx and migrations (you can find an example inside integration tests folder).

Example sql:

```sql
-- ensure PgVector extension is installed
CREATE EXTENSION IF NOT EXISTS vector;

-- create table with embeddings using 1536 dimensions (based on OpenAI model text-embedding-3-small)
CREATE TABLE documents (
  id uuid DEFAULT gen_random_uuid(), -- we can have repeated entries
  document jsonb NOT NULL,
  embedded_text text NOT NULL,
  embedding vector(1536)
);

-- create index on embeddings
CREATE INDEX IF NOT EXISTS document_embeddings_idx ON documents
USING hnsw(embedding vector_cosine_ops); -- recommended for text embeddings

```

You can change the table name and the number of dimensions but keep the same fields schema.

You can use different indexes depending the type of distance method you want to use, check [PgVector documentation](https://github.com/pgvector/pgvector?tab=readme-ov-file#querying).

## Usage

Declare the database URL:

```sh
export DATABASE_URL="postgres://myuser:mypassword@localhost:5432/mydatabase"
```

Define the document you want to index, it has to implement `Embed`, `Serialize` and `Deserialize`.

> Note: you can index different type of documents in the same table.

Example:

```rust
#[derive(Embed, Clone, Serialize, Deserialize, Debug)]
pub struct Product {
    name: String,
    category: String,
    #[embed]
    description: String,
    price: f32
}
```

Example usage

```rust
    // Create OpenAI client
    let openai_client = rig::providers::openai::Client::from_env();
    let model = openai_client.embedding_model(rig::providers::openai::TEXT_EMBEDDING_3_SMALL);

    // connect to Postgres
    let database_url = std::env::var("DATABASE_URL").expect("DATABASE_URL not set");
    let pool = PgPoolOptions::new() .connect(&database_url) .await?;

    // run migrations (optional but recommended)
    sqlx::migrate!("./migrations").run(&pool).await?;

    // init documents
    let products: Vec<Product> = ...;

    let documents = EmbeddingsBuilder::new(model.clone())
        .documents(products)
        .unwrap()
        .build()
        .await?;

    // Create your index
    let vector_store = PostgresVectorStore::default(model, pool);

    // store documents
    vector_store.insert_documents(documents).await?;

    // retrieve embeddings
    let results = vector_store.top_n::<Product>("Which phones have more than 16Gb and support 5G", 50).await?

    ...

```
