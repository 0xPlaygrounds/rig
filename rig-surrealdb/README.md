# Rig SurrealDB integration
This crate integrates SurrealDB into Rig, allowing you to easily use RAG with this database.

## How to use
To use this integration, you need a SurrealDB instance that has a table with the following fields:
- `

## How to run the example
To run the example, add your OpenAI API key as an environment variable:
```bash
export OPENAI_API_KEY=my_key
```

Finally, use the following command below to run the example:
```bash
cargo run --example vector_search_surreal --features rig-core/derive
```
