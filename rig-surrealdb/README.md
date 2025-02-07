# Rig SurrealDB integration
This crate integrates SurrealDB into Rig, allowing you to easily use RAG with this database.

## How to run the example
To run the example, spin up a docker  container:
```bash
docker run --rm --pull always -p 9999:8000 surrealdb/surrealdb:latest start -u root --password root
```

Once done, open another tab then add your OpenAI API key as an environment variable:
```bash
export OPENAI_API_KEY=my_key
```

Finally, use the following command below to run the example:
```bash
cargo run --example vector_search_surreal --features rig-core/derive
```
