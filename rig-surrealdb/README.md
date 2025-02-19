# Rig SurrealDB integration
This crate integrates SurrealDB into Rig, allowing you to easily use RAG with this database.

## Installation
To install this crate, run the following command in a Rust project directory which will add `rig-surrealdb` as a dependency (requires `rig-core` added for intended usage):
```bash
cargo add rig-surrealdb
```

There's a few different ways you can run SurrealDB:
- [Install it locally and run it](https://surrealdb.com/docs/surrealdb/installation/linux)
- [Through a Docker container, either locally or on Docker-compatible architecture](https://surrealdb.com/docs/surrealdb/installation/running/docker)
  - `docker run --rm --pull always -p 8000:8000 surrealdb/surrealdb:latest start --username root --password root` starts up a SurrealDB instance at port 8000 with the username and password as "root".
- [Using SurrealDB's cloud offering](https://surrealdb.com/cloud)
  - Using the cloud offering you can manage your SurrealDB instance through their web UI.

## How to run the example
To run the example, add your OpenAI API key as an environment variable:
```bash
export OPENAI_API_KEY=my_key
```

Finally, use the following command below to run the example:
```bash
cargo run --example vector_search_surreal --features rig-core/derive
```
