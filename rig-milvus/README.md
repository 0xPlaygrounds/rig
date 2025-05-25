# Rig Milvus integration
This crate integrates [Milvus](https://milvus.io/) into Rig, allowing you to easily use RAG with this database.

## Installation
To install this crate, run the following command in a Rust project directory which will add `rig-surrealdb` as a dependency (requires `rig-core` added for intended usage):
```bash
cargo add rig-milvus
```

There's a few different ways you can run SurrealDB:
- [Install it locally and run it](https://surrealdb.com/docs/surrealdb/installation/linux)
- [Through a Docker container, either locally or on Docker-compatible architecture](https://milvus.io/docs/install_standalone-docker.md)
  - Their Docker container requires using an install script to use which is listed on the page and can be found directly on the Milvus repo.
- [Using the Zilliz cloud offering](https://surrealdb.com/cloud)
  - Using the cloud offering you can manage your Milvus instance through their web UI.

Before creating a collection that is compliant with `rig-milvus`, ensure you set up your Milvus related environment variables:
```bash
export MILVUS_DATABASE_NAME=
export MILVUS_COLLECTION_NAME=
export MILVUS_USERNAME=
export MILVUS_PASSWORD=
export MILVUS_BASE_URL=
```

To create a collection, you will need the bash script below.
```bash
export TOKEN="${MILVUS_USERNAME}:${MILVUS_PASSWORD}"

curl --request POST \
--url "${MILVUS_BASE_URL}/v2/vectordb/collections/create" \
--header "Authorization: Bearer ${TOKEN}" \
--header "Content-Type: application/json" \
-d '{
    "collectionName": "${MILVUS_COLLECTION_NAME}",
    "dbName": "${MILVUS_DATABASE_NAME}",
    "schema": {
        "autoId": true,
        "enabledDynamicField": false,
        "fields": [
            {
                "fieldName": "embedding",
                "dataType": "FloatVector",
                "elementTypeParams": {
                    "dim": "1536"
                }
            },
            {
                "fieldName": "document",
                "dataType": "JSON",
            },
            {
                "fieldName": "embeddedText",
                "dataType": "VarChar"
            }
        ]
    }
}'
```

This will create a collection with the following fields:
- an ID field (auto-generated as i64 but converted to String when using `top_n()`)
- an `embedding` field for storing vectors
- a `document` field (for storing metadata)
- an `embeddedText` field (for the actual item that was embedded)

Once done, you'll now be ready to use!

## How to run the example
To run the example, add your OpenAI API key as an environment variable. Don't forget your Milvus username, password, database & collection name and endpoint URL if you're executing this from a new terminal tab/window:
```bash
export OPENAI_API_KEY=my_key
export MILVUS_DATABASE_NAME=
export MILVUS_COLLECTION_NAME=
export MILVUS_USERNAME=
export MILVUS_PASSWORD=
export MILVUS_BASE_URL=
```

Finally, use the following command below to run the example:
```bash
cargo run --example vector_search_milvus --features rig-core/derive
```
