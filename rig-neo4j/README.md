

<div style="display: flex; align-items: center; justify-content: center;">
    <picture>
        <source media="(prefers-color-scheme: dark)" srcset="../img/rig_logo_dark.svg">
        <source media="(prefers-color-scheme: light)" srcset="../img/rig_logo.svg">
        <img src="../img/rig_logo.svg" width="200" alt="Rig logo">
    </picture>
    <span style="font-size: 48px; margin: 0 20px; font-weight: regular; font-family: Open Sans, sans-serif;"> + </span>
    <picture>
        <source media="(prefers-color-scheme: dark)" srcset="https://cdn.prod.website-files.com/653986a9412d138f23c5b8cb/65c3ee6c93dc929503742ff6_1_E5u7PfGGOQ32_H5dUVGerQ%402x.png">
        <source media="(prefers-color-scheme: light)" srcset="https://commons.wikimedia.org/wiki/File:Neo4j-logo_color.png">
        <img src="https://commons.wikimedia.org/wiki/File:Neo4j-logo_color.png" width="200" alt="Neo4j logo">
    </picture>

</div>

<br><br>

This companion crate implements a Rig vector store based on Neo4j Graph database. It uses the [neo4rs](https://github.com/neo4j-labs/neo4rs) crate to interact with Neo4j. Note that the neo4rs crate is a work in progress and does not yet support all Neo4j features. Further documentation on Neo4j & vector search integration can be found on the [neo4rs docs](https://neo4j.com/docs/cypher-manual/current/indexes/semantic-indexes/vector-indexes/).

## Prerequisites

The GenAI plugin is enabled by default in Neo4j Aura.

The plugin needs to be installed on self-managed instances. This is done by moving the neo4j-genai.jar file from /products to /plugins in the Neo4j home directory, or, if you are using Docker, by starting the Docker container with the extra parameter --env NEO4J_PLUGINS='["genai"]'. For more information, see Operations Manual → Configure plugins.


## Usage

Add the companion crate to your `Cargo.toml`, along with the rig-core crate:

```toml
[dependencies]
rig-neo4j = "0.1"
```

You can also run `cargo add rig-neo4j rig-core` to add the most recent versions of the dependencies to your project.

See the [examples](./examples) folder for usage examples.

- [examples/vector_search_simple.rs](examples/vector_search_simple.rs) shows how to create an index on simple data.
- [examples/vector_search_movies_consume.rs](examples/vector_search_movies_consume.rs) shows how to query an existing index.
- [examples/vector_search_movies_create.rs](examples/vector_search_movies_create.rs) shows how to create embeddings & index on a large DB and query it in one go.

## Notes

- The `rig-neo4j::vector_index` module offers utility functions to create and query a Neo4j vector index. You can also create indexes using the Neo4j browser or directly call cypther queries with the Neo4rs crate. See the [Neo4j documentation](https://neo4j.com/docs/genai/tutorials/embeddings-vector-indexes/setup/vector-index/) for more information. Example [examples/vector_search_simple.rs](examples/vector_search_simple.rs) shows how to create an index on existing data.

```Cypher
CREATE VECTOR INDEX moviePlots
FOR (m:Movie)
ON m.embedding
OPTIONS {indexConfig: {
    `vector.dimensions`: 1536,
    `vector.similarity_function`: 'cosine'
}}
```

## Roadmap

- Add support for creating the vector index through RIG.
- Add support for adding embeddings to an existing database
- Add support for uploading documents to an existing database
