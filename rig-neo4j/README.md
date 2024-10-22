# Rig-Neo4j

<br>

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

This companion crate implements a Rig vector store based on Neo4j Graph database. It uses the [neo4rs](https://github.com/neo4j-labs/neo4rs) crate to interact with Neo4j. Note that the neo4rs crate is a work in progress and does not yet support all Neo4j features.


## Usage

Add the companion crate to your `Cargo.toml`, along with the rig-core crate:

```toml
[dependencies]
rig-neo4j = "0.1"
```

You can also run `cargo add rig-neo4j rig-core` to add the most recent versions of the dependencies to your project.

See the [examples](./examples) folder for usage examples.

## Notes

- The `Neo4jVectorStore` is designed to work with a pre-existing Neo4j vector index. You can create the index using the Neo4j browser or the Neo4j language. See the [Neo4j documentation](https://neo4j.com/docs/genai/tutorials/embeddings-vector-indexes/setup/vector-index/) for more information.

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
