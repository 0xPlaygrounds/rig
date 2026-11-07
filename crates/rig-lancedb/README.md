<!-- <div style="display: flex; align-items: center; justify-content: center;">
    <picture>
        <source media="(prefers-color-scheme: dark)" srcset="../img/rig_logo_dark.svg">
        <source media="(prefers-color-scheme: light)" srcset="../img/rig_logo.svg">
        <img src="../img/rig_logo.svg" width="200" alt="Rig logo">
    </picture>
    <span style="font-size: 48px; margin: 0 20px; font-weight: regular; font-family: Open Sans, sans-serif;"> + </span>
    <picture>
        <source media="(prefers-color-scheme: dark)" srcset="https://companieslogo.com/img/orig/MDB_BIG.D-96d632a9.png?t=1720244492">
        <source media="(prefers-color-scheme: light)" srcset="https://cdn.iconscout.com/icon/free/png-256/free-mongodb-logo-icon-download-in-svg-png-gif-file-formats--wordmark-programming-langugae-freebies-pack-logos-icons-1175140.png?f=webp&w=256">
        <img src="https://cdn.iconscout.com/icon/free/png-256/free-mongodb-logo-icon-download-in-svg-png-gif-file-formats--wordmark-programming-langugae-freebies-pack-logos-icons-1175140.png?f=webp&w=256" width="200" alt="MongoDB logo">
    </picture>
</div>

<br><br> -->

## Rig-Lancedb
This companion crate implements a Rig vector store based on Lancedb.

## Pre-requisites
If you are using `rig-lancedb` locally, you must ensure you have `protoc` (the [Protobuf Compiler](https://protobuf.dev/installation/)) installed.

## Usage

Add the companion crate to your `Cargo.toml`, along with the rig-core crate:

```toml
[dependencies]
rig-lancedb = "0.46.0"
rig-core = "0.46.0"
```

You can also run `cargo add rig-lancedb rig-core` to add the most recent versions of the dependencies to your project.

See the [`/examples`](./examples) folder for usage examples.

## Inserting documents

`LanceDbVectorIndex` implements `InsertDocuments`, so you can add embedded documents to an existing table without building Arrow record batches:

```rust,ignore
let documents = EmbeddingsBuilder::new(model.clone()).documents(words)?.build().await?;
let index = LanceDbVectorIndex::new(table, model, "id", SearchParams::default()).await?;
index.insert_documents(documents).await?;
```

Each embedding becomes one row holding the document's fields. The table must already exist, with a column for each document field and a fixed-size list of floats for the embedding. Name the embedding column with `SearchParams::column` if the table has more than one. A document is refused, and nothing is written, if it is not a JSON object, has no id, has no embeddings, has a field with no column or one named like the embedding column, leaves out a non-nullable column, or has a binary value that is not a hex string, so a `Vec<u8>` field is refused. An embedding of the wrong length is refused too, and so is one with a value that is not finite in the column's float type. Other document fields convert as the JSON decoder chooses: `1.5` in an integer column stores `1`, `1e300` in a Float32 column stores infinity, and the odd-length hex string `abc` stores the bytes `ab0c`.
