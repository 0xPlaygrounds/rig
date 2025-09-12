# Rig HelixDB integration
This crate integrates HelixDB into Rig, allowing you to easily use RAG with this database.

## Installation
To install this crate, run the following command in a Rust project directory which will add `rig-helixdb` as a dependency (requires `rig-core` added for intended usage):
```bash
cargo add rig-surrealdb
```

There's a few different ways you can run HelixDB:
- Through HelixDB's cloud offering
- Running it locally through their `helix dockerdev run` command (requires the Helix CLI to be installed).

## How to run the example
Before running the example, you will need to ensure that you are running an instance of HelixDB which you can do with `helix dockerdev run`.

Once done, you will then need to deploy your queries/schema. **The queries/schema in the `examples/helixdb-cfg` folder are a required minimum to be use this integration.** `rig-helixdb` also additionally provides a way to get a manual handle on the client yourself so that you can invoke your own queries should you need to.

Assuming `rig-helixdb` is your current working directory, you can deploy the config with one command:
```bash
helix deploy --path examples/helixdb-cfg
```

This will then deploy the queries/schema into your instance.

To run the example, add your OpenAI API key as an environment variable:
```bash
export OPENAI_API_KEY=my_key
```

Finally, use the following command below to run the example:
```bash
cargo run --example vector_search_helixdb --features rig-core/derive
```

## Licensing
Unlike the rest of the crates in this workspace, `rig-helixdb` is licensed as AGPL 3.0 due to using `helix-rs` which also uses AGPL 3.0.
