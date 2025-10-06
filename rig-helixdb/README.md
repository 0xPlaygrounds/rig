# Rig HelixDB integration
This crate integrates HelixDB into Rig, allowing you to easily use RAG with this database.

## Installation
To install this crate, run the following command in a Rust project directory which will add `rig-helixdb` as a dependency (requires `rig-core` added for intended usage):
```bash
cargo add rig-helixdb
```

There's a few different ways you can run HelixDB:
- Through HelixDB's cloud offering
- Running it locally through their `helix start` command (requires the Helix CLI to be installed).
  - For local dev, you will likely want to use `helix push dev` for continuous iteration.

## How to run the example
Before running the example, you will need to ensure that you are running an instance of HelixDB which you can do with `helix dockerdev run`.

Once done, you will then need to deploy your queries/schema. **The queries/schema in the `examples/helixdb-cfg` folder are a required minimum to be use this integration.** `rig-helixdb` also additionally provides a way to get a manual handle on the client yourself so that you can invoke your own queries should you need to.

Assuming `rig-helixdb` is your current working directory, to deploy a minimum viable configuration for HelixDB (with `rig-helixdb`) you will need to `cd` into the `helixdb-cfg` folder and then run the following:
```bash
helix push dev
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
