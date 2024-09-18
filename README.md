<p align="center">
<img src="img/rig_logo.svg" alt="Rig Logo" style="width: 75%; height: 75%;"><br>
    
<a href="https://crates.io/crates/rig-core"><img src="https://img.shields.io/crates/v/rig-core.svg" /></a>
&nbsp;
<a href="[https://discord.gg/gkp6UqEUph](https://discord.gg/playgrounds)"><img src="https://img.shields.io/discord/511303648119226382?color=%236d82cc&label=Discord&logo=discord&logoColor=white" /></a>
&nbsp;
<a href=""><img src="https://img.shields.io/badge/built_with-Rust-dca282.svg?logo=rust" /></a>
&nbsp;
<a href="https://twitter.com/Playgrounds0x"><img src="https://img.shields.io/badge/Twitter-Follow_Us-1d9bf0.svg?style=flat /"></a>
&nbsp;
<a href="https://github.com/0xPlaygrounds/rig?tab=MIT-1-ov-file"><img src="https://img.shields.io/badge/license-MIT-00bfff.svg?color=red" /></a>
</p>
&nbsp;


## What is Rig?
Rig is a Rust library for building scalable, modular, and ergonomic LLM-powered applications.

More information about this crate can be found in the [crate documentation](https://docs.rs/rig-core/latest/rig/).

We'd love your feedback. Please take a moment to let us know what you think using this [Feedback form](https://bit.ly/Rig-Feeback-Form).

## Table of contents

- [What is Rig?](#what-is-rig)
- [Table of contents](#table-of-contents)
- [High-level features](#high-level-features)
- [Installation](#installation)
- [Simple example:](#simple-example)
- [Integrations](#integrations)

## High-level features
- Full support for LLM completion and embedding workflows
- Simple but powerful common abstractions over LLM providers (e.g. OpenAI, Cohere) and vector stores (e.g. MongoDB, in-memory)
- Integrate LLMs in your app with minimal boilerplate

## Installation
```bash
cargo add rig-core
```

## Simple example:
```rust
use rig::{completion::Prompt, providers::openai};

#[tokio::main]
async fn main() {
    // Create OpenAI client and model
    // This requires the `OPENAI_API_KEY` environment variable to be set.
    let openai_client = openai::Client::from_env();

    let gpt4 = openai_client.model("gpt-4").build();

    // Prompt the model and print its response
    let response = gpt4
        .prompt("Who are you?")
        .await
        .expect("Failed to prompt GPT-4");

    println!("GPT-4: {response}");
}
```
Note using `#[tokio::main]` requires you enable tokio's `macros` and `rt-multi-thread` features
or just `full` to enable all features (`cargo add tokio --features macros,rt-multi-thread`).

## Integrations
Rig supports the following LLM providers natively:
- OpenAI
- Cohere

Additionally, Rig currently has the following integration sub-libraries:
- MongoDB vector store: `rig-mongodb`
