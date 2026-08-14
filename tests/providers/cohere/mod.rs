mod agent;
mod streaming;
mod streaming_tools;
mod support;
mod tools;

const CASSETTE_MODEL: &str = rig::providers::cohere::COMMAND_A_03_2025;

mod cassette {
    mod agent;
    mod context;
    mod embeddings;
    mod errors;
    mod response_identity;
    mod streaming;
    mod streaming_grammar;
    mod streaming_tools;
    mod tools;
}
