mod agent;
mod streaming;
mod streaming_tools;
mod support;
mod tools;

const CASSETTE_MODEL: &str = rig::providers::cohere::COMMAND_A_03_2025;

mod cassette {
    mod agent;
    mod context;
    mod embedding_matrix;
    mod embeddings;
    mod errors;
    mod history_survival_matrix;
    mod prompt_caching;
    mod raw_capture_matrix;
    mod raw_stream_capture_matrix;
    mod streaming;
    mod streaming_grammar;
    mod streaming_tools;
    mod tools;
}
