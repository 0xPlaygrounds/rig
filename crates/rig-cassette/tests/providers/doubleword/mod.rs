mod support;

const DEFAULT_MODEL: &str = rig::providers::doubleword::QWEN3_5_9B;
const TOOL_MODEL: &str = rig::providers::doubleword::QWEN3_5_397B_A17B;

mod cassette {
    mod agent;
    mod conformance;
    mod corpus_faults;
    mod corpus_matrix;
    mod embedding_dimensions;
    mod embedding_matrix;
    mod embeddings;
    mod error_matrix;
    mod finish_reason_matrix;
    mod history_survival_matrix;
    mod prompt_caching;
    mod request_parameter_matrix;
    mod streaming;
    mod structured_output;
    mod tools;
    mod typed_prompt_tools;
}
