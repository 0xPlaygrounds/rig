mod support;

const DEFAULT_MODEL: &str = rig::providers::doubleword::QWEN3_5_9B;
const TOOL_MODEL: &str = rig::providers::doubleword::QWEN3_5_397B_A17B;

mod cassette {
    mod agent;
    mod conformance;
    mod embedding_dimensions;
    mod embeddings;
    mod error_matrix;
    mod extractor;
    mod finish_reason_matrix;
    mod model_family_matrix;
    mod prompt_caching;
    mod raw_capture_matrix;
    mod raw_stream_capture_matrix;
    mod reasoning_matrix;
    mod request_hook;
    mod request_parameter_matrix;
    mod response_identity_edge;
    mod streaming;
    mod streaming_tools;
    mod structured_output;
    mod tools;
    mod typed_prompt_tools;
}
