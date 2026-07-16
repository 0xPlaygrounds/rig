mod support;

const DEFAULT_MODEL: &str = rig::providers::doubleword::QWEN3_5_9B;
const TOOL_MODEL: &str = rig::providers::doubleword::QWEN3_5_397B_A17B;

mod cassette {
    mod agent;
    mod conformance;
    mod embeddings;
    mod extractor;
    mod request_hook;
    mod streaming;
    mod streaming_tools;
    mod structured_output;
    mod tools;
    mod typed_prompt_tools;
}
