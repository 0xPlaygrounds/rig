//! Loader fixture that intentionally mirrors the agent-with-loaders flow.

use rig::agent::AgentBuilder;
use rig::loaders::FileLoader;

fn build_agent() {
    let _files = FileLoader::with_glob("rig-core/tests/data/loaders/*.rs")
        .expect("fixture glob should parse")
        .read_with_path()
        .ignore_errors();

    let _agent = AgentBuilder::new("model");
}
