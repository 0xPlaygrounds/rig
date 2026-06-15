//! Loader fixture that intentionally mirrors the agent-with-loaders flow.

use rig::agent::AgentBuilder;
use rig::loaders::FileLoader;

const LOADERS_GLOB: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/loaders/*.rs");

fn build_agent() {
    let _files = FileLoader::with_glob(LOADERS_GLOB)
        .expect("fixture glob should parse")
        .read_with_path()
        .ignore_errors();

    let _agent = AgentBuilder::new("model");
}
