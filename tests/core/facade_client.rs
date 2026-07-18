#[test]
fn wildcard_client_import_exposes_one_agent_constructor() {
    use rig::client::*;

    let client = rig::providers::openai::Client::from_val("test-key".into())
        .expect("test client should be constructible");
    let _builder = client.agent("test-model");
}
