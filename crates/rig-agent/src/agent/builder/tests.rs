use super::*;
use crate::test_utils::{MockAddTool, MockCompletionModel};

#[derive(Clone)]
struct BuilderHook;

impl AgentHook for BuilderHook {}

#[test]
fn hook_can_be_set_after_tool_configuration() {
    let _agent = AgentBuilder::new(MockCompletionModel::text("ok"))
        .tool(MockAddTool)
        .add_hook(BuilderHook)
        .build();
}

/// An agent over a host's bus holds no driver to tap: recording is the
/// host's, through its driver. Asking the builder for it is the host's
/// programming error, refused at build like a wrong-family host key —
/// never a debug assertion about generated keys, never a silent agent
/// that records nothing.
#[test]
#[should_panic(expected = "cannot record: the host records through its driver")]
fn recording_over_a_host_bus_is_refused_at_build() {
    let (dispatcher, registrar, _driver) = crate::bus::Bus::channel();
    let _agent = AgentBuilder::over_bus(
        dispatcher,
        registrar,
        "host",
        rig_core::effect::HandlerKey::from("model"),
    )
    .record_to(rig_cassette::effect_log::EffectLogRecorder::new())
    .build();
}

/// A store that keeps a record around the stored document (MongoDB, S3
/// Vectors) still yields it, under the result's id when it has none; any
/// other value is sent as JSON.
#[test]
fn retrieved_values_become_context_documents() {
    let result = |document| VectorSearchResult {
        score: 1.0,
        id: "doc0".to_owned(),
        document,
    };
    let wrapped = serde_json::json!({
        "document": {"id": "", "text": "A tabby."},
        "embedded_text": "A tabby.",
    });
    assert_eq!(
        context_document(result(wrapped)),
        Document::new("doc0", "A tabby.")
    );
    let other = serde_json::json!({"word": "flurbo"});
    assert_eq!(
        context_document(result(other)),
        Document::new("doc0", "{\n  \"word\": \"flurbo\"\n}")
    );
}
