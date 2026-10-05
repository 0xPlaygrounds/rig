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

mod conversation_without_memory {
    use super::*;
    use crate::test_utils::TraceCapture;

    /// The messages of every event at warn level or above.
    fn warnings(capture: &TraceCapture) -> Vec<String> {
        capture
            .events()
            .iter()
            .filter(|event| event.level <= tracing::Level::WARN)
            .map(|event| event.message())
            .collect()
    }

    /// A conversation id with no memory backend is not silently inert: the
    /// build warns once, naming the setter. With a backend it does not.
    #[tokio::test]
    async fn build_warns_when_conversation_has_no_memory_backend() {
        let _isolation = crate::test_utils::scoped_tracing_subscriber_guard().await;
        let capture = TraceCapture::default();
        let _default = tracing::subscriber::set_default(capture.subscriber());

        let _agent = AgentBuilder::new(MockCompletionModel::text("x"))
            .conversation("thread-1")
            .build();
        let seen = warnings(&capture);
        assert_eq!(seen.len(), 1, "{seen:?}");
        assert!(seen[0].contains("AgentBuilder::conversation"), "{seen:?}");

        capture.clear();
        let _agent = AgentBuilder::new(MockCompletionModel::text("x"))
            .memory(rig_core::memory::InMemoryConversationMemory::new())
            .conversation("thread-1")
            .build();
        assert!(warnings(&capture).is_empty());
    }
}
