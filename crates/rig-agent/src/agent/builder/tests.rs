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
