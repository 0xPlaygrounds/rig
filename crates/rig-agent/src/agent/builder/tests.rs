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

/// Provider-hosted tools declared on the builder reach the request beside
/// the function tools rig runs, after the agent's own parameters.
#[tokio::test]
async fn provider_tools_reach_the_request_beside_function_tools() {
    let model = MockCompletionModel::text("ok");
    let agent = AgentBuilder::new(model.clone())
        .additional_params(serde_json::json!({"top_p": 0.5}))
        .provider_tool(ProviderToolDefinition::new("web_search"))
        .tool(MockAddTool)
        .build();
    agent.prompt("search").await.expect("the run answers");

    let requests = model.requests();
    let request = requests.first().expect("one request");
    assert_eq!(
        request
            .tools
            .iter()
            .map(|tool| tool.name.as_str())
            .collect::<Vec<_>>(),
        ["add"]
    );
    assert_eq!(
        request.additional_params,
        Some(serde_json::json!({"top_p": 0.5, "tools": [{"type": "web_search"}]}))
    );
}
