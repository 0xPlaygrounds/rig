//! A deterministic ACP agent for local protocol tests, without credentials.
use rig_acp::AcpPlugin;
use rig_basics::approval::{ApprovalPlugin, ApprovalSystems, NeedsApproval};
use rig_harness::prelude::*;
use rig_harness::rig_core::catalog::ModelSpec;
use rig_harness::rig_core::providers::registry::ProviderId;
use rig_harness::rig_core::test_utils::MockCompletionModel;
use rig_harness_test_support::{calls, reply, scripted_models};
use serde_json::{Value, json};

fn main() -> bevy_app::AppExit {
    let model = MockCompletionModel::from_stream_turns((0..32).map(|_| reply("scripted reply")));
    let failure = MockCompletionModel::from_stream_turns([reply("last available reply")]);
    let tool = MockCompletionModel::from_stream_turns([
        calls("call", "wait", json!({})),
        reply("permission resolved"),
    ]);
    let mcp = MockCompletionModel::from_stream_turns([
        calls("call", "injected__hello", json!({})),
        reply("MCP resolved"),
    ]);
    let read = MockCompletionModel::from_stream_turns([
        calls("call", "read", json!({"path":"fixture.txt"})),
        reply("read resolved"),
    ]);
    let mut models = scripted_models([
        ("test/acp", &model),
        ("test/failure", &failure),
        ("test/tool", &tool),
        ("test/mcp", &mcp),
        ("test/read", &read),
    ]);
    let Ok(unavailable) = ProviderId::declare("acp-unavailable") else {
        return AppExit::from_code(1);
    };
    models
        .0
        .catalog_mut()
        .insert(ModelSpec::new(unavailable, "example").with_tools(true));
    let mut app = App::new();
    app.add_plugins((HeadlessPlugins, RigHarnessPlugins))
        .insert_resource(models)
        .add_plugins((
            AcpPlugin,
            ApprovalPlugin,
            rig_basics::models::ModelsPlugin,
            rig_basics::models::defaults::DefaultsPlugin,
            rig_basics::coding_tools::ReadTool,
        ))
        .add_command(
            "scripted_turn",
            "Deliver a scripted turn",
            |In(args): In<CommandArgs>, mut commands: Commands| {
                commands.trigger(Deliver::new(args.agent, "scripted command work", args.mode));
            },
        )
        .add_open_tool(
            "wait",
            "Wait until cancelled.",
            ToolOptions::default(),
            |_called: On<ToolCalled<Value>>| {},
        )
        .add_systems(
            AgentPass,
            approve.in_set(Stage::Gate).before(ApprovalSystems),
        );
    rig_harness::run(app)
}

fn approve(
    calls: Query<(Entity, &ToolRequest), (ReadyToStart, ToolCalls)>,
    mut commands: Commands,
) {
    for (call, request) in &calls {
        if request.call.function.name == "wait" {
            commands
                .entity(call)
                .queue(NeedsApproval::add("scripted", "Waits for cancellation."));
        }
    }
}
