use rig_harness::rig_core::message::{ToolFunction, ToolName};
use rig_harness::rig_ecs::session_log::SessionStore;
use rig_harness::rig_memory::MemoryStore;

use super::*;

/// The kernel, the plugin and a stand-in `write` tool, after the first frame restored the
/// session and spawned its agent.
fn start() -> (App, Entity) {
    let mut app = App::new();
    app.insert_resource(SessionStore::new(MemoryStore::default()))
        .add_plugins((AgentPlugin, SessionLogPlugin, SrcOnlyWritesPlugin))
        .add_open_tool(
            TOOL,
            "Write a file.",
            ToolOptions::default(),
            |_: On<ToolCalled<serde_json::Value>>| {},
        );
    app.finish();
    app.update();
    let world = app.world_mut();
    let agent = world.query_filtered::<Entity, With<Agent>>().single(world);
    (app, agent.unwrap_or(Entity::PLACEHOLDER))
}

/// Why the `write` of `path`, spawned for `agent` as its model would ask, was refused.
fn refusal(app: &mut App, agent: Entity, id: &str, path: &str) -> Option<String> {
    let name = ToolName::new(TOOL).ok()?;
    let args = serde_json::json!({ "path": path });
    let call = ToolCall::from_wire(id, ToolFunction::new(name, args));
    let call = app
        .world_mut()
        .spawn((ToolRequest { call }, ForAgent(agent)))
        .id();
    app.update();
    let outcome = app.world().get::<Outcome>(call)?;
    outcome.0.as_ref().err().map(ToString::to_string)
}

#[test]
fn only_writes_outside_src_are_refused_and_the_stages_are_ordered() {
    let (mut app, agent) = start();
    let conflicts = rig_harness::stage_conflicts(app.world_mut());
    assert!(conflicts.is_empty(), "{}", conflicts.join("\n"));
    let outside = refusal(&mut app, agent, "c1", "README.md");
    assert!(
        outside
            .as_deref()
            .is_some_and(|why| why.contains("only under src/")),
        "{outside:?}"
    );
    assert_eq!(refusal(&mut app, agent, "c2", "src/lib.rs"), None);
    assert!(refusal(&mut app, agent, "c3", "srcs/lib.rs").is_some());
}
