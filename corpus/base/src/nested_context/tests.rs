use std::path::PathBuf;

use rig_basics::coding_tools::ReadTool;
use rig_core::test_utils::MockCompletionModel;
use rig_harness_test_support::MemoryStore;
use serde_json::json;

use super::*;
use rig_harness_test_support::{app_on, calls, connect, reply, run_until};

/// A project directory next to the test binary, inside the target
/// directory, made empty.
fn project() -> Option<PathBuf> {
    let exe = std::env::current_exe().ok()?;
    let dir = exe
        .parent()?
        .join(format!("nested-context-{}", std::process::id()));
    std::fs::remove_dir_all(&dir).ok();
    std::fs::create_dir_all(dir.join("pkg")).ok()?;
    std::fs::write(dir.join("pkg/AGENTS.md"), "Use tabs.\n").ok()?;
    std::fs::write(dir.join("pkg/a.txt"), "alpha\n").ok()?;
    Some(dir)
}

/// Each tool result in the conversation, as text, in order.
fn results(conversation: &Conversation) -> Vec<String> {
    conversation
        .messages()
        .iter()
        .filter_map(|message| match &**message {
            message::Message::User { content } => Some(content.iter()),
            _ => None,
        })
        .flatten()
        .filter_map(|item| match item {
            UserContent::ToolResult(result) => Some(format!("{result:?}")),
            _ => None,
        })
        .collect()
}

#[test]
fn a_nested_agents_md_comes_with_the_first_read_under_it_only() {
    let dir = project();
    assert!(dir.is_some(), "could not make the project directory");
    let Some(dir) = dir else {
        return;
    };
    let file = dir.join("pkg/a.txt").display().to_string();
    let model = MockCompletionModel::from_stream_turns([
        calls("r1", "read", json!({ "path": file })),
        calls("r2", "read", json!({ "path": file })),
        reply("done"),
    ]);
    let (mut app, wakes) = app_on(&MemoryStore::default());
    app.set_error_handler(bevy_ecs::error::warn)
        .insert_resource(ProjectRoot(dir.clone()))
        .add_plugins((AgentPlugin, SessionLogPlugin, ReadTool, NestedContextPlugin));
    app.finish();
    app.update();
    let world = app.world_mut();
    let agent = world.query_filtered::<Entity, With<Agent>>().single(world);
    let (agent, connection) = (agent.ok(), connect(&model));
    assert!(
        agent.is_some() && connection.is_some(),
        "an agent and a model"
    );
    let (Some(agent), Some(connection)) = (agent, connection) else {
        return;
    };
    world.entity_mut(agent).insert(connection);
    world.trigger(Deliver::new(agent, "Read it twice.", DeliveryMode::Steer));
    run_until(&mut app, &wakes, |world| {
        world.get::<ActiveTurn>(agent).is_none() && model.requests().len() == 3
    });

    let results = app
        .world()
        .get::<Conversation>(agent)
        .map(results)
        .unwrap_or_default();
    std::fs::remove_dir_all(&dir).ok();
    assert_eq!(results.len(), 2, "two tool results: {results:?}");
    let [first, second] = results.as_slice() else {
        return;
    };
    assert!(first.contains("alpha"), "{first}");
    assert!(first.contains("Use tabs."), "{first}");
    assert!(first.contains("Rules for `pkg/`"), "{first}");
    assert!(second.contains("alpha"), "{second}");
    assert!(!second.contains("Use tabs."), "{second}");
}

#[test]
fn stages_are_ordered_with_the_defaults() {
    rig_defaults_test_support::assert_stages_ordered(NestedContextPlugin);
}
