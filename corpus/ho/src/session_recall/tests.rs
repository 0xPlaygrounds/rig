use rig_core::test_utils::MockCompletionModel;
use rig_harness::harness_protocol::{Home, SessionId};
use rig_harness::prelude::*;
use rig_harness_test_support::{
    Ended, ToolOutputs, app_in, calls, first_agent, log_ended, log_tool_outputs, reply, run_until,
    scratch_home, scripted_models,
};

use super::RecallPlugin;

/// A `read` that finds a secret in every file.
fn read(called: On<ToolCalled<serde_json::Value>>, mut calls: Query<&mut Decision, ToolCalls>) {
    if let Ok(mut call) = calls.get_mut(called.call) {
        call.decide(
            Decider::plugin("test"),
            ToolResult::success("TOKEN=hunter2".into()),
        );
    }
}

/// Runs session `id` of `home`, the user saying `typed` and the model
/// answering with `script`, with `plugin` added; the tool outputs.
fn run_session(
    home: &Home,
    id: &str,
    typed: &str,
    script: Vec<Vec<rig_core::test_utils::MockStreamEvent>>,
    plugin: Option<RecallPlugin>,
) -> ToolOutputs {
    let model = MockCompletionModel::from_stream_turns(script);
    let id: SessionId = id.parse().unwrap_or_else(|_| SessionId::generate());
    let (mut app, wakes) = app_in(home, &id);
    app.insert_resource(scripted_models([("test/model", &model)]))
        .add_plugins((AgentPlugin, SessionLogPlugin))
        .add_open_tool("read", "Read a file.", ToolOptions::default(), read)
        .init_resource::<ToolOutputs>()
        .add_observer(log_tool_outputs)
        .init_resource::<Ended>()
        .add_observer(log_ended);
    if let Some(plugin) = plugin {
        app.add_plugins(plugin);
    }
    app.finish();
    app.update();
    let agent = first_agent(app.world_mut()).unwrap_or(Entity::PLACEHOLDER);
    let model = "test/model".to_owned();
    app.world_mut().trigger(SetModel {
        entity: agent,
        model,
        kind: ChoiceKind::ForThisRun,
    });
    app.world_mut()
        .trigger(Deliver::new(agent, typed, DeliveryMode::Steer));
    run_until(&mut app, &wakes, |world| {
        !world.resource::<Ended>().0.is_empty()
    });
    // One more frame writes the logs.
    app.update();
    app.world_mut()
        .remove_resource::<ToolOutputs>()
        .unwrap_or_default()
}

#[test]
fn recall_finds_the_past_fix_and_never_the_current_session() {
    let home = scratch_home("ho-session-recall");
    run_session(
        &home,
        "1000-1",
        "The release build fails with LNK2019, unresolved external foo_init.",
        vec![
            calls("r1", "read", serde_json::json!({ "path": ".env" })),
            reply("LNK2019 fixed by adding -lfoo to the link flags."),
        ],
        None,
    );
    run_session(
        &home,
        "2000-2",
        "Rename the parser module.",
        vec![reply("Renamed it to `syntax`.")],
        None,
    );
    // A session whose log is not JSON is skipped.
    let broken = home.session(&"3000-3".parse().unwrap_or_else(|_| SessionId::generate()));
    std::fs::create_dir_all(broken.path()).ok();
    std::fs::write(broken.path().join("agent-x.jsonl"), "not json\n").ok();
    broken.mark_saved().ok();

    let outputs = run_session(
        &home,
        "4000-4",
        "LNK2019 again! What did we do about it last time?",
        vec![
            calls(
                "c1",
                "recall",
                serde_json::json!({ "query": "LNK2019 linker", "scope": "all" }),
            ),
            reply("Add -lfoo."),
        ],
        Some(RecallPlugin),
    );
    let output = outputs.0.get("c1").cloned().unwrap_or_default();
    let first = output.split("\n\n").next().unwrap_or_default();
    assert!(first.starts_with("1. "), "{output}");
    assert!(first.contains("1000-1"), "{output}");
    assert!(first.contains("-lfoo"), "{output}");
    assert!(!output.contains("2000-2"), "{output}");
    assert!(!output.contains("4000-4"), "{output}");
    let index = home.root().join("recall-index").join("1000-1.json");
    let index = std::fs::read_to_string(index).unwrap_or_default();
    assert!(index.contains("LNK2019"), "the index is kept in RIG_HOME");
    assert!(!index.contains("hunter2"), "{index}");
}

#[test]
fn stages_are_ordered_with_the_defaults() {
    rig_defaults_test_support::assert_stages_ordered(RecallPlugin);
}
