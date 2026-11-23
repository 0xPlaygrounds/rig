use std::fs;

use rig_core::test_utils::MockCompletionModel;
use rig_harness_test_support::{
    Ended, MemoryStore, ToolOutputs, app_on, calls, first_agent, log_ended, log_tool_outputs,
    reply, run_until, scratch_dir, scripted_models,
};

use super::*;

/// A call the model makes is answered with the file's word count.
#[test]
fn a_call_is_answered_with_the_count() {
    let dir = scratch_dir(&format!("word-count-{}", std::process::id()));
    fs::write(dir.join("notes.txt"), "one two  three\nfour\n").ok();
    let model = MockCompletionModel::from_stream_turns([
        calls(
            "c1",
            "word_count",
            serde_json::json!({ "path": "notes.txt" }),
        ),
        reply("It has 4 words."),
    ]);
    let (mut app, wakes) = app_on(&MemoryStore::default());
    app.insert_resource(scripted_models([("test/model", &model)]))
        .insert_resource(ProjectRoot(dir.clone()))
        .add_plugins((AgentPlugin, SessionLogPlugin, WordCountPlugin))
        .init_resource::<ToolOutputs>()
        .add_observer(log_tool_outputs)
        .init_resource::<Ended>()
        .add_observer(log_ended);
    app.finish();
    app.update();
    let world = app.world_mut();
    let agent = first_agent(world).unwrap_or(Entity::PLACEHOLDER);
    world.trigger(SetModel {
        entity: agent,
        model: "test/model".to_owned(),
        kind: ChoiceKind::ForThisRun,
    });
    world.trigger(Deliver::new(
        agent,
        "How long is notes.txt?",
        DeliveryMode::Steer,
    ));
    run_until(&mut app, &wakes, |world| {
        !world.resource::<Ended>().0.is_empty()
    });

    let output = app.world().resource::<ToolOutputs>().0.get("c1").cloned();
    assert_eq!(output.as_deref(), Some("4"));
    fs::remove_dir_all(&dir).ok();
}
