use std::path::Path;

use rig_core::test_utils::MockCompletionModel;
use rig_harness::harness_protocol::SessionId;
use serde_json::json;

use super::*;
use rig_harness_test_support::{
    Ended, ToolOutputs, app_in, calls, first_agent, log_ended, log_tool_outputs, reply, run_until,
    scratch_home, scripted_models,
};

/// The progress of the running `run` call, if one runs.
fn progress(world: &World) -> Option<&RunProgress> {
    let mut calls = world.try_query::<&RunProgress>()?;
    calls.iter(world).next()
}

/// A `run` call streams its lines into its progress; `/cancel-call` stops
/// its process group and answers it with the output so far, and the turn
/// carries on with the model reading that result.
#[test]
fn a_cancelled_run_gives_the_model_its_output_so_far() {
    let command = "for i in 1 2 3; do echo $i; sleep 0.1; done; sleep 30";
    let model = MockCompletionModel::from_stream_turns([
        calls("c1", RUN, json!({ "command": command })),
        reply("The build was cancelled; I stop here."),
    ]);
    let home = scratch_home("ho-cancellable-run");
    let (mut app, wakes) = app_in(&home, &SessionId::generate());
    app.insert_resource(scripted_models([("test/model", &model)]))
        .add_plugins((AgentPlugin, JournalPlugin, CancellableRunPlugin))
        .init_resource::<ToolOutputs>()
        .add_observer(log_tool_outputs)
        .init_resource::<Ended>()
        .add_observer(log_ended);
    app.finish();
    app.update();
    let agent = first_agent(app.world_mut()).unwrap_or(Entity::PLACEHOLDER);
    let model_name = "test/model".to_owned();
    app.world_mut().trigger(SetModel {
        entity: agent,
        model: model_name,
    });
    app.world_mut()
        .trigger(Deliver::new(agent, "Build it.", DeliveryMode::Steer));

    run_until(&mut app, &wakes, |world| {
        progress(world).is_some_and(|progress| progress.lines.iter().any(|line| line == "3"))
    });
    let pid = progress(app.world()).map_or(0, |progress| progress.pid);
    assert_ne!(pid, 0, "the call never printed 3");

    app.world_mut()
        .trigger(RunCommand::new(agent, CANCEL.to_owned()));
    run_until(&mut app, &wakes, |world| {
        !world.resource::<Ended>().0.is_empty()
    });

    let outputs = &app.world().resource::<ToolOutputs>().0;
    let output = outputs.get("c1").map(String::as_str).unwrap_or_default();
    assert!(output.contains("1\n2\n3\n"), "{output}");
    assert!(output.contains("[cancelled by user after "), "{output}");
    assert!(
        !Path::new(&format!("/proc/{pid}")).exists(),
        "pid {pid} still runs"
    );

    let requests = model.requests();
    assert_eq!(requests.len(), 2);
    let last = requests
        .get(1)
        .and_then(|request| request.chat_history.last())
        .map(|message| format!("{message:?}"))
        .unwrap_or_default();
    assert!(last.contains("cancelled by user"), "{last}");
    let ended = &app.world().resource::<Ended>().0;
    assert!(
        matches!(ended.as_slice(), [TurnOutcome::Answered(_)]),
        "{ended:?}"
    );
}
