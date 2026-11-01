use std::path::Path;
use std::sync::mpsc::Receiver;

use rig_core::test_utils::MockCompletionModel;
use rig_harness::harness_protocol::SessionId;
use serde_json::json;

use super::*;
use rig_harness_test_support::{
    Ended, ToolOutputs, app_in, calls, first_agent, log_ended, log_tool_outputs, reply, run_until,
    scratch_home, scripted_models,
};

/// Prints 1, 2 and 3, then waits half a minute.
const COMMAND: &str = "for i in 1 2 3; do echo $i; sleep 0.1; done; sleep 30";

/// The progress line and pid of the running `run` call, if one runs.
fn progress(world: &World) -> Option<(&ToolProgress, u32)> {
    let mut calls = world.try_query::<(&ToolProgress, &RunOutput)>()?;
    let (progress, output) = calls.iter(world).next()?;
    Some((progress, output.pid))
}

/// An app whose model asks to run [`COMMAND`] and answers in text after,
/// run until the command printed 3: its wakes, agent and the command's pid.
fn running(model: &MockCompletionModel, name: &str) -> (App, Receiver<()>, Entity, u32) {
    let home = scratch_home(name);
    let (mut app, wakes) = app_in(&home, &SessionId::generate());
    app.insert_resource(scripted_models([("test/model", model)]))
        .add_plugins((AgentPlugin, SessionLogPlugin, CancellableRunPlugin))
        .init_resource::<ToolOutputs>()
        .add_observer(log_tool_outputs)
        .init_resource::<Ended>()
        .add_observer(log_ended);
    app.finish();
    app.update();
    let agent = first_agent(app.world_mut()).unwrap_or(Entity::PLACEHOLDER);
    app.world_mut().trigger(SetModel {
        entity: agent,
        model: "test/model".to_owned(),
        kind: ChoiceKind::ForThisRun,
    });
    app.world_mut()
        .trigger(Deliver::new(agent, "Build it.", DeliveryMode::Steer));
    run_until(&mut app, &wakes, |world| {
        progress(world).is_some_and(|(ToolProgress(line), _)| line == "3")
    });
    let pid = progress(app.world()).map_or(0, |(_, pid)| pid);
    assert_ne!(pid, 0, "the call never printed 3");
    (app, wakes, agent, pid)
}

fn script(model_says: &str) -> MockCompletionModel {
    MockCompletionModel::from_stream_turns([
        calls("c1", RUN, json!({ "command": COMMAND })),
        reply(model_says),
    ])
}

/// A `run` call streams its lines into its progress; `/cancel-call` stops
/// its process group and answers it with the output so far, and the turn
/// carries on with the model reading that result.
#[test]
fn a_cancelled_run_gives_the_model_its_output_so_far() {
    let model = script("The build was cancelled; I stop here.");
    let (mut app, wakes, agent, pid) = running(&model, "ho-cancellable-run");

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

/// Interrupting the turn kills the command, and its result is the output
/// so far with `[interrupted]`.
#[test]
fn an_interrupted_run_keeps_its_output_so_far() {
    let model = script("unused");
    let (mut app, wakes, agent, pid) = running(&model, "ho-cancellable-run-interrupt");

    app.world_mut().trigger(StopTurn::new(agent));
    run_until(&mut app, &wakes, |world| {
        !world.resource::<Ended>().0.is_empty() && !Path::new(&format!("/proc/{pid}")).exists()
    });

    let mut conversations = app.world_mut().query::<&Conversation>();
    let last = conversations
        .get(app.world(), agent)
        .ok()
        .and_then(|conversation| conversation.messages().last())
        .map(|message| format!("{message:?}"))
        .unwrap_or_default();
    assert!(last.contains("1\\n2\\n3\\n[interrupted]"), "{last}");
    assert!(
        !Path::new(&format!("/proc/{pid}")).exists(),
        "pid {pid} still runs"
    );
    assert_eq!(model.requests().len(), 1);
}
