use rig_core::test_utils::{MockCompletionModel, MockStreamEvent};
use rig_harness::harness_protocol::SessionId;
use rig_harness::host::exit::ExitPlugin;
use serde_json::json;

use super::*;
use rig_harness_test_support::{
    Ended, ToolOutputs, app_in, calls, first_agent, log_ended, log_tool_outputs, reply, run_until,
    scratch_home, scripted_models,
};

#[test]
fn stages_are_ordered_with_the_defaults() {
    rig_defaults_test_support::assert_stages_ordered(TerminalSessionsPlugin);
}

/// A shell's state survives across turns; a running job times out and
/// Ctrl+C stops it; the screen shows what a carriage return overwrote; an
/// exit is reported with its code; a fifth terminal is refused naming the
/// four open; quitting kills what still runs.
#[test]
fn a_shell_is_driven_across_turns_and_dies_with_the_app() {
    let send = |id: &str, input: &str, extra: serde_json::Value| {
        let mut args = json!({ "name": "s", "input": input });
        if let (Some(args), Some(extra)) = (args.as_object_mut(), extra.as_object()) {
            args.extend(extra.clone());
        }
        calls(id, "term_send", args)
    };
    let open = |id: &str, name: &str, command: &str| {
        MockStreamEvent::tool_call(id, "term_open", json!({ "command": command, "name": name }))
    };
    let shell = "PS1='$ ' sh";
    // Calls wait on what the shell prints, not on a pause, which a loaded
    // machine can stretch: here a prompt alone on its line.
    let prompt_line = "(?m)^\\$ *$";
    let model = MockCompletionModel::from_stream_turns([
        // Turn 1.
        calls(
            "open",
            "term_open",
            json!({ "command": shell, "name": "s" }),
        ),
        send("set", "X=41<Enter>", json!({ "wait_for": prompt_line })),
        reply("set"),
        // Turn 2.
        send("echo", "echo $((X+1))<Enter>", json!({ "wait_for": "42" })),
        send("sleep", "sleep 100<Enter>", json!({ "timeout": 1 })),
        send(
            "stop",
            "<C-c>",
            json!({ "timeout": 2, "wait_for": prompt_line }),
        ),
        send(
            "cr",
            "printf 'aaa\\rb\\n'<Enter>",
            json!({ "wait_for": "(?m)^baa" }),
        ),
        calls("screen", "term_read", json!({ "name": "s" })),
        // A pattern never printed: the call waits for the exit.
        send(
            "exit",
            "exit 3<Enter>",
            json!({ "wait_for": "never printed" }),
        ),
        calls("exited", "term_read", json!({ "name": "s" })),
        vec![
            open("a", "a", shell),
            open("b", "b", shell),
            open("c", "c", shell),
            open("d", "d", "sleep 1000"),
            open("e", "e", shell),
            MockStreamEvent::final_response_with_default_usage(),
        ],
        reply("done"),
    ]);
    let home = scratch_home("terminal-sessions");
    let (mut app, wakes) = app_in(&home, &SessionId::generate());
    app.insert_resource(scripted_models([("test/model", &model)]))
        .add_plugins((
            AgentPlugin,
            SessionLogPlugin,
            ExitPlugin,
            TerminalSessionsPlugin,
        ))
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
        kind: ChoiceKind::ForThisRun,
    });
    let ended = |count: usize| move |world: &World| world.resource::<Ended>().0.len() >= count;
    app.world_mut()
        .trigger(Deliver::new(agent, "Start a shell.", DeliveryMode::Steer));
    run_until(&mut app, &wakes, ended(1));
    app.world_mut()
        .trigger(Deliver::new(agent, "Carry on.", DeliveryMode::Steer));
    run_until(&mut app, &wakes, ended(2));

    let outputs = app.world().resource::<ToolOutputs>().0.clone();
    let output = |id: &str| outputs.get(id).cloned().unwrap_or_default();
    // A prompt alone on its line; the screen's lines lose trailing blanks.
    let prompt = |text: &str| text.lines().any(|line| line == "$");
    // The first screen may still be blank: the shell can start slowly.
    assert!(
        output("open").starts_with("Opened terminal `s`"),
        "{}",
        output("open")
    );
    assert!(prompt(&output("set")), "{}", output("set"));
    assert!(output("echo").contains("42"), "{}", output("echo"));
    assert!(
        output("sleep").contains("still running"),
        "{}",
        output("sleep")
    );
    let stopped = output("stop");
    assert!(
        prompt(&stopped) && !stopped.contains("timed out"),
        "{stopped}"
    );
    let screen = output("screen");
    assert!(screen.lines().any(|line| line == "baa"), "{screen}");
    assert!(
        output("exited").contains("exited with code 3"),
        "{}",
        output("exited")
    );
    let refused = output("e");
    for name in ["`a`", "`b`", "`c`", "`d`"] {
        assert!(refused.contains(name.trim_matches('`')), "{refused}");
    }
    assert!(refused.contains("At most 4"), "{refused}");

    let mut terminals = app.world_mut().query::<&Terminal>();
    let sleeper = terminals
        .iter(app.world())
        .find(|terminal| terminal.name == "d")
        .map(|terminal| terminal.pty.pid)
        .and_then(|pid| i32::try_from(pid).ok())
        .unwrap_or(-1);
    assert!(!pty::in_session(sleeper).is_empty(), "`sleep 1000` runs");

    app.world_mut().trigger(Exit(ExitReason::Quit));
    for _ in 0..20 {
        if app.should_exit().is_some() {
            break;
        }
        app.update();
    }
    assert!(app.should_exit().is_some(), "the app quit");
    // Killed at once; reaped a moment later.
    let since = std::time::Instant::now();
    while !pty::in_session(sleeper).is_empty() && since.elapsed() < Duration::from_secs(2) {
        std::thread::sleep(Duration::from_millis(20));
    }
    assert!(
        pty::in_session(sleeper).is_empty(),
        "nothing outlived the quit"
    );
}
