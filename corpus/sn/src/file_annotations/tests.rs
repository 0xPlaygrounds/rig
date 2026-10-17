use std::path::{Path, PathBuf};
use std::sync::mpsc::Receiver;

use rig_coding_tools::ReadTool;
use rig_core::test_utils::{MockCompletionModel, MockStreamEvent};
use rig_harness::harness_protocol::{Home, SessionId};
use rig_harness_test_support::{
    Ended, ToolOutputs, app_in, calls, connect, log_ended, log_tool_outputs, reply, run_until,
    scratch_home,
};
use serde_json::json;

use super::*;

/// A new session of the project in `dir`, under `home`, with the `read`
/// tool and the plugin, its agent talking to `model`.
fn session(home: &Home, dir: &Path, model: &MockCompletionModel) -> (App, Receiver<()>, Entity) {
    let (mut app, wakes) = app_in(home, &SessionId::generate());
    app.insert_resource(ProjectRoot(dir.to_path_buf()))
        .add_plugins((
            AgentPlugin,
            SessionLogPlugin,
            ReadTool,
            FileAnnotationsPlugin,
        ))
        .init_resource::<Ended>()
        .add_observer(log_ended)
        .init_resource::<ToolOutputs>()
        .add_observer(log_tool_outputs);
    app.finish();
    app.update();
    let world = app.world_mut();
    let agent = world
        .query_filtered::<Entity, With<Agent>>()
        .iter(world)
        .next();
    let agent = agent.unwrap_or(Entity::PLACEHOLDER);
    if let Some(connection) = connect(model) {
        world.entity_mut(agent).insert(connection);
    }
    (app, wakes, agent)
}

/// Runs one turn of `agent`, its `turns`-th.
fn turn(app: &mut App, wakes: &Receiver<()>, agent: Entity, turns: usize) {
    app.world_mut()
        .trigger(Deliver::new(agent, "Go on.", DeliveryMode::Steer));
    run_until(app, wakes, |world| {
        world.resource::<Ended>().0.len() >= turns
    });
}

/// What call `id` returned.
fn output(app: &App, id: &str) -> String {
    let outputs = &app.world().resource::<ToolOutputs>().0;
    outputs.get(id).cloned().unwrap_or_default()
}

/// The line of a read's output after file line `number` of a read from
/// line 1, counting the file's lines, not the notes between them.
fn after_line(output: &str, number: usize) -> Option<&str> {
    let mut lines = output.lines();
    let mut counted = 0;
    lines.find(|line| {
        counted += usize::from(!line.starts_with("  ┆"));
        counted == number
    })?;
    lines.next()
}

fn read(id: &str, lines: Option<(usize, usize)>) -> Vec<MockStreamEvent> {
    let args = match lines {
        Some((offset, limit)) => json!({ "path": "net.rs", "offset": offset, "limit": limit }),
        None => json!({ "path": "net.rs" }),
    };
    calls(id, "read", args)
}

#[test]
fn notes_follow_the_code_across_sessions_and_go_stale() {
    let home = scratch_home("sn-file-annotations-home");
    let dir: PathBuf = home.root().with_file_name("sn-file-annotations-project");
    std::fs::remove_dir_all(&dir).ok();
    assert!(std::fs::create_dir_all(&dir).is_ok());
    let dir = std::fs::canonicalize(&dir).unwrap_or(dir);
    let file = dir.join("net.rs");
    let lines: Vec<String> = (1..=10)
        .map(|n| format!("let line_{n} = {n}; // distinctive {n}"))
        .collect();
    assert!(std::fs::write(&file, lines.join("\n") + "\n").is_ok());
    let clean = |step: &str| {
        let bytes = std::fs::read_to_string(&file).unwrap_or_default();
        assert!(!bytes.contains("retry is intentional"), "{step}: {bytes}");
    };

    // Session 1: the agent notes lines 4-5.
    let note = json!({ "path": "net.rs", "lines": "4-5", "note": "retry is intentional" });
    let first =
        MockCompletionModel::from_stream_turns([calls("a", "annotate", note), reply("noted")]);
    let (mut app, wakes, agent) = session(&home, &dir, &first);
    turn(&mut app, &wakes, agent, 1);
    let added = output(&app, "a");
    assert!(added.contains("Added note n1"), "{added}");
    clean("annotated");
    drop(app);

    // Session 2: a new app on the same project.
    let second = MockCompletionModel::from_stream_turns([
        read("r1", None),
        reply("read"),
        read("r2", None),
        reply("read again"),
        read("r3", Some((1, 3))),
        reply("read the top"),
        read("r4", None),
        read("r5", None),
        reply("read twice"),
    ]);
    let (mut app, wakes, agent) = session(&home, &dir, &second);
    turn(&mut app, &wakes, agent, 1);
    let r1 = output(&app, "r1");
    let note_line = after_line(&r1, 5).unwrap_or_default();
    assert!(
        note_line.contains("┆ note n1") && note_line.contains("retry is intentional"),
        "{r1}"
    );
    assert!(
        after_line(&r1, 4).is_some_and(|line| line.starts_with("let line_5 ")),
        "{r1}"
    );
    clean("read");

    // Three lines inserted at the top, outside rig.
    let moved = format!("// a\n// b\n// c\n{}\n", lines.join("\n"));
    assert!(std::fs::write(&file, &moved).is_ok());
    turn(&mut app, &wakes, agent, 2);
    let r2 = output(&app, "r2");
    assert!(
        after_line(&r2, 8).is_some_and(|line| line.contains("retry is intentional")),
        "{r2}"
    );
    clean("moved");

    turn(&mut app, &wakes, agent, 3);
    let r3 = output(&app, "r3");
    assert!(
        !r3.contains("retry is intentional") && r3.starts_with("// a\n// b\n// c\n"),
        "{r3}"
    );

    // The anchored lines deleted.
    let kept: Vec<&String> = lines
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != 3 && *i != 4)
        .map(|(_, line)| line)
        .collect();
    let kept: Vec<&str> = kept.iter().map(|line| line.as_str()).collect();
    assert!(std::fs::write(&file, format!("// a\n// b\n// c\n{}\n", kept.join("\n"))).is_ok());
    turn(&mut app, &wakes, agent, 4);
    let (r4, r5) = (output(&app, "r4"), output(&app, "r5"));
    assert_eq!(
        r4.matches("stale note n1 (was at lines 7–8): retry is intentional")
            .count(),
        1,
        "{r4}"
    );
    assert!(r4.trim_end().ends_with("retry is intentional"), "{r4}");
    assert!(!r5.contains("retry is intentional"), "{r5}");
    clean("deleted");
    std::fs::remove_dir_all(&dir).ok();
    std::fs::remove_dir_all(home.root()).ok();
}

#[test]
fn stages_are_ordered_with_the_defaults() {
    rig_defaults_test_support::assert_stages_ordered(FileAnnotationsPlugin);
}
