use std::path::Path;
use std::sync::mpsc::Receiver;

use rig_basics::coding_tools::{EditTool, ShellTool};
use rig_core::test_utils::{MockCompletionModel, MockStreamEvent};
use rig_harness_test_support::MemoryStore;
use rig_harness_test_support::{
    Ended, app_on, calls, connect, first_agent, log_ended, reply, run_until, scratch_home,
};

use super::*;

/// `lib.rs` before the first turn: a TODO that was already there.
const LIB: &str = "// TODO: old
fn main() {
    let x = 1;
    let a = 2;
    let b = 3;
    let c = 4;
    let d = 5;
    let e = 6;
    let f = 7;
    let g = 8;
}
";

/// An `edit` of `path` replacing each `(old, new)`.
fn edit(id: &str, path: &Path, edits: &[(&str, &str)]) -> Vec<MockStreamEvent> {
    let edits: Vec<_> = edits
        .iter()
        .map(|(old, new)| serde_json::json!({ "old_text": old, "new_text": new }))
        .collect();
    let args = serde_json::json!({ "path": path.display().to_string(), "edits": edits });
    calls(id, "edit", args)
}

/// Each request's text, as sent.
fn sent(model: &MockCompletionModel) -> Vec<String> {
    let requests = model.requests();
    let text = |request| serde_json::to_string(request).unwrap_or_default();
    requests.iter().map(text).collect()
}

/// Sends `text` from the user and runs until its turn ended, and a few
/// frames more, in case anything else would start.
fn prompt(app: &mut App, wakes: &Receiver<()>, agent: Entity, text: &str) {
    let ended = app.world().resource::<Ended>().0.len();
    app.world_mut()
        .trigger(Deliver::new(agent, text, DeliveryMode::Steer));
    run_until(app, wakes, |world| {
        world.resource::<Ended>().0.len() > ended
    });
    for _ in 0..20 {
        app.update();
    }
}

/// Edits by `edit` and by `shell` count; only the lines a turn added.
#[test]
fn added_leftovers_get_one_nudge_and_kept_ones_are_not_raised_again() {
    let dir = scratch_home("corpus-sn-leftover-detector")
        .root()
        .to_path_buf();
    let keep = scratch_home("corpus-sn-leftover-detector-scratch")
        .root()
        .to_path_buf();
    assert!(std::fs::create_dir_all(&dir).is_ok(), "a scratch project");
    let lib = dir.join("lib.rs");
    assert!(std::fs::write(&lib, LIB).is_ok(), "lib.rs is written");
    let app_js = dir.join("app.js");
    let js = "console.log(\"old\");\n";
    assert!(std::fs::write(&app_js, js).is_ok(), "app.js is written");
    // A work tree of its own: the build directory it is in is ignored by
    // the repository around it, which a checkpoint would honour.
    let git = std::process::Command::new("git")
        .args(["init", "-q"])
        .current_dir(&dir)
        .status();
    assert!(git.is_ok_and(|status| status.success()), "git init");

    let model = MockCompletionModel::from_stream_turns([
        // Prompt 1: adds `dbg!(x);` at line 10 and `// TODO later` at 12.
        edit(
            "1",
            &lib,
            &[
                ("    let f = 7;\n", "    let f = 7;\n    dbg!(x);\n"),
                ("    let g = 8;\n", "    let g = 8;\n    // TODO later\n"),
            ],
        ),
        reply("Fixed."),
        // The nudge: removes `dbg!(x);`, keeps the TODO.
        edit("2", &lib, &[("    dbg!(x);\n", "")]),
        reply("Removed the dbg!; the TODO stays, it marks real follow-up work."),
        // Prompt 2: adds nothing that matches.
        edit(
            "3",
            &lib,
            &[("    let g = 8;\n", "    let g = 8;\n    let y = 9;\n")],
        ),
        reply("Added y."),
        // Prompt 3: `shell` adds a log to app.js and a breakpoint in a
        // new file.
        calls(
            "5",
            "shell",
            serde_json::json!({
                "command": "printf 'console.log(y);\\n' >> app.js && printf 'breakpoint()\\n' > new.py"
            }),
        ),
        reply("Logged y."),
        reply("Both stay: the log and the breakpoint are the feature."),
        // Prompt 4, with the checks off: adds a `dbg!`.
        edit(
            "4",
            &lib,
            &[("    let y = 9;\n", "    let y = 9;\n    dbg!(y);\n")],
        ),
        reply("Debugging y."),
    ]);
    let (mut app, wakes) = app_on(&MemoryStore::default());
    app.insert_resource(ProjectRoot(dir.clone()))
        .insert_resource(Scratch(Some(keep.clone())))
        .add_plugins((
            AgentPlugin,
            SessionLogPlugin,
            EditTool,
            ShellTool,
            LeftoverDetectorPlugin,
        ))
        .init_resource::<Ended>()
        .add_observer(log_ended);
    app.finish();
    app.update();
    let agent = first_agent(app.world_mut());
    let connection = connect(&model);
    assert!(
        agent.is_some() && connection.is_some(),
        "an agent on the model"
    );
    let (Some(agent), Some(connection)) = (agent, connection) else {
        return;
    };
    app.world_mut().entity_mut(agent).insert(connection);

    prompt(&mut app, &wakes, agent, "Fix the bug.");
    let requests = sent(&model);
    assert_eq!(
        requests.len(),
        4,
        "the turn, the nudge, and no second nudge"
    );
    let nudge = requests.get(2).map(String::as_str).unwrap_or_default();
    assert_eq!(nudge.matches("- lib.rs:").count(), 2, "{nudge}");
    assert!(nudge.contains("- lib.rs:10: dbg!(x);"), "{nudge}");
    assert!(nudge.contains("- lib.rs:12: // TODO later"), "{nudge}");
    assert!(!nudge.contains("TODO: old"), "{nudge}");
    let last = requests.get(3).map(String::as_str).unwrap_or_default();
    assert_eq!(
        last.matches("- lib.rs:").count(),
        2,
        "one nudge only: {last}"
    );

    prompt(&mut app, &wakes, agent, "Add y.");
    assert_eq!(model.request_count(), 6, "no nudge without leftovers added");

    prompt(&mut app, &wakes, agent, "Log y.");
    let requests = sent(&model);
    assert_eq!(requests.len(), 9, "the turn and its nudge");
    let nudge = requests.get(8).map(String::as_str).unwrap_or_default();
    assert!(nudge.contains("- app.js:2: console.log(y);"), "{nudge}");
    assert!(nudge.contains("- new.py:1: breakpoint()"), "{nudge}");
    assert_eq!(
        nudge.matches("- app.js:").count(),
        1,
        "not the old log: {nudge}"
    );

    app.world_mut()
        .trigger(RunCommand::new(agent, "leftovers off"));
    app.update();
    prompt(&mut app, &wakes, agent, "Debug y.");
    assert_eq!(model.request_count(), 11, "no nudge with the checks off");
    std::fs::remove_dir_all(&dir).ok();
    std::fs::remove_dir_all(&keep).ok();
}

#[test]
fn stages_are_ordered_with_the_defaults() {
    rig_defaults_test_support::assert_stages_ordered(LeftoverDetectorPlugin);
}
