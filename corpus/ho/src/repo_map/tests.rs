use std::path::PathBuf;
use std::process::Command;
use std::sync::mpsc::Receiver;

use rig_basics::coding_tools::{EditTool, ReadTool};
use rig_core::test_utils::MockCompletionModel;
use rig_harness_test_support::{
    Ended, MemoryStore, ToolOutputs, app_on, calls, connect, first_agent, log_ended,
    log_tool_outputs, reply, run_until,
};
use serde_json::json;

use super::*;

/// A git project beside the test's binary with `a.rs`, `b.rs` and `c.rs`.
fn project() -> std::io::Result<PathBuf> {
    let dir = std::env::current_exe()?.with_file_name("repo-map-project");
    std::fs::remove_dir_all(&dir).ok();
    std::fs::create_dir_all(&dir)?;
    let dir = std::fs::canonicalize(dir)?;
    std::fs::write(dir.join("a.rs"), "fn alpha() {}\n")?;
    std::fs::write(
        dir.join("b.rs"),
        "pub struct Beta;\n\nimpl Beta {\n    fn new() {}\n}\n",
    )?;
    std::fs::write(dir.join("c.rs"), "pub fn gamma() {}\n")?;
    for args in [&["init", "-q"][..], &["add", "."][..]] {
        let status = Command::new("git").args(args).current_dir(&dir).status()?;
        if !status.success() {
            return Err(std::io::Error::other("git failed"));
        }
    }
    Ok(dir)
}

fn turns_ended(app: &mut App, wakes: &Receiver<()>, count: usize) {
    run_until(app, wakes, |world| {
        world.resource::<Ended>().0.len() >= count
    });
}

#[test]
fn the_map_is_in_the_prompt_ranked_by_work_and_kept_per_turn() {
    let project = project();
    assert!(project.is_ok(), "{project:?}");
    let Ok(project) = project else {
        return;
    };
    let store = MemoryStore::default();
    let (mut app, wakes) = app_on(&store);
    app.insert_resource(ProjectRoot(project.clone()))
        .add_plugins((
            AgentPlugin,
            SessionLogPlugin,
            ReadTool,
            EditTool,
            RepoMapPlugin,
        ))
        .init_resource::<ToolOutputs>()
        .init_resource::<Ended>()
        .add_observer(log_tool_outputs)
        .add_observer(log_ended);
    app.finish();
    app.update();
    run_until(&mut app, &wakes, |world| {
        world.resource::<RepoMap>().is_built()
    });

    let c = project.join("c.rs").to_string_lossy().into_owned();
    let model = MockCompletionModel::from_stream_turns([
        calls("1", "read", json!({ "path": c })),
        calls(
            "2",
            "edit",
            json!({ "path": c, "edits": [{
                "old_text": "pub fn gamma()",
                "new_text": "pub fn gamma_new() {}\n\npub fn gamma()",
            }] }),
        ),
        reply("done"),
        calls("3", "repo_map", json!({ "symbol": "alpha" })),
        reply("found"),
    ]);
    let world = app.world_mut();
    let (agent, connection) = (first_agent(world), connect(&model));
    assert!(agent.is_some() && connection.is_some());
    let (Some(agent), Some(connection)) = (agent, connection) else {
        return;
    };
    world.entity_mut(agent).insert(connection);
    world.trigger(Deliver::new(agent, "Look at c.", DeliveryMode::Queue));
    turns_ended(&mut app, &wakes, 1);
    // The edit's file is scanned again in the background.
    let rescanned = |world: &World| {
        let symbols = world
            .resource::<RepoMap>()
            .symbols("c.rs")
            .unwrap_or_default();
        symbols.iter().any(|symbol| symbol.name == "gamma_new")
    };
    run_until(&mut app, &wakes, rescanned);
    app.world_mut()
        .trigger(Deliver::new(agent, "Where is alpha?", DeliveryMode::Queue));
    turns_ended(&mut app, &wakes, 2);
    std::fs::remove_dir_all(&project).ok();

    let preambles: Vec<String> = model
        .requests()
        .iter()
        .map(|request| serde_json::to_string(&request.chat_history.first()).unwrap_or_default())
        .collect();
    assert_eq!(preambles.len(), 5, "{preambles:?}");
    let (Some(first), Some(second)) = (preambles.first(), preambles.get(3)) else {
        return;
    };
    assert!(
        first.contains("<repo_map>") && first.contains("fn alpha"),
        "{first}"
    );
    assert!(!first.contains("gamma_new"), "{first}");
    // Turn 1's requests share their prompt, though it read and edited c.rs.
    assert!(preambles.iter().take(3).all(|preamble| preamble == first));
    let at = |name: &str| second.find(name).unwrap_or(usize::MAX);
    assert!(
        at("c.rs") < at("a.rs") && at("c.rs") < at("b.rs"),
        "{second}"
    );
    assert!(second.contains("gamma_new"), "{second}");
    let outputs = &app.world().resource::<ToolOutputs>().0;
    let found = outputs.get("3").map(String::as_str).unwrap_or_default();
    assert!(found.contains("a.rs:1"), "{found}");
}
