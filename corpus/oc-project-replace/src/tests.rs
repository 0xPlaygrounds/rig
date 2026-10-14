use std::fs;
use std::path::Path;
use std::sync::mpsc::Receiver;

use rig_core::test_utils::MockCompletionModel;
use rig_harness_test_support::{
    Ended, MemoryStore, ToolOutputs, app_on, calls, connect, first_agent, log_ended,
    log_tool_outputs, reply, run_until, scratch_home,
};
use serde_json::json;

use super::*;

/// The paths each settled call changed, by call id, from its `FilesChanged`.
#[derive(Resource, Default)]
struct Changed(Vec<(String, usize)>);

fn log_changed(
    settled: On<Add<Settled>>,
    calls: Query<(&ToolRequest, &Published)>,
    mut log: ResMut<Changed>,
) {
    if let Ok((request, Published(published))) = calls.get(settled.entity)
        && let Ok(Some(FilesChanged(changes))) = published.get::<FilesChanged>()
    {
        log.0.push((request.call.id.to_string(), changes.len()));
    }
}

fn write(root: &Path, path: &str, bytes: &[u8]) {
    let path = root.join(path);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).ok();
    }
    fs::write(path, bytes).ok();
}

fn read(root: &Path, path: &str) -> String {
    fs::read_to_string(root.join(path)).unwrap_or_default()
}

fn set_read_only(root: &Path, path: &str, read_only: bool) {
    let path = root.join(path);
    if let Ok(meta) = fs::metadata(&path) {
        let mut permissions = meta.permissions();
        permissions.set_readonly(read_only);
        fs::set_permissions(path, permissions).ok();
    }
}

/// Sends `agent` a message, runs its turn to its end, and returns what
/// call `id` answered.
fn turn(app: &mut App, wakes: &Receiver<()>, agent: Entity, id: &str) -> String {
    let ended = app.world().resource::<Ended>().0.len();
    app.world_mut()
        .trigger(Deliver::new(agent, "Go on.", DeliveryMode::Steer));
    run_until(app, wakes, |world| {
        world.resource::<Ended>().0.len() > ended
    });
    let outputs = &app.world().resource::<ToolOutputs>().0;
    outputs.get(id).cloned().unwrap_or_default()
}

const A: &str = "fn old_name() {}\nfn load_v1() { old_name(); }\nconst X: &str = \"old_name\";\n";
const B: &str = "use crate::old_name;\n";

#[test]
fn previews_applies_all_or_nothing_and_undoes() {
    let root = scratch_home("oc-project-replace").root().to_owned();
    fs::create_dir_all(root.join(".git")).ok();
    write(&root, ".gitignore", b"target/\ngenerated.rs\n");
    write(&root, "src/a.rs", A.as_bytes());
    write(&root, "src/b.rs", B.as_bytes());
    write(&root, "src/generated.rs", b"old_name\n");
    write(&root, "target/gen.rs", b"old_name old_name\n");
    write(&root, "logo.png", b"\x89PNG\0\0old_name\0");

    let rename = json!({ "pattern": "old_name", "replacement": "new_name" });
    let rename_apply = json!({ "pattern": "old_name", "replacement": "new_name", "apply": true });
    let regex = json!({
        "pattern": "fn (\\w+)_v1", "replacement": "fn ${1}_v2", "regex": true, "apply": true,
    });
    let again = json!({ "pattern": "new_name", "replacement": "newer_name" });
    let again_apply = json!({ "pattern": "new_name", "replacement": "newer_name", "apply": true });
    let model = MockCompletionModel::from_stream_turns([
        calls("p1", REPLACE_IN_FILES, rename),
        reply("Previewed."),
        calls("a1", REPLACE_IN_FILES, rename_apply),
        reply("Applied."),
        calls("r1", REPLACE_IN_FILES, regex),
        reply("Renamed."),
        calls("c1", REPLACE_IN_FILES, again_apply.clone()),
        reply("Failed."),
        calls("p2", REPLACE_IN_FILES, again),
        reply("Previewed."),
        calls("c2", REPLACE_IN_FILES, again_apply),
        reply("Failed."),
    ]);

    let (mut app, wakes) = app_on(&MemoryStore::default());
    app.add_plugins((AgentPlugin, SessionLogPlugin, ProjectReplacePlugin))
        .init_resource::<ToolOutputs>()
        .add_observer(log_tool_outputs)
        .init_resource::<Changed>()
        .add_observer(log_changed)
        .init_resource::<Ended>()
        .add_observer(log_ended);
    app.finish();
    app.update();
    let world = app.world_mut();
    let agent = first_agent(world).unwrap_or(Entity::PLACEHOLDER);
    if let Some(connection) = connect(&model) {
        world
            .entity_mut(agent)
            .insert((connection, WorkDir(root.clone())));
    }

    // A preview counts per file, skips ignored, target and binary files,
    // and writes nothing.
    let preview = turn(&mut app, &wakes, agent, "p1");
    assert!(preview.contains("src/a.rs: 3"), "{preview}");
    assert!(preview.contains("src/b.rs: 1"), "{preview}");
    assert!(preview.contains("- use crate::old_name;"), "{preview}");
    assert!(preview.contains("+ use crate::new_name;"), "{preview}");
    assert!(preview.contains("Total: 4 matches in 2 files"), "{preview}");
    for skipped in ["target", "generated", "logo"] {
        assert!(!preview.contains(skipped), "{preview}");
    }
    assert_eq!(read(&root, "src/a.rs"), A);
    assert_eq!(read(&root, "src/b.rs"), B);

    // Applying writes both files in one call that reports both changes.
    let applied = turn(&mut app, &wakes, agent, "a1");
    assert!(
        applied.contains("Changed 2 files, 4 replacements"),
        "{applied}"
    );
    assert!(applied.contains("/replace-undo"), "{applied}");
    let renamed = A.replace("old_name", "new_name");
    assert_eq!(read(&root, "src/a.rs"), renamed);
    assert_eq!(read(&root, "src/b.rs"), "use crate::new_name;\n");
    assert_eq!(read(&root, "target/gen.rs"), "old_name old_name\n");
    assert_eq!(read(&root, "src/generated.rs"), "old_name\n");
    let changed = &app.world().resource::<Changed>().0;
    assert!(changed.contains(&("a1".to_owned(), 2)), "{changed:?}");

    // A regex replacement fills in its captures.
    let captured = turn(&mut app, &wakes, agent, "r1");
    assert!(captured.contains("Changed 1 files"), "{captured}");
    assert!(
        read(&root, "src/a.rs").contains("fn load_v2()"),
        "{captured}"
    );

    // `/replace-undo` puts back what the last apply changed.
    app.world_mut()
        .trigger(RunCommand::new(agent, "replace-undo".to_owned()));
    run_until(&mut app, &wakes, |_| read(&root, "src/a.rs") == renamed);
    assert_eq!(read(&root, "src/a.rs"), renamed);

    // One file that cannot be written leaves every file as it was.
    set_read_only(&root, "src/b.rs", true);
    let refused = turn(&mut app, &wakes, agent, "c1");
    set_read_only(&root, "src/b.rs", false);
    assert!(refused.contains("src/b.rs is read-only"), "{refused}");
    assert_eq!(read(&root, "src/a.rs"), renamed);

    // A file changed since the preview stops the apply.
    let preview = turn(&mut app, &wakes, agent, "p2");
    assert!(preview.contains("Total: 4 matches"), "{preview}");
    write(
        &root,
        "src/a.rs",
        format!("{renamed}// edited\n").as_bytes(),
    );
    let stale = turn(&mut app, &wakes, agent, "c2");
    assert!(
        stale.contains("src/a.rs changed since the preview"),
        "{stale}"
    );
    assert_eq!(read(&root, "src/b.rs"), "use crate::new_name;\n");
}

#[test]
fn stages_are_ordered_with_the_defaults() {
    rig_defaults_test_support::assert_stages_ordered(ProjectReplacePlugin);
}
