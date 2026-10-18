use std::path::PathBuf;

use corpus_shared::plan_mode::{PlanMode, PlanModePlugin, REFUSED};
use rig_basics::coding_tools::EditTool;
use rig_core::catalog::{Catalog, Connector};
use rig_core::test_utils::MockCompletionModel;
use rig_harness::prelude::*;
use rig_harness_test_support::{
    MemoryStore, ToolOutputs, app_on, calls, connect, log_tool_outputs, reply, run_until,
    start_call,
};
use serde_json::json;

use super::StructuredConfigEditPlugin;

const CARGO: &str = "[package]\nname = \"demo\"\nversion = \"0.1.0\"\n\n\
    # Runtime dependencies.\n[dependencies] # keep sorted\n\
    serde = { version = \"1.0.100\", features = [\"derive\"] }\n\
    anyhow = \"1\"   # errors\n";

const PACKAGE: &str = "{\n    \"name\": \"demo\",\n    \"scripts\": {\n        \"build\": \"tsc\",\n        \
    \"lint\": \"eslint .\"\n    },\n    \"version\": \"1.0.0\"\n}\n";

const CI: &str = "name: CI\non: [push]\n\njobs:\n  test:\n    runs-on: ubuntu-latest  # the default\n    \
    steps:\n      - uses: actions/checkout@v4\n      - name: Test\n        run: cargo test --workspace\n      \
    # lint later\n";

const BROKEN: &str = "name: demo\njobs:\n  test: run: x\n";

/// A directory of its own for `name`, beside the test binary, holding the
/// fixtures.
fn project(name: &str) -> PathBuf {
    let exe = std::env::current_exe().unwrap_or_default();
    let dir = exe.with_file_name(name);
    std::fs::remove_dir_all(&dir).ok();
    std::fs::create_dir_all(dir.join(".github")).ok();
    for (file, text) in [
        ("Cargo.toml", CARGO),
        ("package.json", PACKAGE),
        (".github/ci.yml", CI),
        ("broken.yml", BROKEN),
    ] {
        std::fs::write(dir.join(file), text).ok();
    }
    dir
}

#[test]
fn values_change_by_key_path_and_nothing_else_moves() {
    let dir = project("sp-config-edit-scripted");
    let model = MockCompletionModel::from_stream_turns([
        calls(
            "c1",
            "config_set",
            json!({ "path": "Cargo.toml", "key": "dependencies.serde.version", "value": "1.0.200" }),
        ),
        calls(
            "c2",
            "config_set",
            json!({ "path": "package.json", "key": "scripts.test", "value": "cargo test", "create": true }),
        ),
        calls(
            "c3",
            "config_get",
            json!({ "path": ".github/ci.yml", "key": "jobs.test.steps[1].run" }),
        ),
        calls(
            "c4",
            "config_set",
            json!({ "path": ".github/ci.yml", "key": "jobs.test.steps[1].run", "value": "cargo test --all" }),
        ),
        calls(
            "c5",
            "config_set",
            json!({ "path": "broken.yml", "key": "name", "value": "x" }),
        ),
        calls(
            "c6",
            "config_set",
            json!({ "path": "package.json", "key": "name.first", "value": "x", "create": true }),
        ),
        calls(
            "c7",
            "config_delete",
            json!({ "path": "Cargo.toml", "key": "dependencies.anyhow" }),
        ),
        reply("Bumped serde and added the test script."),
    ]);
    let (mut app, wakes) = app_on(&MemoryStore::default());
    app.insert_resource(Models(Connector::new(Catalog::default())))
        .insert_resource(ProjectRoot(dir.clone()))
        .init_resource::<ToolOutputs>()
        .add_observer(log_tool_outputs)
        .add_plugins((AgentPlugin, StructuredConfigEditPlugin));
    let mut agent = app.world_mut().spawn(Agent);
    if let Some(connection) = connect(&model) {
        let choice = ModelChoice(connection.spec.reference());
        agent.insert((connection, choice));
    }
    let agent = agent.id();
    app.finish();
    app.update();
    app.world_mut().trigger(Deliver::new(
        agent,
        "Bump serde to 1.0.200 and add a test script.",
        DeliveryMode::Queue,
    ));
    run_until(&mut app, &wakes, |world| {
        model.request_count() == 8 && !world.entity(agent).contains::<ActiveTurn>()
    });
    let outputs = &app.world().resource::<ToolOutputs>().0;
    let output = |id: &str| outputs.get(id).cloned().unwrap_or_default();
    let read = |file: &str| std::fs::read_to_string(dir.join(file)).unwrap_or_default();

    // TOML: only the version changed, comments and the inline table kept,
    // then `anyhow` and its comment removed.
    assert!(
        output("c1").contains("+serde = { version = \"1.0.200\""),
        "{}",
        output("c1")
    );
    let bumped = CARGO.replace("1.0.100", "1.0.200");
    assert_eq!(
        read("Cargo.toml"),
        bumped.replace("anyhow = \"1\"   # errors\n", "")
    );
    assert!(
        output("c7").contains("-anyhow = \"1\"   # errors"),
        "{}",
        output("c7")
    );

    // JSON: 4-space indent, key order, the new script after the others.
    let added = PACKAGE.replace(
        "\"lint\": \"eslint .\"\n",
        "\"lint\": \"eslint .\",\n        \"test\": \"cargo test\"\n",
    );
    assert_eq!(read("package.json"), added);
    assert!(
        output("c2").contains("+        \"test\": \"cargo test\""),
        "{}",
        output("c2")
    );

    // YAML: read as JSON, then one line changed.
    assert_eq!(output("c3").trim(), "\"cargo test --workspace\"");
    assert_eq!(read(".github/ci.yml"), CI.replace("--workspace", "--all"));

    // A file that does not parse is refused where it breaks, untouched.
    assert!(output("c5").contains("line 3"), "{}", output("c5"));
    assert_eq!(read("broken.yml"), BROKEN);

    // A key under a string is a type error.
    assert!(output("c6").contains("type error"), "{}", output("c6"));
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn a_read_only_mode_refuses_config_set_as_it_refuses_edit_and_lets_config_get_read() {
    let dir = project("sp-config-edit-read-only");
    let (mut app, wakes) = app_on(&MemoryStore::default());
    app.insert_resource(ProjectRoot(dir.clone()))
        .init_resource::<ToolOutputs>()
        .add_observer(log_tool_outputs)
        .add_plugins((
            AgentPlugin,
            EditTool,
            PlanModePlugin,
            StructuredConfigEditPlugin,
        ));
    app.finish();
    app.update();
    let world = app.world_mut();
    let agent = world.spawn((Agent, PlanMode)).id();
    let edit = json!({ "path": "Cargo.toml", "edits": [{ "old_text": "1.0.100", "new_text": "1.0.200" }] });
    let set =
        json!({ "path": "Cargo.toml", "key": "dependencies.serde.version", "value": "1.0.200" });
    let get = json!({ "path": "Cargo.toml", "key": "dependencies.serde.features" });
    start_call(world, agent, ("edit", "e1"), edit);
    start_call(world, agent, ("config_set", "s1"), set);
    start_call(world, agent, ("config_get", "g1"), get);
    run_until(&mut app, &wakes, |world| {
        world.resource::<ToolOutputs>().0.len() == 3
    });
    let outputs = &app.world().resource::<ToolOutputs>().0;
    let output = |id: &str| outputs.get(id).cloned().unwrap_or_default();
    assert!(output("e1").contains(REFUSED), "{}", output("e1"));
    assert!(output("s1").contains(REFUSED), "{}", output("s1"));
    assert!(output("g1").contains("\"derive\""), "{}", output("g1"));
    assert_eq!(
        std::fs::read_to_string(dir.join("Cargo.toml")).unwrap_or_default(),
        CARGO
    );
    std::fs::remove_dir_all(&dir).ok();
}
