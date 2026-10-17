use std::path::Path;

use rig_basics::coding_tools::ReadTool;
use rig_core::test_utils::MockCompletionModel;
use rig_harness::watch::WatchPlugin;
use rig_harness_test_support::{
    Collected, Ended, MemoryStore, ToolOutputs, app_on, calls, collect, connect, log_ended,
    log_tool_outputs, reply, run_until,
};
use serde_json::json;

use super::*;

/// Writes the executable script `name` into `dir`.
fn script(dir: &Path, name: &str, text: &str) -> Option<()> {
    use std::os::unix::fs::PermissionsExt;
    fs::create_dir_all(dir).ok()?;
    let path = dir.join(name);
    fs::write(&path, text).ok()?;
    fs::set_permissions(&path, fs::Permissions::from_mode(0o755)).ok()
}

const GREET: &str = "#!/bin/sh
# rig-tool: greet
# description: Greet someone.
# arg who string required Who to greet.
# readonly
printf 'hello %s' \"$RIG_ARG_WHO\"
";

const SHOUT: &str = "#!/bin/sh
# rig-tool: shout
# description: Shout what stdin says.
# arg text string required What to shout.
tr a-z A-Z
";

const READ: &str = "#!/bin/sh
# rig-tool: read
# description: Not the real read.
cat
";

/// Whether a tool named `name` is registered.
fn has_tool(world: &World, name: &str) -> bool {
    let tools = world.try_query::<&ToolDef>();
    tools
        .iter()
        .flat_map(|query| query.iter_manual(world))
        .any(|def| def.0.name.as_str() == name)
}

/// The tool names of the model's request `index`.
fn offered(model: &MockCompletionModel, index: usize) -> Vec<String> {
    let requests = model.requests();
    let tools = requests.get(index).map(|request| request.tools.clone());
    let tools = tools.unwrap_or_default();
    tools
        .iter()
        .map(|tool| tool.name.as_str().to_owned())
        .collect()
}

#[test]
fn scripts_are_tools_that_come_and_go_with_their_files() {
    let home = rig_harness_test_support::scratch_home("corpus-ho-script-tools");
    let root = home.root().to_path_buf();
    let tools = root.join(".rig").join("tools");
    assert!(script(&tools, "greet.sh", GREET).is_some());
    assert!(script(&tools, "read.sh", READ).is_some());
    let model = MockCompletionModel::from_stream_turns([
        calls("1", "greet", json!({ "who": "x" })),
        reply("greeted"),
        calls("3", "shout", json!({ "text": "hi" })),
        reply("noted"),
        calls("2", "greet", json!({ "who": "y" })),
        reply("gone"),
    ]);
    let (mut app, wakes) = app_on(&MemoryStore::default());
    app.set_error_handler(rig_harness::error::warn)
        .insert_resource(ProjectRoot(root.clone()))
        .insert_resource(RigHome(home))
        .add_plugins((AgentPlugin, SessionLogPlugin, WatchPlugin, ReadTool))
        .add_plugins(ScriptToolsPlugin)
        .init_resource::<Ended>()
        .add_observer(log_ended)
        .init_resource::<ToolOutputs>()
        .add_observer(log_tool_outputs);
    collect::<Notice>(&mut app);
    let agent = connect(&model).map(|connection| app.world_mut().spawn((Agent, connection)).id());
    app.finish();
    app.update();
    assert!(agent.is_some(), "the catalog lists the model");
    let Some(agent) = agent else {
        return;
    };

    // A script there at startup is a tool from the first request.
    app.world_mut()
        .trigger(Deliver::new(agent, "Greet x.", DeliveryMode::Steer));
    run_until(&mut app, &wakes, |world| {
        world.resource::<Ended>().0.len() == 1
    });
    let outputs = &app.world().resource::<ToolOutputs>().0;
    assert_eq!(outputs.get("1").map(String::as_str), Some("hello x"));

    // `read` is taken by the built-in tool: one warning, and it stays.
    let notices = &app.world().resource::<Collected<Notice>>().0;
    let warned: Vec<_> = notices
        .iter()
        .filter(|notice| notice.text.contains("read.sh"))
        .collect();
    assert_eq!(warned.len(), 1, "{notices:?}");
    let read = app
        .world_mut()
        .query::<&ToolDef>()
        .iter(app.world())
        .find(|def| def.0.name.as_str() == "read")
        .map(|def| def.0.description.clone());
    assert!(read.is_some_and(|description| !description.contains("Not the real read")));

    // A script added while the app runs is offered from the next request.
    assert!(script(&tools, "shout.sh", SHOUT).is_some());
    run_until(&mut app, &wakes, |world| has_tool(world, "shout"));
    app.world_mut()
        .trigger(Deliver::new(agent, "Anything new?", DeliveryMode::Steer));
    run_until(&mut app, &wakes, |world| {
        world.resource::<Ended>().0.len() == 2
    });
    let third = offered(&model, 2);
    assert!(third.contains(&"shout".to_owned()), "{third:?}");
    assert!(third.contains(&"greet".to_owned()), "{third:?}");
    let outputs = &app.world().resource::<ToolOutputs>().0;
    let shouted = outputs.get("3").map(String::as_str);
    assert_eq!(
        shouted,
        Some(r#"{"TEXT":"HI"}"#),
        "the arguments come on stdin"
    );

    // A script removed is gone from the next request, and calling it is
    // calling a tool that does not exist.
    assert!(fs::remove_file(tools.join("greet.sh")).is_ok());
    run_until(&mut app, &wakes, |world| !has_tool(world, "greet"));
    app.world_mut()
        .trigger(Deliver::new(agent, "Greet y.", DeliveryMode::Steer));
    run_until(&mut app, &wakes, |world| {
        world.resource::<Ended>().0.len() == 3
    });
    let fourth = offered(&model, 4);
    assert!(!fourth.contains(&"greet".to_owned()), "{fourth:?}");
    let outputs = &app.world().resource::<ToolOutputs>().0;
    let output = outputs.get("2").cloned().unwrap_or_default();
    assert!(output.contains("no tool named `greet`"), "{output}");
}
