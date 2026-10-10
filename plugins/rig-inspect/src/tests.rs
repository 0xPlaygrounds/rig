use std::time::{Duration, Instant};

use bevy_remote::BrpReceiver;
use rig_basics::BasicCommandsPlugin;
use rig_harness::load;
use rig_harness::rig_core::message::{ToolCall, ToolFunction, ToolName};
use rig_harness::rig_ecs::turn::ToolStarter;
use serde_json::json;

use super::*;

/// Starts an `inspect` call of `method` with `params` by `agent`.
fn start(app: &mut App, agent: Entity, method: &str, params: Value) -> Option<Entity> {
    let args = json!({ "method": method, "params": params });
    let name = ToolName::new(INSPECT_TOOL).ok()?;
    let call = ToolCall::from_wire("call", ToolFunction::new(name, args));
    let world = app.world_mut();
    let start = world.register_system(move |starter: ToolStarter, mut commands: Commands| {
        let run = starter.run(call.clone(), None);
        let entity = commands.spawn(run.clone()).id();
        starter.start(&mut commands, entity, agent, &run);
        entity
    });
    world.run_system(start).ok()
}

/// The output of `call`, once it has one.
fn output(app: &mut App, call: Option<Entity>) -> String {
    let (started, call) = (Instant::now(), call.unwrap_or(Entity::PLACEHOLDER));
    while app.world().get::<ToolOutput>(call).is_none() && started.elapsed().as_secs() < 10 {
        app.update();
        std::thread::sleep(Duration::from_millis(2));
    }
    let output = app.world().get::<ToolOutput>(call);
    output
        .map(|output| output.0.output().render())
        .unwrap_or_default()
}

#[test]
fn inspect_reads_the_world_and_nothing_else() {
    let spill = std::env::temp_dir().join(format!("rig-inspect-{}", std::process::id()));
    let mut app = App::new();
    app.add_plugins((TaskPoolPlugin::default(), AgentPlugin, InspectPlugin))
        .insert_resource(Answers(Some(Spill(spill.clone()))));
    load::<BasicCommandsPlugin>(&mut app, "rig-basics", "test");
    let agent = app.world_mut().spawn(Agent).id();
    app.finish();
    app.update();
    let ask = |app: &mut App, method, params| {
        let call = start(app, agent, method, params);
        output(app, call)
    };

    // `rpc.discover` lists the read-only methods only; the registry names saved types.
    let discovered = ask(&mut app, "rpc.discover", json!({}));
    for method in READ_ONLY {
        assert!(
            discovered.contains(&format!("\"{method}\"")),
            "{discovered}"
        );
    }
    assert!(
        !discovered.contains("world.mutate_components"),
        "{discovered}"
    );
    let saved = json!({ "type_limit": { "with": ["Saved"] } });
    assert!(ask(&mut app, "registry.schema", saved).contains("\"ModelChoice\""));

    // A method that writes is refused before it reaches Bevy's mailbox.
    let params = json!({ "entity": 0, "component": "Name", "path": "", "value": "x" });
    let call = start(&mut app, agent, "world.mutate_components", params);
    assert!(app.world().resource::<BrpReceiver>().is_empty());
    assert!(output(&mut app, call).contains("not a read-only method"));

    // What the basics plugin added, asked and answered with short type names.
    let world = app.world_mut();
    let mut plugins = world.query_filtered::<Entity, With<PluginSource>>();
    let plugin = plugins.iter(world).next().map(Entity::to_bits);
    let query = json!({ "data": { "components": ["Name", "ProvidedBy"] } });
    let answer = ask(&mut app, "world.query", query);
    let rows: Vec<Value> = serde_json::from_str(&answer).unwrap_or_default();
    let mut added: Vec<&str> = rows
        .iter()
        .filter(|row| {
            row.pointer("/components/ProvidedBy")
                .and_then(Value::as_u64)
                == plugin
        })
        .filter_map(|row| row.pointer("/components/Name")?.as_str())
        .collect();
    added.sort_unstable();
    assert_eq!(added, ["/agents", "/help", "/quit", "/retry"], "{answer}");

    // A long answer is cut, and kept whole under a handle the model can read.
    for n in 0..1000 {
        app.world_mut()
            .spawn(Name::new(format!("entity {n} of many")));
    }
    let query = json!({ "data": { "components": ["Name"] } });
    let answer = ask(&mut app, "world.query", query);
    let handle = answer.rsplit("all of it is in ").next().unwrap_or_default();
    let whole = std::fs::read_to_string(handle.split(": read").next().unwrap_or_default());
    let rows: Vec<Value> = serde_json::from_str(&whole.unwrap_or_default()).unwrap_or_default();
    assert!(answer.len() < CAP + 200 && rows.len() > 1000, "{answer}");
    std::fs::remove_dir_all(&spill).ok();
}
