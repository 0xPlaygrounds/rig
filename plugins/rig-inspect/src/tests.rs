use std::time::{Duration, Instant};

use bevy_remote::BrpReceiver;
use rig_basics::BasicCommandsPlugin;
use rig_harness::load;
use rig_harness_test_support::start_call;
use serde_json::json;

use super::*;

/// Starts an `inspect` call of `method` with `params` by `agent`.
fn start(app: &mut App, agent: Entity, method: &str, params: Value) -> Option<Entity> {
    let args = json!({ "method": method, "params": params });
    start_call(app.world_mut(), agent, (INSPECT_TOOL, "call"), args)
}

/// The output of `call`, once it has one.
fn output(app: &mut App, call: Option<Entity>) -> String {
    let (started, call) = (Instant::now(), call.unwrap_or(Entity::PLACEHOLDER));
    while app.world().get::<Outcome>(call).is_none() && started.elapsed().as_secs() < 10 {
        app.update();
        std::thread::sleep(Duration::from_millis(2));
    }
    let output = app.world().get::<Outcome>(call);
    output
        .map(|output| output.tool_result().output().render())
        .unwrap_or_default()
}

#[test]
fn inspect_reads_the_world_and_nothing_else() {
    let spill = std::env::temp_dir().join(format!("rig-inspect-{}", std::process::id()));
    let mut app = App::new();
    app.add_plugins((TaskPoolPlugin::default(), AgentPlugin, InspectPlugin))
        .insert_resource(Scratch(Some(spill.clone())));
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
    // `ModelChoice` is saved under its stable key `rig::model`.
    let saved = json!({ "type_limit": { "with": ["Saved"] } });
    let schema = ask(&mut app, "registry.schema", saved);
    assert!(
        schema.contains("\"model\":{\"shortPath\":\"model\""),
        "{schema}"
    );

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
    assert_eq!(added, ["/help", "/quit", "/retry"], "{answer}");

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

#[test]
fn the_owner_queries_the_agents_world_through_the_channel_and_nobody_else_does() {
    use rig_harness::channel::{Caller, call};
    use rig_harness::view::protocol::{SessionId, codes};
    let mut app = App::new();
    app.add_plugins((TaskPoolPlugin::default(), AgentPlugin, InspectPlugin));
    app.world_mut().spawn((Agent, Name::new("main agent")));
    app.finish();
    app.update();
    let caller = |principal| Caller {
        client: Entity::PLACEHOLDER,
        principal,
    };
    let ask = |app: &mut App, principal, method: &str| {
        let params = json!({
            "sessionId": "s-1",
            "method": method,
            "params": { "data": { "components": ["Name"] }, "filter": { "with": ["Agent"] } },
        });
        call(
            app.world_mut(),
            InspectQuery::NAME,
            caller(principal),
            params,
        )
    };
    let answer = ask(&mut app, Principal::Owner, "world.query");
    let text = format!("{answer:?}");
    assert!(text.contains("main agent"), "{text}");
    let child = Principal::Session(SessionId("s-1".into()));
    let refused = ask(&mut app, child, "world.query");
    assert!(matches!(refused, Some(Err(e)) if e.code == codes::UNAUTHORIZED));
    let refused = ask(&mut app, Principal::Owner, "world.despawn_entity");
    assert!(matches!(refused, Some(Err(e)) if e.code == codes::INVALID_PARAMS));
}
