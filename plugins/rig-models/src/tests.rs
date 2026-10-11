use std::fs;

use rig_harness::harness_protocol::{Home, Invocation};

use super::*;

/// Whether the `route` command ran.
#[derive(Resource, Default)]
struct Routed(bool);

/// An app choosing models, with a plugin's `router` choice whose command
/// is `route on`, and an agent.
fn routed() -> (App, Entity) {
    let mut app = App::new();
    app.add_plugins((AgentPlugin, ModelsPlugin))
        .init_resource::<Routed>()
        .add_command(
            "route",
            "Route each request",
            |_: In<CommandArgs>, mut routed: ResMut<Routed>| {
                routed.0 = true;
            },
        );
    let world = app.world_mut();
    world.spawn(ModelItem::new("router", "router  per request", "route on"));
    let agent = world.spawn(Agent).id();
    (app, agent)
}

/// Runs `/line` for `agent`, with everything it triggers.
fn run(app: &mut App, agent: Entity, line: &str) {
    let world = app.world_mut();
    world.trigger(RunCommand {
        entity: agent,
        line: line.to_owned(),
    });
    world.flush();
}

#[test]
fn the_agent_starts_on_the_model_named_with_model() {
    let mut app = App::new();
    app.insert_resource(Invoked {
        args: Invocation {
            model: Some("deepseek/deepseek-flash".to_owned()),
            ..Invocation::default()
        },
        terminal: false,
    })
    .add_plugins((AgentPlugin, ModelsPlugin));
    app.update();
    app.update();
    let world = app.world_mut();
    let chosen: Vec<String> = world
        .query::<&ModelChoice>()
        .iter(world)
        .map(|choice| choice.0.clone())
        .collect();
    assert_eq!(chosen, ["deepseek/deepseek-flash"]);
}

#[test]
fn model_with_a_contributed_name_runs_its_command() {
    let (mut app, agent) = routed();
    run(&mut app, agent, "model router");
    assert!(app.world().resource::<Routed>().0);
    assert!(app.world().get::<ModelChoice>(agent).is_none());
}

#[cfg(feature = "tui")]
#[test]
fn the_picker_lists_contributed_choices_first_and_the_status_line_their_label() {
    let (mut app, agent) = routed();
    run(&mut app, agent, "model");
    let world = app.world_mut();
    let mut picks = world.resource_mut::<Messages<rig_tui::PickRequest>>();
    let items: Vec<_> = picks.drain().flat_map(|pick| pick.items).collect();
    let first = items.first().map(|item| item.command.as_str());
    assert_eq!(first, Some("route on"), "{items:?}");

    world.entity_mut(agent).insert((
        rig_tui::StatusItems::default(),
        ModelLabel("router".to_owned()),
    ));
    app.update();
    let shown = app.world().get::<rig_tui::StatusItems>(agent);
    let says = |text: &str| shown.is_some_and(|items| items.iter().any(|item| item.text == text));
    assert!(says("router"), "{shown:?}");
    app.world_mut().entity_mut(agent).remove::<ModelLabel>();
    app.update();
    let shown = app.world().get::<rig_tui::StatusItems>(agent);
    let says = |text: &str| shown.is_some_and(|items| items.iter().any(|item| item.text == text));
    assert!(says("no model: /model picks one"), "{shown:?}");
}

#[test]
fn the_last_model_chosen_is_kept_in_the_apps_home() {
    let root = std::env::temp_dir().join(format!("rig-models-{}", std::process::id()));
    fs::remove_dir_all(&root).ok();
    let home = Home::at(&root);
    let app = |home: &Home| {
        let mut app = App::new();
        app.insert_resource(RigHome(home.clone()))
            .add_plugins((AgentPlugin, DefaultsPlugin));
        app
    };
    let mut first = app(&home);
    first.insert_resource(SessionRestored);
    let world = first.world_mut();
    let agent = world.spawn((Agent, Effort::default())).id();
    world
        .entity_mut(agent)
        .insert(ModelChoice("deepseek/deepseek-flash".to_owned()));
    world.flush();
    assert!(home.defaults().is_file());

    let mut next = app(&home);
    let agent = next.world_mut().spawn(Agent).id();
    next.update();
    let chosen = next
        .world()
        .get::<ModelChoice>(agent)
        .map(|choice| choice.0.clone());
    assert_eq!(chosen.as_deref(), Some("deepseek/deepseek-flash"));
    fs::remove_dir_all(&root).ok();
}
