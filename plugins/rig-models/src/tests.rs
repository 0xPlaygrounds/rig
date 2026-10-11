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
    world.trigger(RunCommand::new(agent, line.to_owned()));
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
    let mut pickers = world.query::<&rig_harness::pick::Picker>();
    let items: Vec<_> = pickers
        .iter(world)
        .flat_map(|picker| picker.items.clone())
        .collect();
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

#[cfg(feature = "tui")]
#[test]
fn a_model_without_its_key_is_listed_with_why_and_offered_as_the_argument_once_typed() {
    use std::borrow::Cow;

    use rig_harness::rig_core::catalog::{Catalog, Connector, ModelSpec};
    use rig_harness::rig_core::providers::registry::{ConnectError, ProviderId};

    let Ok(corp) = ProviderId::declare("rig-models-corp") else {
        return;
    };
    let mut catalog = Catalog::default();
    catalog.insert(ModelSpec::new(corp, "big").with_tools(true));
    let mut connector = Connector::new(catalog);
    connector.provide(corp, |_: &ModelSpec, _: &Catalog| {
        Err(ConnectError::MissingKey {
            vendor: "rig-models-corp".to_owned(),
            tried: vec![Cow::Borrowed("CORP_KEY")],
        })
    });
    let (mut app, agent) = routed();
    app.insert_resource(Models(connector))
        .init_resource::<rig_tui::TuiInput>();
    app.update();
    let mut completions = app.world_mut().query::<&rig_tui::ArgCompletion>();
    // Listing the catalog waits for the input to be a `/model` command.
    assert_eq!(completions.iter(app.world()).count(), 0);
    app.world_mut()
        .resource_mut::<rig_tui::TuiInput>()
        .set("/model rig");
    app.update();
    run(&mut app, agent, "model");
    let world = app.world_mut();
    let mut pickers = world.query::<&rig_harness::pick::Picker>();
    let labels: Vec<String> = pickers
        .iter(world)
        .flat_map(|picker| picker.items.iter().map(|item| item.label.clone()))
        .collect();
    let listed = labels
        .iter()
        .find(|label| label.starts_with("rig-models-corp/big"));
    let listed = listed.map(String::as_str);
    assert!(
        listed.is_some_and(|label| label.ends_with("(no key: set CORP_KEY)")),
        "{labels:?}"
    );
    let offered: Vec<String> = completions
        .iter(world)
        .flat_map(|completion| completion.candidates.iter().map(|item| item.text.clone()))
        .collect();
    assert_eq!(offered, ["router", "rig-models-corp/big"]);
}
