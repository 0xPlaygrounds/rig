use std::fs;
use std::sync::mpsc::Receiver;

use rig_core::test_utils::{MockCompletionModel, MockStreamEvent};
use rig_harness::harness_protocol::{Home, Invocation, SessionId};
use rig_harness::rig_core::catalog::Catalog;
use rig_harness_test_support::{
    Ended, app_in, first_agent, log_ended, reply, run_until, scratch_home, scripted_models,
};

use super::defaults::DefaultsPlugin;
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

/// Every notice written so far.
#[derive(Resource, Default)]
struct Told(Vec<String>);

#[test]
fn model_waits_for_the_turn_a_resumed_session_continues() {
    let mut app = App::new();
    app.insert_resource(Invoked {
        args: Invocation {
            model: Some("deepseek/deepseek-flash".to_owned()),
            ..Invocation::default()
        },
        terminal: false,
    })
    .add_plugins((AgentPlugin, ModelsPlugin));
    tell(&mut app);
    // The continued turn, there before the first frame.
    let world = app.world_mut();
    let agent = world.spawn(Agent).id();
    let turn = world.spawn(TurnOf(agent)).id();
    app.update();
    let told = &app.world().resource::<Told>().0;
    assert_eq!(
        told,
        &["--model deepseek/deepseek-flash applies once the current turn ends."]
    );
    assert_eq!(app.world().get::<ModelChoice>(agent), None);
    app.world_mut().entity_mut(turn).despawn();
    app.update();
    app.update();
    let chosen = app
        .world()
        .get::<ModelChoice>(agent)
        .map(|choice| choice.0.as_str());
    assert_eq!(chosen, Some("deepseek/deepseek-flash"));
    // Said once; what the model's connection says (no key, say) is its own.
    let told = &app.world().resource::<Told>().0;
    let waits = told.iter().filter(|text| text.starts_with("--model "));
    assert_eq!(waits.count(), 1, "{told:?}");
}

/// Keeps every notice written in [`Told`].
fn tell(app: &mut App) {
    app.init_resource::<Told>().add_systems(
        Last,
        |mut read: MessageReader<Notice>, mut told: ResMut<Told>| {
            told.0.extend(read.read().map(|notice| notice.text.clone()));
        },
    );
}

#[test]
fn an_unknown_model_given_with_model_leaves_the_agent_without_one() {
    let mut app = App::new();
    app.insert_resource(Invoked {
        args: Invocation {
            model: Some("deepseek/deepseek-flsh".to_owned()),
            ..Invocation::default()
        },
        terminal: false,
    })
    .add_plugins((AgentPlugin, ModelsPlugin));
    tell(&mut app);
    // The remembered model, as `defaults.json` gives it.
    app.add_systems(
        PostStartup,
        |agents: Query<Entity, With<Agent>>, mut commands: Commands| {
            for agent in &agents {
                let remembered = ModelChoice("deepseek/deepseek-flash".to_owned());
                commands.entity(agent).insert(remembered);
            }
        },
    );
    let agent = app.world_mut().spawn(Agent).id();
    app.update();
    app.update();
    let world = app.world();
    assert_eq!(world.get::<ModelChoice>(agent), None);
    assert!(world.get::<Connection>(agent).is_none());
    let told = &world.resource::<Told>().0;
    let refused: Vec<&String> = told
        .iter()
        .filter(|text| text.contains("deepseek-flsh"))
        .collect();
    assert_eq!(
        refused,
        [
            "--model: the catalog lists no model `deepseek/deepseek-flsh`; did you mean one \
          of `deepseek/deepseek-flash`, `deepseek/deepseek-v4-pro`? No model is picked."
        ],
        "{told:?}"
    );
}

#[test]
fn an_unknown_model_keeps_the_current_one_and_says_so() {
    let (mut app, agent) = routed();
    let current = ModelChoice("deepseek/deepseek-flash".to_owned());
    app.world_mut().entity_mut(agent).insert(current.clone());
    run(&mut app, agent, "model deepseek/deepseek-flsh");
    let world = app.world();
    assert_eq!(world.get::<ModelChoice>(agent), Some(&current));
    // A refused command line is recalled, with why.
    let why: Vec<Option<&str>> = world
        .resource::<Messages<Recalled>>()
        .iter_current_update_messages()
        .map(|recalled| recalled.why.as_deref())
        .collect();
    assert_eq!(
        why,
        [Some(
            "Not picked: the catalog lists no model `deepseek/deepseek-flsh`; did you mean one \
              of `deepseek/deepseek-flash`, `deepseek/deepseek-v4-pro`? Still on \
              deepseek/deepseek-flash."
        )]
    );
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

/// What `defaults.json` holds before each test: a model no test chooses.
const SAVED: &str = r#"{"model":"corpus-test/saved","effort":null}"#;

/// An app in a scratch home named `name` whose saved default is [`SAVED`],
/// choosing among scripted `corpus-test/saved` and `corpus-test/chosen`,
/// which answers as `chosen` does, with the defaults plugin; `-m` names
/// `invoked`. Its first frame ran; returns its loop's wakes, the home and
/// the agent.
fn remembering(
    name: &str,
    chosen: &MockCompletionModel,
    invoked: Option<&str>,
) -> (App, Receiver<()>, Home, Entity) {
    let saved = MockCompletionModel::from_stream_turns([reply("saved")]);
    let home = scratch_home(name);
    fs::create_dir_all(home.root()).ok();
    fs::write(home.defaults(), SAVED).ok();
    let (mut app, wakes) = app_in(&home, &SessionId::generate());
    let served = [
        ("corpus-test/saved", &saved),
        ("corpus-test/chosen", chosen),
    ];
    app.insert_resource(scripted_models(served))
        .insert_resource(Invoked {
            args: Invocation {
                model: invoked.map(str::to_owned),
                ..Invocation::default()
            },
            terminal: false,
        })
        .add_plugins((AgentPlugin, SessionLogPlugin, ModelsPlugin, DefaultsPlugin))
        .init_resource::<Ended>()
        .add_observer(log_ended);
    app.finish();
    app.update();
    let agent = first_agent(app.world_mut()).unwrap_or(Entity::PLACEHOLDER);
    app.update();
    (app, wakes, home, agent)
}

/// Sends `text` to `agent` and runs `app` until the turn ended.
fn ask(app: &mut App, wakes: &Receiver<()>, agent: Entity, text: &str) {
    let ended = app.world().resource::<Ended>().0.len();
    let world = app.world_mut();
    world.trigger(Deliver::new(agent, text, DeliveryMode::Steer));
    run_until(app, wakes, |world| {
        world.resource::<Ended>().0.len() > ended
    });
}

/// What `defaults.json` in `home` holds.
fn saved(home: &Home) -> String {
    fs::read_to_string(home.defaults()).unwrap_or_default()
}

/// The model `agent` is on.
fn model_of(app: &App, agent: Entity) -> Option<&str> {
    let choice = app.world().get::<ModelChoice>(agent);
    choice.map(|choice| choice.0.as_str())
}

#[test]
fn a_model_named_with_model_is_used_for_that_run_only() {
    let chosen = MockCompletionModel::from_stream_turns([reply("hi")]);
    let (mut app, wakes, home, agent) =
        remembering("defaults-flag", &chosen, Some("corpus-test/chosen"));
    ask(&mut app, &wakes, agent, "Hello.");
    assert_eq!(model_of(&app, agent), Some("corpus-test/chosen"));
    assert_eq!(chosen.request_count(), 1);
    assert_eq!(saved(&home), SAVED);
}

#[test]
fn a_model_picked_with_model_is_remembered_once_it_answered() {
    let chosen = MockCompletionModel::from_stream_turns([reply("hi")]);
    let (mut app, wakes, home, agent) = remembering("defaults-picked", &chosen, None);
    assert_eq!(model_of(&app, agent), Some("corpus-test/saved"));
    run(&mut app, agent, "model corpus-test/chosen");
    app.update();
    assert_eq!(model_of(&app, agent), Some("corpus-test/chosen"));
    assert_eq!(saved(&home), SAVED, "not before it answered");
    ask(&mut app, &wakes, agent, "Hello.");
    assert!(
        saved(&home).contains("\"corpus-test/chosen\""),
        "{}",
        saved(&home)
    );
}

#[test]
fn a_model_picked_with_model_is_not_remembered_while_it_fails() {
    let failure = MockStreamEvent::error("no credit");
    let chosen = MockCompletionModel::from_stream_turns([vec![failure], reply("hi")]);
    let (mut app, wakes, home, agent) = remembering("defaults-failed", &chosen, None);
    run(&mut app, agent, "model corpus-test/chosen");
    ask(&mut app, &wakes, agent, "Hello.");
    let ended = &app.world().resource::<Ended>().0;
    assert!(
        matches!(ended.last(), Some(TurnOutcome::Failed(_))),
        "{ended:?}"
    );
    assert_eq!(saved(&home), SAVED);
    // Kept until it answers.
    ask(&mut app, &wakes, agent, "Again.");
    assert!(
        saved(&home).contains("\"corpus-test/chosen\""),
        "{}",
        saved(&home)
    );
}

#[test]
fn a_model_picked_while_a_turn_runs_is_not_remembered() {
    let chosen = MockCompletionModel::from_stream_turns([reply("hi")]);
    let (mut app, wakes, home, agent) =
        remembering("defaults-busy", &chosen, Some("corpus-test/chosen"));
    // A turn runs, as one a resumed session continues.
    let turn = app.world_mut().spawn(TurnOf(agent)).id();
    assert!(app.world().get::<ActiveTurn>(agent).is_some());
    // Refused: a model is not changed under a running turn.
    run(&mut app, agent, "model corpus-test/chosen");
    app.world_mut().entity_mut(turn).despawn();
    ask(&mut app, &wakes, agent, "Hello.");
    assert_eq!(chosen.request_count(), 1);
    assert_eq!(saved(&home), SAVED);
}

#[test]
fn a_model_put_on_an_agent_without_a_choice_is_never_remembered() {
    let chosen = MockCompletionModel::from_stream_turns([reply("hi")]);
    let (mut app, wakes, home, agent) = remembering("defaults-inserted", &chosen, None);
    // As restoring a session or forking an agent does.
    let choice = ModelChoice("corpus-test/chosen".to_owned());
    app.world_mut().entity_mut(agent).insert(choice);
    ask(&mut app, &wakes, agent, "Hello.");
    assert_eq!(chosen.request_count(), 1);
    assert_eq!(saved(&home), SAVED);
}

#[test]
fn an_effort_picked_is_remembered_with_the_users_model_only() {
    let Ok(found) = Catalog::builtin().resolve("deepseek/deepseek-flash") else {
        return;
    };
    let choices = found.spec.reasoning.choices();
    let Some(level) = choices.iter().find(|choice| choice.reasoning.is_some()) else {
        return;
    };
    let effort = ReasoningSetting(level.reasoning);
    for (name, users) in [
        ("defaults-effort-own", true),
        ("defaults-effort-flag", false),
    ] {
        let model = MockCompletionModel::from_stream_turns([reply("hi")]);
        let (mut app, wakes, home, agent) = remembering(name, &model, None);
        let saved_before = match users {
            true => r#"{"model":"deepseek/deepseek-flash","effort":null}"#,
            false => SAVED,
        };
        fs::write(home.defaults(), saved_before).ok();
        // On DeepSeek as `--model` leaves an agent: not the user's choice.
        let Some(connected) = rig_harness_test_support::connect(&model) else {
            return;
        };
        let choice = ModelChoice("deepseek/deepseek-flash".to_owned());
        app.world_mut()
            .entity_mut(agent)
            .insert((connected, choice));
        app.world_mut().trigger(SetReasoning {
            entity: agent,
            effort,
            kind: ChoiceKind::ByUser,
        });
        app.update();
        assert_eq!(saved(&home), saved_before, "not before it answered");
        ask(&mut app, &wakes, agent, "Hello.");
        let kept = saved(&home);
        assert_eq!(kept != saved_before, users, "{name}: {kept}");
    }
}

#[cfg(feature = "tui")]
#[test]
fn a_model_without_its_key_is_listed_with_why_and_offered_as_the_argument_once_typed() {
    use std::borrow::Cow;

    use rig_harness::rig_core::catalog::{Catalog, Connector, ModelSpec, Transport};
    use rig_harness::rig_core::providers::registry::{ConnectError, ProviderId};

    let Ok(corp) = ProviderId::declare("rig-models-corp") else {
        return;
    };
    let mut catalog = Catalog::default();
    catalog.insert(ModelSpec::new(corp, "big").with_tools(true));
    let mut connector = Connector::new(catalog);
    connector.provide(corp, |_: &ModelSpec, _: &Catalog, _: &Transport| {
        Err(ConnectError::MissingKey {
            vendor: "rig-models-corp".to_owned(),
            tried: vec![Cow::Borrowed("CORP_KEY")],
        })
    });
    let (mut app, agent) = routed();
    app.insert_resource(Models(connector));
    let input = (rig_tui::Editor::default(), rig_tui::MainEditor);
    let input = app.world_mut().spawn(input).id();
    app.update();
    let mut completions = app.world_mut().query::<&rig_tui::CompletionSource>();
    // Listing the catalog waits for the input to be a `/model` command.
    assert_eq!(completions.iter(app.world()).count(), 0);
    if let Some(mut input) = app.world_mut().get_mut::<rig_tui::Editor>(input) {
        input.set("/model rig");
    }
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

#[cfg(feature = "tui")]
#[test]
fn the_model_picker_opens_on_the_agents_model_and_says_it_is_current() {
    let model = MockCompletionModel::from_stream_turns([reply("hi")]);
    let (mut app, agent) = routed();
    let served = [("corpus-test/one", &model), ("corpus-test/two", &model)];
    app.insert_resource(scripted_models(served));
    let choice = ModelChoice("corpus-test/two".to_owned());
    app.world_mut().entity_mut(agent).insert(choice);
    run(&mut app, agent, "model");
    let world = app.world_mut();
    let mut pickers = world.query::<&rig_harness::pick::Picker>();
    let picker = pickers.iter(world).next();
    let selected = picker.and_then(|picker| picker.items.get(picker.selected));
    assert_eq!(
        selected.map(|item| (item.command.as_str(), item.detail.as_deref())),
        Some(("model corpus-test/two", Some("current")))
    );
    let marked = picker.map(|picker| {
        let current = picker.items.iter().filter(|item| item.detail.is_some());
        current.count()
    });
    assert_eq!(marked, Some(1));
}
