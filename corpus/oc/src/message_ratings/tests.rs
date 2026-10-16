use std::fs;
use std::sync::mpsc::Receiver;

use rig_core::catalog::ModelSpec;
use rig_core::providers::registry::ProviderId;
use rig_core::test_utils::{MockCompletionModel, MockStreamEvent};
use rig_harness::harness_protocol::SessionId;
use rig_harness_test_support::{
    Ended, MemoryStore, app_in, app_on, commit_message, first_agent, log_ended, press, reply,
    run_until, scratch_home, scripted_models, tui_view,
};

use super::*;

/// The kernel, the session service and the ratings on session `id` of
/// `home`, after its first frame restored it, with its first agent (for a
/// new session, `main`) answered by `model`.
fn start(
    home: &Home,
    id: &str,
    models: &[(&str, &MockCompletionModel)],
    model: &str,
) -> Option<(App, Receiver<()>, Entity)> {
    let id: SessionId = id.parse().ok()?;
    let new = !home.session(&id).is_saved();
    let (mut app, wakes) = app_in(home, &id);
    if new {
        app.world_mut()
            .spawn((Name::new("main"), Agent, AgentId("main".to_owned())));
    }
    app.init_resource::<Ended>()
        .insert_resource(scripted_models(models.iter().copied()))
        .add_observer(log_ended)
        .add_plugins((AgentPlugin, SessionLogPlugin, MessageRatingsPlugin));
    app.finish();
    app.update();
    let agent = first_agent(app.world_mut())?;
    use_model(&mut app, agent, model)?;
    Some((app, wakes, agent))
}

/// Answers `agent`'s next turns with `model`.
fn use_model(app: &mut App, agent: Entity, model: &str) -> Option<()> {
    let connection = app.world().resource::<Models>().connect(model).ok()?;
    app.world_mut().entity_mut(agent).insert(connection);
    Some(())
}

/// Sends `line` as the user typed it: a command runs in that frame, a
/// message's turn is run to its end.
fn send(app: &mut App, wakes: &Receiver<()>, agent: Entity, line: &str) {
    let ended = app.world().resource::<Ended>().0.len();
    let typed = Deliver::new(agent, line, DeliveryMode::Queue).with_author(Author::user());
    app.world_mut().trigger(typed);
    if line.starts_with('/') {
        app.update();
    } else {
        run_until(app, wakes, |world| {
            world.resource::<Ended>().0.len() > ended
        });
    }
}

/// The newest picker `line` opened.
fn picker(app: &mut App, wakes: &Receiver<()>, agent: Entity, line: &str) -> Option<Picker> {
    send(app, wakes, agent, line);
    let world = app.world_mut();
    let mut pickers = world.query::<(Entity, &Picker)>();
    let (entity, picker) = pickers.iter(world).map(|(e, p)| (e, p.clone())).last()?;
    world.despawn(entity);
    Some(picker)
}

/// A scripted model whose replies name `name` as their model, as a
/// provider's own wire does.
fn named(name: &str, turns: Vec<Vec<MockStreamEvent>>) -> MockCompletionModel {
    let model = MockCompletionModel::from_stream_turns(turns);
    match ProviderId::declare("fake") {
        Ok(vendor) => model.priced_as(ModelSpec::new(vendor, name)),
        Err(_) => model,
    }
}

#[test]
fn ratings_stay_with_their_replies_never_reach_the_model_and_tally_by_model() {
    let home = scratch_home("oc-message-ratings-home");
    let a = named(
        "fake-a",
        vec![reply("Fixed the off-by-one."), reply("Renamed it.")],
    );
    let b = named("fake-b", vec![reply("Refactored the parser.")]);
    let c = named("fake-c", vec![reply("Added the test.")]);
    let models = [
        ("fake/fake-a", &a),
        ("fake/fake-b", &b),
        ("fake/fake-c", &c),
    ];

    // Two turns on fake-a, one on fake-b, rated as they go.
    let started = start(&home, "1000-1", &models, "fake/fake-a");
    assert!(started.is_some(), "a session with an agent");
    let Some((mut app, wakes, agent)) = started else {
        return;
    };
    send(&mut app, &wakes, agent, "Fix the bug.");
    send(&mut app, &wakes, agent, "/good clear fix");
    send(&mut app, &wakes, agent, "Rename it.");
    assert!(use_model(&mut app, agent, "fake/fake-b").is_some());
    send(&mut app, &wakes, agent, "Refactor the parser.");
    send(&mut app, &wakes, agent, "/bad");
    send(&mut app, &wakes, agent, "/bad wrong file");
    drop(app);

    // Reloaded, `/ratings` lists both; turn 3's says why it was bad.
    let restored = start(&home, "1000-1", &models, "fake/fake-a");
    assert!(restored.is_some(), "the session was restored");
    let Some((mut app, wakes, agent)) = restored else {
        return;
    };
    let listed = picker(&mut app, &wakes, agent, "/ratings");
    let labels: Vec<String> = listed
        .iter()
        .flat_map(|p| p.items.iter().map(|i| i.label.clone()))
        .collect();
    assert_eq!(
        labels,
        ["good · fake-a · clear fix", "bad · fake-b · wrong file"]
    );

    drop(app);

    // A second session with one `/good`; `/ratings --all` tallies all three.
    let second = start(&home, "1000-3", &models, "fake/fake-c");
    assert!(second.is_some(), "a second session");
    let Some((mut app, wakes, agent)) = second else {
        return;
    };
    send(&mut app, &wakes, agent, "Add a test.");
    send(&mut app, &wakes, agent, "/good");
    let all = picker(&mut app, &wakes, agent, "/ratings --all");
    let body = all.as_ref().map(|all| all.body.clone()).unwrap_or_default();
    for tally in [
        "fake-a: 1 good, 1 unrated",
        "fake-b: 1 bad",
        "fake-c: 1 good",
    ] {
        assert!(body.lines().any(|line| line == tally), "{tally} in {body}");
    }
    let labels: Vec<String> = all
        .iter()
        .flat_map(|p| p.items.iter().map(|i| i.label.clone()))
        .collect();
    assert_eq!(
        labels,
        [
            "good · fake-a · clear fix",
            "bad · fake-b · wrong file",
            "good · fake-c"
        ]
    );

    // No model ever read a reason.
    let requests = [a.requests(), b.requests(), c.requests()].concat();
    assert_eq!(requests.len(), 4);
    for request in requests
        .iter()
        .map(|request| serde_json::to_string(request).unwrap_or_default())
    {
        assert!(
            !request.contains("clear fix") && !request.contains("wrong file"),
            "{request}"
        );
    }
    drop(app);
    fs::remove_dir_all(home.root()).ok();
}

#[test]
fn alt_down_rates_the_last_reply_from_the_input() {
    let store = MemoryStore::default();
    let (mut app, _wakes) = app_on(&store);
    app.add_plugins((AgentPlugin, SessionLogPlugin, MessageRatingsPlugin));
    tui_view(&mut app);
    let agent = app
        .world_mut()
        .spawn((Name::new("main"), Agent, AgentId("main".to_owned())))
        .id();
    // The agent the view shows, whose input the keys are pressed in.
    app.insert_resource(ShownAgent(Some(agent)));
    app.finish();
    app.update();
    for message in [Message::user("Fix it."), Message::assistant("Fixed.")] {
        assert_eq!(commit_message(app.world_mut(), agent, message), Ok(()));
    }
    app.update();
    press(&mut app, "alt+down");
    app.update();
    let conversation = app.world().get::<Conversation>(agent);
    let rated = conversation.map(|c| ratings_in(c)).unwrap_or_default();
    let verdicts: Vec<Verdict> = rated
        .iter()
        .map(|(rated, _)| rated.rating.verdict)
        .collect();
    assert_eq!(verdicts, [Verdict::Bad]);
}
