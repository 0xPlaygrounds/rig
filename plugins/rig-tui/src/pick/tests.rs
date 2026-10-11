use rig_harness::prelude::*;

use super::*;
use crate::TuiInput;
use crate::tests::{press, type_text, view_app};

/// What the pickers reported, in order.
#[derive(Resource, Default)]
struct Reported(Vec<Option<Chosen>>);

fn report(picked: On<Picked>, mut reported: ResMut<Reported>) {
    reported.0.push(picked.chosen.clone());
}

/// The commands run, by line.
#[derive(Resource, Default)]
struct Ran(Vec<String>);

/// The title of the picker or question shown.
fn shown(app: &mut App) -> Option<String> {
    let world = app.world_mut();
    let entity = world.resource::<TuiView>().choosing.as_ref()?.entity;
    let title = world
        .get::<Picker>(entity)
        .map(|picker| picker.title.clone());
    title.or_else(|| world.get::<Ask>(entity).map(|ask| ask.title.clone()))
}

#[test]
fn pickers_wait_their_turn_and_say_what_was_picked() {
    let (mut app, agent) = view_app();
    app.init_resource::<Reported>()
        .init_resource::<Ran>()
        .add_observer(report)
        .add_command(
            "note",
            "Notes its arguments",
            |In(args): In<CommandArgs>, mut ran: ResMut<Ran>| {
                ran.0.push(args.args);
            },
        );
    let world = app.world_mut();
    let first = world
        .spawn(
            Picker::new("Size")
                .item("small", "note small")
                .item("large", "note large"),
        )
        .id();
    world.spawn(Picker::new("Branch").item("main", "").free_text());
    world.spawn(
        Picker::new("Model")
            .for_agent(agent)
            .item("fast", "note fast"),
    );
    app.update();
    assert_eq!(shown(&mut app).as_deref(), Some("Size"));
    // Keys go to the picker, not the input.
    press(&mut app, "down enter");
    assert!(app.world().get_entity(first).is_err());
    assert_eq!(app.world().resource::<Ran>().0, ["large"]);
    assert_eq!(shown(&mut app).as_deref(), Some("Branch"));
    // Typed text that no item holds is the answer.
    type_text(&mut app, "feature");
    press(&mut app, "enter");
    assert_eq!(shown(&mut app).as_deref(), Some("Model"));
    press(&mut app, "esc");
    assert_eq!(shown(&mut app), None);
    assert_eq!(
        app.world().resource::<Reported>().0,
        [
            Some(Chosen::Item(1)),
            Some(Chosen::Text("feature".to_owned())),
            None
        ]
    );
    assert_eq!(app.world().resource::<TuiInput>().text(), "");
}

#[test]
fn a_plugin_replaces_and_closes_its_own_picker() {
    let (mut app, _) = view_app();
    app.init_resource::<Reported>().add_observer(report);
    let picker = app
        .world_mut()
        .spawn(Picker::new("Files").item("a.rs", "").item("b.rs", ""))
        .id();
    app.update();
    type_text(&mut app, "b");
    // Replaced: the new items, and the filter starts over.
    let replaced = Picker::new("Files").item("c.rs", "").item("d.rs", "");
    app.world_mut().entity_mut(picker).insert(replaced);
    app.update();
    press(&mut app, "down enter");
    assert_eq!(
        app.world().resource::<Reported>().0,
        [Some(Chosen::Item(1))]
    );
    // Despawned by its plugin, a picker closes without a report.
    let picker = app
        .world_mut()
        .spawn(Picker::new("Gone").item("x", ""))
        .id();
    app.update();
    app.world_mut().despawn(picker);
    app.update();
    assert_eq!(shown(&mut app), None);
    type_text(&mut app, "hi");
    assert_eq!(app.world().resource::<TuiInput>().text(), "hi");
}

/// Each question answered, by title, with its answer: `None` for one
/// refused.
#[derive(Resource, Default)]
struct Answered(Vec<(String, Option<Answer>)>);

fn record_answers(
    answered: On<Insert<Outcome>>,
    asks: Query<(&Ask, &Outcome)>,
    mut log: ResMut<Answered>,
) {
    if let Ok((ask, outcome)) = asks.get(answered.entity) {
        log.0.push((ask.title.clone(), Answer::of(outcome)));
    }
}

#[test]
fn the_view_answers_the_questions_of_every_agent() {
    let (mut app, agent) = view_app();
    app.init_resource::<Answered>().add_observer(record_answers);
    let can_answer = app
        .world_mut()
        .run_system_cached_with(
            |In(agent): In<Entity>, answerers: Answerers| answerers.can_answer(Some(agent)),
            agent,
        )
        .unwrap_or(false);
    assert!(can_answer);
    let mut ask = |question: Ask| {
        app.world_mut().spawn((question, ForAgent(agent)));
    };
    ask(Ask::yes_no("Run it?"));
    ask(Ask::yes_no("Why not?").free_text());
    ask(Ask::yes_no("Really?"));
    app.update();
    assert_eq!(shown(&mut app).as_deref(), Some("Run it?"));
    press(&mut app, "down enter");
    type_text(&mut app, "too slow");
    // Esc refuses the last one.
    press(&mut app, "enter esc");
    let answered = |title: &str, answer: Option<Answer>| (title.to_owned(), answer);
    assert_eq!(
        app.world().resource::<Answered>().0,
        [
            answered("Run it?", Some(Answer::Choice("no".to_owned()))),
            answered("Why not?", Some(Answer::Text("too slow".to_owned()))),
            answered("Really?", None),
        ]
    );
    // An answer from elsewhere closes the question here.
    let elsewhere = app
        .world_mut()
        .spawn((Ask::yes_no("Again?"), ForAgent(agent)))
        .id();
    app.update();
    assert_eq!(shown(&mut app).as_deref(), Some("Again?"));
    app.world_mut()
        .entity_mut(elsewhere)
        .insert(Answer::Choice("yes".to_owned()).outcome());
    app.update();
    assert_eq!(shown(&mut app), None);
}
