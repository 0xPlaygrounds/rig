use rig_harness::prelude::*;

use super::*;
use crate::TuiInput;
use crate::tests::{press, type_text, view_app};
use rig_harness::pick::{Chosen, Picked, Picker};

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
        [Some(Chosen::item(1)), Some(Chosen::text("feature")), None]
    );
    assert_eq!(app.world().resource::<TuiInput>().text(), "");
}

#[test]
fn tab_ticks_several_items_and_enter_picks_them_in_list_order() {
    let (mut app, agent) = view_app();
    app.init_resource::<Reported>()
        .init_resource::<Ran>()
        .init_resource::<Answered>()
        .add_observer(report)
        .add_observer(record_answers)
        .add_command(
            "note",
            "Notes its arguments",
            |In(args): In<CommandArgs>, mut ran: ResMut<Ran>| {
                ran.0.push(args.args);
            },
        );
    let picker = Picker::new("Delete which files?")
        .multiple()
        .item("a.tmp", "note a")
        .item("b.log", "note b")
        .checked(true)
        .item("c.tmp", "note c");
    app.world_mut().spawn(picker);
    app.update();
    // b starts ticked: untick it, tick c, then go back up and tick a.
    press(&mut app, "down tab tab up up tab");
    assert_eq!(shown(&mut app).as_deref(), Some("Delete which files?"));
    press(&mut app, "enter");
    assert_eq!(
        app.world().resource::<Reported>().0,
        [Some(Chosen::items([0, 2]))]
    );
    assert_eq!(app.world().resource::<Ran>().0, ["a", "c"]);

    // With nothing ticked, Enter answers the selected choice.
    let checks = Ask::new("Which checks?")
        .choice("fmt", "fmt")
        .choice("test", "test")
        .multiple();
    app.world_mut().spawn((checks, ForAgent(agent)));
    app.update();
    press(&mut app, "down enter");
    let answers = &app.world().resource::<Answered>().0;
    let test = Answer::choice("test");
    assert_eq!(answers, &[("Which checks?".to_owned(), Some(test))]);
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
        [Some(Chosen::item(1))]
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
            answered("Run it?", Some(Answer::choice("no"))),
            answered("Why not?", Some(Answer::text("too slow"))),
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
        .insert(Answer::choice("yes").outcome());
    app.update();
    assert_eq!(shown(&mut app), None);
}

#[test]
fn a_long_label_is_shortened_beside_its_whole_detail() {
    use unicode_width::UnicodeWidthStr;

    let item = Item {
        label: "fix the flaky picker test in the terminal view · ~/code/rigs",
        detail: Some("$0.42 · 3h ago"),
        checked: false,
    };
    let row = item.line(40).to_string();
    assert_eq!(row.width(), 40, "{row}");
    assert!(row.ends_with(" $0.42 · 3h ago"), "{row}");
    assert!(row.starts_with("fix the flaky picker te…  "), "{row}");
    let short = Item {
        label: "main",
        detail: None,
        checked: false,
    };
    assert_eq!(short.line(40).to_string(), "main");
}

#[test]
fn a_question_with_initial_text_is_answered_in_the_input_and_the_draft_comes_back() {
    let (mut app, agent) = view_app();
    app.init_resource::<Answered>().add_observer(record_answers);
    type_text(&mut app, "half a thought");
    let title = Ask::new("PR title").initial("Fix the parser");
    app.world_mut().spawn((title, ForAgent(agent)));
    let body = Ask::new("PR body").initial("It loops.").multiline();
    app.world_mut().spawn((body, ForAgent(agent)));
    app.update();
    // The proposed answer is in the input, to edit; nothing is in front.
    assert_eq!(crate::tests::typed(&app), "Fix the parser");
    press(
        &mut app,
        "backspace backspace backspace backspace backspace backspace",
    );
    type_text(&mut app, "lexer");
    press(&mut app, "enter");
    // The next one takes the input; Shift+Enter adds a line, Esc refuses.
    assert_eq!(crate::tests::typed(&app), "It loops.");
    press(&mut app, "shift+enter");
    type_text(&mut app, "Fixed.");
    press(&mut app, "esc");
    assert_eq!(crate::tests::typed(&app), "half a thought");
    let answered = |title: &str, answer: Option<Answer>| (title.to_owned(), answer);
    assert_eq!(
        app.world().resource::<Answered>().0,
        [
            answered("PR title", Some(Answer::text("Fix the lexer"))),
            answered("PR body", None),
        ]
    );
    // Nothing reached the agent.
    assert!(
        app.world()
            .resource::<crate::tests::Delivered>()
            .0
            .is_empty()
    );
}

#[test]
fn a_question_beside_the_work_leaves_the_keys_to_the_input() {
    let (mut app, agent) = view_app();
    app.init_resource::<Answered>().add_observer(record_answers);
    let offer = Ask::new("Save rule “tabs”?")
        .choice("save", "Save")
        .choice("no", "Dismiss")
        .free_text()
        .inline();
    app.world_mut().spawn((offer.clone(), ForAgent(agent)));
    app.update();
    assert_eq!(shown(&mut app), None);
    type_text(&mut app, "go on");
    assert_eq!(crate::tests::typed(&app), "go on");
    press(&mut app, "alt+1");
    let answered = |answer: &str| {
        (
            "Save rule “tabs”?".to_owned(),
            Some(Answer::choice(answer.to_owned())),
        )
    };
    assert_eq!(app.world().resource::<Answered>().0, [answered("save")]);
    // Its keys are free again once none shows.
    press(&mut app, "alt+1");
    assert_eq!(app.world().resource::<Answered>().0.len(), 1);
    // The choice after its choices answers it in the input.
    app.world_mut().spawn((offer, ForAgent(agent)));
    app.update();
    press(&mut app, "alt+3");
    assert_eq!(crate::tests::typed(&app), "");
    type_text(&mut app, "only in Go");
    press(&mut app, "enter");
    assert_eq!(crate::tests::typed(&app), "go on");
    let last = app.world().resource::<Answered>().0.last().cloned();
    let text = Some(Answer::text("only in Go"));
    assert_eq!(last, Some(("Save rule “tabs”?".to_owned(), text)));
}

#[test]
fn the_item_highlighted_is_known_by_its_place_among_all_the_items() {
    use crate::TuiInputState;

    let (mut app, _) = view_app();
    let picker = Picker::new("Branch")
        .item("main", "")
        .item("jam", "")
        .item("jet", "");
    app.world_mut().spawn(picker);
    // Shown in one frame, noted at the start of the next.
    app.update();
    app.update();
    let selection = |app: &App| app.world().resource::<TuiInputState>().selection();
    assert_eq!(selection(&app), Some(0));
    press(&mut app, "down");
    assert_eq!(selection(&app), Some(1));
    // Typing `je` leaves `jet` alone, the third item of all.
    type_text(&mut app, "je");
    assert_eq!(selection(&app), Some(2));
}
