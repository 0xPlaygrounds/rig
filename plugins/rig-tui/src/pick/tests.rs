use rig_harness::prelude::*;

use super::*;
use crate::tests::{press, type_text, view_app};
use rig_harness::ask::Choice;
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
    let entity = world
        .query_filtered::<Entity, With<PickerInput>>()
        .single(world)
        .ok()?;
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
    assert_eq!(crate::tests::input(&app).text(), "");
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
fn a_picker_filled_in_batches_keeps_what_is_typed_and_selected() {
    use rig_harness::pick::Highlighted;

    let (mut app, _) = view_app();
    app.init_resource::<Reported>().add_observer(report);
    let batch = |items: &[&str]| {
        let picker = Picker::new("Files");
        items.iter().fold(picker, |picker, item| {
            picker.item(*item, format!("open {item}"))
        })
    };
    let picker = app.world_mut().spawn(batch(&["a.rs", "b.rs", "c.md"])).id();
    app.update();
    type_text(&mut app, "rs");
    press(&mut app, "down");
    let highlighted = |app: &App| app.world().get::<Highlighted>(picker).copied();
    assert_eq!(highlighted(&app), Some(Highlighted(1)));
    // The next batch comes in, an item before the selected one among them.
    let more = batch(&["a.rs", "aa.rs", "b.rs", "c.md", "d.rs"]);
    app.world_mut().entity_mut(picker).insert(more);
    app.update();
    let state = app.world().get::<PickerInput>(picker);
    assert_eq!(state.map(PickerInput::filter), Some("rs"));
    assert_eq!(highlighted(&app), Some(Highlighted(2)));
    press(&mut app, "enter");
    assert_eq!(
        app.world().resource::<Reported>().0,
        [Some(Chosen::item(2))]
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
    assert_eq!(crate::tests::input(&app).text(), "hi");
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
fn a_ticked_choice_goes_with_the_note_typed() {
    let (mut app, agent) = view_app();
    app.init_resource::<Answered>().add_observer(record_answers);
    let ask = Ask::new("Allow `rm`?")
        .choice("deny", "Deny")
        .choice("allow", "Allow")
        .free_text();
    app.world_mut().spawn((ask, ForAgent(agent)));
    app.update();
    // One choice only: ticking Allow unticks Deny.
    press(&mut app, "tab tab");
    type_text(&mut app, "just once");
    press(&mut app, "enter");
    let noted = Answer::choice("allow").with_text("just once");
    assert_eq!(
        app.world().resource::<Answered>().0,
        [("Allow `rm`?".to_owned(), Some(noted))]
    );
}

#[test]
fn a_long_detail_leaves_the_label_half_the_row() {
    use unicode_width::UnicodeWidthStr;

    let item = Item {
        id: "",
        label: "rig_basics::subagents::SubagentsPlugin",
        detail: Some(
            "(rig-basics) Subagents: the task, message and wait tools, with which the agent starts other agents",
        ),
        checked: false,
        more: None,
    };
    let column = detail_column(std::slice::from_ref(&item), 60);
    let row = item.line(60, column).to_string();
    assert_eq!(row.width(), 60, "{row}");
    assert!(
        row.starts_with("rig_basics::subagents::SubagentsPlugin  "),
        "{row}"
    );
    assert!(row.ends_with('…'), "{row}");
}

#[test]
fn a_long_label_is_shortened_beside_its_whole_detail() {
    use unicode_width::UnicodeWidthStr;

    let item = Item {
        id: "",
        label: "fix the flaky picker test in the terminal view · ~/code/rigs",
        detail: Some("$0.42 · 3h ago"),
        checked: false,
        more: None,
    };
    let column = detail_column(std::slice::from_ref(&item), 40);
    let row = item.line(40, column).to_string();
    assert_eq!(row.width(), 40, "{row}");
    assert!(row.ends_with(" $0.42 · 3h ago"), "{row}");
    assert!(row.starts_with("fix the flaky picker te…  "), "{row}");
    let short = Item {
        id: "",
        label: "main",
        detail: None,
        checked: false,
        more: None,
    };
    assert_eq!(short.line(40, 0).to_string(), "main");
}

#[test]
fn details_start_at_one_column() {
    use unicode_width::UnicodeWidthStr;

    let item = |label, detail| Item {
        id: "",
        label,
        detail: Some(detail),
        checked: false,
        more: None,
    };
    let items = [
        item("main", "the default branch"),
        item("日本語のラベル", "a wide label"),
        item("rig_basics::Plugin", "(rig-basics) The basics."),
    ];
    let starts = |width: usize| -> Vec<usize> {
        let column = detail_column(&items, width);
        let rows = items.iter().map(|item| item.line(width, column));
        let start = |row: Line<'static>| {
            let spans = row.spans.iter().rev().skip(1);
            spans.map(|span| span.content.width()).sum()
        };
        rows.map(start).collect()
    };
    // Two after the widest label.
    assert_eq!(starts(60), [20, 20, 20]);
    // A narrow row keeps a third for the details and cuts the labels.
    assert_eq!(starts(24), [16, 16, 16]);
    // A short detail, such as a session's folder and age, stays whole in
    // a narrow row; the label is cut.
    let session = [item(
        "Write a 40-line poem about rust",
        "…/pr7/work · 15m ago",
    )];
    let column = detail_column(&session, 28);
    let row: String = session
        .iter()
        .map(|item| item.line(28, column).to_string())
        .collect();
    assert!(row.ends_with("…/pr7/work · 15m ago"), "{row}");
    assert!(row.starts_with("Write…  …"), "{row}");
    // One session's long detail leaves the others theirs.
    let sessions = [
        item("Write a 40-line poem about rust", "…/pr7/work · 15m ago"),
        item("Say hi in two words", "…/a-very-long-folder/name · 3h ago"),
    ];
    let column = detail_column(&sessions, 28);
    let row = sessions
        .first()
        .map(|item| item.line(28, column).to_string());
    assert!(
        row.as_ref()
            .is_some_and(|row| row.ends_with("…/pr7/work · 15m ago")),
        "{row:?}"
    );
}

#[test]
fn the_whole_detail_shows_after_right() {
    use crate::TuiTestScreen;

    let (mut app, _) = view_app();
    app.insert_resource(TuiTestScreen::new(80, 24));
    let words: Vec<String> = (0..12).map(|n| format!("detail{n:02}")).collect();
    let detail = words.join(" ");
    assert_eq!(detail.len(), 107);
    let mut picker = Picker::new("Plugins").item("rig_basics::subagents::SubagentsPlugin", "");
    if let Some(item) = picker.items.last_mut() {
        item.detail = Some(detail);
    }
    app.world_mut().spawn(picker);
    app.update();
    let screen = |app: &App| app.world().resource::<TuiTestScreen>().rows().join("\n");
    assert!(!screen(&app).contains("detail11"), "{}", screen(&app));
    assert!(screen(&app).contains("→ shows more"), "{}", screen(&app));
    press(&mut app, "right");
    let shown = screen(&app);
    assert!(words.iter().all(|word| shown.contains(word)), "{shown}");
}

#[test]
fn typing_lists_the_items_it_names_before_those_it_describes() {
    for (keys, selected) in [("enter", 1), ("down enter", 0)] {
        let (mut app, _) = view_app();
        app.init_resource::<Reported>().add_observer(report);
        let mut picker = Picker::new("Plugins")
            .item("rig_inspect::InspectPlugin", "")
            .item("rig_basics::reload::ReloadPlugin", "");
        if let Some(item) = picker.items.first_mut() {
            item.detail = Some("[runs until the next reload] The inspect tool.".to_owned());
        }
        app.world_mut().spawn(picker);
        app.update();
        type_text(&mut app, "reload");
        press(&mut app, keys);
        assert_eq!(
            app.world().resource::<Reported>().0,
            [Some(Chosen::item(selected))]
        );
    }
}

#[test]
fn typing_finds_an_item_by_its_detail_or_full_text() {
    for filter in ["just now", "ancestor"] {
        let (mut app, _) = view_app();
        app.init_resource::<Reported>().add_observer(report);
        let mut picker = Picker::new("Sessions")
            .item("another session", "")
            .item("fix the parser", "");
        if let Some(item) = picker.items.last_mut() {
            item.detail = Some("…/repo/work · just now".to_owned());
            item.more = Some("/ancestor/full/repo/work".to_owned());
        }
        app.world_mut().spawn(picker);
        app.update();
        type_text(&mut app, filter);
        press(&mut app, "enter");
        assert_eq!(
            app.world().resource::<Reported>().0,
            [Some(Chosen::item(1))],
            "{filter}"
        );
    }
}

#[test]
fn a_whole_detail_offers_nothing_more() {
    use crate::TuiTestScreen;

    let (mut app, _) = view_app();
    app.insert_resource(TuiTestScreen::new(80, 24));
    let mut picker = Picker::new("Models").item("openai/gpt-5", "");
    if let Some(item) = picker.items.last_mut() {
        item.detail = Some("no key".to_owned());
    }
    app.world_mut().spawn(picker);
    app.update();
    let screen = app.world().resource::<TuiTestScreen>().rows().join("\n");
    assert!(screen.contains("no key"), "{screen}");
    assert!(!screen.contains("→ shows more"), "{screen}");
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

/// A message that comes back while a question is answered in the input
/// waits for the question, then goes before the draft.
#[test]
fn what_comes_back_during_a_question_in_the_input_waits_for_it() {
    let (mut app, agent) = view_app();
    type_text(&mut app, "half a thought");
    let title = Ask::new("PR title").initial("Fix the parser");
    app.world_mut().spawn((title, ForAgent(agent)));
    app.update();
    app.world_mut().write_message(Recalled {
        agent,
        text: "deploy it".to_owned(),
        why: None,
    });
    app.update();
    assert_eq!(crate::tests::typed(&app), "Fix the parser");
    press(&mut app, "esc");
    app.update();
    assert_eq!(crate::tests::typed(&app), "deploy it\n\nhalf a thought");
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
    use crate::InputContext;

    let (mut app, _) = view_app();
    let picker = Picker::new("Branch")
        .item("main", "")
        .item("jam", "")
        .item("jet", "");
    app.world_mut().spawn(picker);
    // Shown in one frame, noted at the start of the next.
    app.update();
    app.update();
    let selection = |app: &App| app.world().resource::<InputContext>().selection();
    assert_eq!(selection(&app), Some(0));
    press(&mut app, "down");
    assert_eq!(selection(&app), Some(1));
    // Typing `je` leaves `jet` alone, the third item of all.
    type_text(&mut app, "je");
    assert_eq!(selection(&app), Some(2));
}

/// A question's own keys: choices that start ticked, Space unticks one and
/// Enter answers exactly the ticked; `n` then Enter answers none; on the
/// typed-answer row a bound character types.
#[test]
fn a_questions_own_keys_tick_and_enter_answers_exactly_the_ticked() {
    let (mut app, agent) = view_app();
    app.init_resource::<Answered>().add_observer(record_answers);
    let keys = ChoiceKeys(vec![
        (KeyCode::Char(' '), ChoiceAction::Tick),
        (KeyCode::Char('a'), ChoiceAction::TickAll),
        (KeyCode::Char('n'), ChoiceAction::TickNone),
    ]);
    let ask = || {
        Ask::new("Approve these calls?")
            .multiple()
            .free_text()
            .with(Choice::new("0", "read a.rs").checked())
            .with(
                Choice::new("1", "write b.rs")
                    .checked()
                    .more("{\"path\":\"b.rs\"}"),
            )
    };
    app.world_mut()
        .spawn((ask(), keys.clone(), ForAgent(agent)));
    app.update();
    press(&mut app, "space enter");
    app.world_mut()
        .spawn((ask(), keys.clone(), ForAgent(agent)));
    app.update();
    press(&mut app, "n enter");
    app.world_mut().spawn((ask(), keys, ForAgent(agent)));
    app.update();
    // On the typed-answer row, `a` types; the ticks go with the note.
    press(&mut app, "down down");
    type_text(&mut app, "a");
    // Typing selects the first row the filter leaves; go back to the note.
    press(&mut app, "down enter");
    let answered = |answer: Answer| ("Approve these calls?".to_owned(), Some(answer));
    let none: [&str; 0] = [];
    assert_eq!(
        app.world().resource::<Answered>().0,
        [
            answered(Answer::choice("1")),
            answered(Answer::choices(none)),
            answered(Answer::choices(["0", "1"]).with_text("a")),
        ]
    );
}

/// A question superseded while shown gives way to its successor, which
/// keeps the user's ticks by choice id and ticks a new choice as it says;
/// the old one is recorded as superseded.
#[test]
fn a_superseded_question_keeps_the_users_ticks() {
    use rig_core::effect::Disposition;
    use rig_ecs::stage::{EffectRecorded, SupersededBy};

    #[derive(Resource, Default)]
    struct Dispositions(Vec<Disposition>);

    let (mut app, agent) = view_app();
    app.init_resource::<Answered>()
        .init_resource::<Dispositions>()
        .add_observer(record_answers)
        .add_systems(
            Last,
            |mut recorded: MessageReader<EffectRecorded>, mut seen: ResMut<Dispositions>| {
                let records = recorded.read().map(|record| record.0.disposition);
                seen.0.extend(records);
            },
        );
    let ask = Ask::new("Approve?")
        .multiple()
        .with(Choice::new("0", "read a.rs").checked())
        .with(Choice::new("1", "read b.rs").checked());
    let old = app.world_mut().spawn((ask.clone(), ForAgent(agent))).id();
    app.update();
    // Untick the first, select the second.
    press(&mut app, "tab");
    let joined = ask.with(Choice::new("2", "write c.rs").checked());
    let new = app.world_mut().spawn((joined, ForAgent(agent))).id();
    app.world_mut().entity_mut(old).insert(SupersededBy(new));
    app.update();
    let world = app.world_mut();
    let shown = world
        .query_filtered::<Entity, With<PickerInput>>()
        .single(world);
    assert_eq!(shown.ok(), Some(new));
    press(&mut app, "enter");
    assert_eq!(
        app.world().resource::<Answered>().0,
        [("Approve?".to_owned(), Some(Answer::choices(["1", "2"])))]
    );
    let seen = &app.world().resource::<Dispositions>().0;
    assert!(seen.contains(&Disposition::Superseded), "{seen:?}");
}

#[test]
fn a_pickers_preview_is_drawn_beside_its_list() {
    use rig_core::completion::Message;
    use rig_harness::pick::PickPreview;

    use crate::TuiTestScreen;

    let (mut app, _) = view_app();
    app.insert_resource(TuiTestScreen::new(100, 30));
    let picker = Picker::new("Sessions").item("fix the parser", "resume a");
    let mut conversation = rig_core::conversation::Conversation::default();
    conversation.push(Message::user("why does the parser drop tabs?"));
    let preview = PickPreview::Conversation {
        conversation,
        at: None,
    };
    let picker = app.world_mut().spawn((picker, preview)).id();
    app.update();
    app.update();
    let screen = |app: &App| app.world().resource::<TuiTestScreen>().rows().join("\n");
    let drawn = screen(&app);
    // In the right half of the picker's box, 90 columns from column 5.
    let row = drawn
        .lines()
        .find(|row| row.contains("why does the parser drop tabs?"));
    let column = row.and_then(|row| row.chars().position(|c| c == '›'));
    assert!(column.is_some_and(|column| column >= 50), "{drawn}");
    assert!(drawn.contains("fix the parser"), "{drawn}");
    app.world_mut()
        .entity_mut(picker)
        .insert(PickPreview::Text("a saved session".into()));
    app.update();
    app.update();
    let drawn = screen(&app);
    assert!(drawn.contains("a saved session"), "{drawn}");
    assert!(!drawn.contains("why does the parser"), "{drawn}");
}

#[test]
fn a_previewing_picker_takes_the_screen_over_its_keys_row() {
    use rig_core::completion::Message;
    use rig_harness::pick::PickPreview;

    use crate::{TuiFrame, TuiTestScreen};

    let conversation = PickPreview::Conversation {
        conversation: rig_core::conversation::Conversation::from_messages([Message::user(
            "why does the parser drop tabs?",
        )]),
        at: None,
    };
    let text = PickPreview::Text("why does the parser drop tabs?".into());
    for (width, height) in [(60, 30), (140, 50)] {
        for preview in [conversation.clone(), text.clone()] {
            let (mut app, agent) = view_app();
            app.insert_resource(TuiTestScreen::new(width, height));
            if let Some(mut behind) = app.world_mut().get_mut::<Conversation>(agent) {
                behind.push(Message::user("text from the transcript behind"));
            }
            let picker = Picker::new("Sessions").item("fix the parser", "resume a");
            app.world_mut().spawn((picker, preview));
            app.update();
            app.update();
            let rows = app.world().resource::<TuiTestScreen>().rows();
            assert!(
                rows.iter().all(|row| !row.contains("from the transcript")),
                "{rows:#?}"
            );
            let keys = rows.last().map(String::as_str).unwrap_or_default();
            assert_eq!(
                keys.trim(),
                "type to filter, Enter picks, Esc closes",
                "{rows:#?}"
            );
            assert!(
                rows.iter().any(|row| row.contains("why does the parser")),
                "{rows:#?}"
            );
            let world = app.world_mut();
            let pane = world
                .query_filtered::<Entity, With<PreviewPane>>()
                .iter(world)
                .next();
            let area = pane.and_then(|pane| world.resource::<TuiFrame>().area(pane));
            assert!(
                area.is_none_or(|area| area.bottom() < height),
                "{area:?} {rows:#?}"
            );
        }
    }
}

#[test]
fn a_narrow_keys_row_leaves_out_its_first_keys() {
    use rig_harness::pick::PickPreview;

    use crate::TuiTestScreen;

    let (mut app, _) = view_app();
    app.insert_resource(TuiTestScreen::new(60, 30));
    let mut picker = Picker::new("Plugins").item("rig_basics::subagents::SubagentsPlugin", "");
    if let Some(item) = picker.items.last_mut() {
        item.more = Some("(rig-basics) Subagents: ".repeat(60));
    }
    app.world_mut()
        .spawn((picker, PickPreview::Text("a plugin".into())));
    app.update();
    press(&mut app, "right");
    let rows = app.world().resource::<TuiTestScreen>().rows();
    let keys = rows.last().map(String::as_str).unwrap_or_default();
    assert_eq!(
        keys.trim(),
        "Shift+↑↓ scrolls, ← hides more, Enter picks, Esc closes",
        "{rows:#?}"
    );
}
