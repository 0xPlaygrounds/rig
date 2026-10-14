use std::fs;
use std::sync::mpsc::Receiver;

use rig_core::message::Message;
use rig_core::test_utils::MockCompletionModel;
use rig_harness::harness_protocol::SessionId;
use rig_harness::rig_ecs::journal::JournalPlugin;
use rig_harness_test_support::{app_in, connect, first_agent, reply, run_until, scratch_home};
use rig_tui::{Editor, MainEditor};

use super::*;

/// A plugin's saved state: the turns the agent ended.
#[derive(Component, Reflect, Default, Debug, PartialEq, Eq)]
#[reflect(Component, Saved, Default)]
struct Turns(u32);

fn count(
    settled: On<Add<Settled>>,
    turns: Query<&ForAgent, With<TurnOutcome>>,
    mut counts: Query<&mut Turns>,
) {
    if let Ok(ForAgent(agent)) = turns.get(settled.entity)
        && let Ok(mut count) = counts.get_mut(*agent)
    {
        count.0 += 1;
    }
}

/// The kernel, the harness's session service and this plugin on session
/// `id` of `home`, after its first frame restored it.
fn start(home: &Home, id: &SessionId) -> (App, Receiver<()>) {
    start_with(home, id, |_| {})
}

/// [`start`], with `more` added before the first frame.
fn start_with(home: &Home, id: &SessionId, more: impl FnOnce(&mut App)) -> (App, Receiver<()>) {
    let (mut app, wakes) = app_in(home, id);
    app.world_mut().spawn((Editor::default(), MainEditor));
    app.add_plugins((AgentPlugin, JournalPlugin, ForkSessionPlugin))
        .register_required_components::<Agent, Turns>()
        .add_observer(count);
    more(&mut app);
    app.finish();
    app.update();
    (app, wakes)
}

/// What each message of the first agent says, and its saved count.
fn state(app: &mut App) -> (Vec<String>, Option<u32>) {
    let world = app.world_mut();
    let Some((conversation, turns)) = world.query::<(&Conversation, &Turns)>().iter(world).next()
    else {
        return (Vec::new(), None);
    };
    let texts = conversation
        .messages()
        .iter()
        .map(|message| match message {
            Message::User { content } => content
                .iter()
                .filter_map(|item| match item {
                    UserContent::Text(text) => Some(text.text.clone()),
                    _ => None,
                })
                .collect(),
            message => message.final_answer().unwrap_or_default(),
        })
        .collect();
    (texts, Some(turns.0))
}

/// The first agent, given the scripted replies "r1".."r3", after the user
/// typed "one", "two" and "three", with `before` run before each.
fn three_turns(
    app: &mut App,
    wakes: &Receiver<()>,
    mut before: impl FnMut(&str),
) -> Option<Entity> {
    let model = MockCompletionModel::from_stream_turns([reply("r1"), reply("r2"), reply("r3")]);
    let agent = first_agent(app.world_mut())?;
    app.world_mut().entity_mut(agent).insert(connect(&model)?);
    for (done, text) in (1..).zip(["one", "two", "three"]) {
        before(text);
        let typed = Deliver::new(agent, text, DeliveryMode::Queue).with_author(Author::user());
        app.world_mut().trigger(typed);
        run_until(app, wakes, |world| {
            world
                .get::<Turns>(agent)
                .is_some_and(|turns| turns.0 == done)
        });
    }
    app.update();
    Some(agent)
}

/// Types `line` for `agent` and picks "two" in the `/branch` picker it
/// opens; `false` without a picker of the three messages.
fn branch_before_two(app: &mut App, wakes: &Receiver<()>, agent: Entity, line: &str) -> bool {
    let line = Deliver::new(agent, line, DeliveryMode::Queue).with_author(Author::user());
    app.world_mut().trigger(line);
    run_until(app, wakes, |world| {
        world
            .iter_entities()
            .any(|entity| entity.contains::<Picker>())
    });
    let world = app.world_mut();
    let picker = world
        .query::<(Entity, &Picker)>()
        .iter(world)
        .next()
        .filter(|(_, picker)| picker.items.len() == 3)
        .and_then(|(entity, picker)| {
            let two = picker.items.iter().position(|item| item.label == "two")?;
            Some((entity, two))
        });
    let Some((picker, two)) = picker else {
        return false;
    };
    app.world_mut().trigger(Picked {
        entity: picker,
        chosen: Some(Chosen::item(two)),
    });
    true
}

/// The session `session` of `home` was forked from, as its store says.
fn parent_of(home: &Home, session: &SessionId) -> Option<ForkedFrom> {
    let store = SessionStores::default().open(&home.session(session)).ok()?;
    ForkedFrom::read(&*store)
}

/// The session forked from another in `home`, once one is written.
fn branched(home: &Home) -> Option<SessionId> {
    fs::read_dir(home.sessions())
        .into_iter()
        .flatten()
        .filter_map(|entry| entry.ok()?.file_name().to_str()?.parse::<SessionId>().ok())
        .find(|session| parent_of(home, session).is_some())
}

#[test]
fn a_branch_before_turn_two_keeps_turn_one_and_its_state() {
    let home = scratch_home("fork-session-home");
    let id: SessionId = "1000-1".parse().unwrap_or_else(|_| SessionId::generate());
    let (mut app, wakes) = start(&home, &id);
    let agent = three_turns(&mut app, &wakes, |_| {});
    assert!(agent.is_some(), "an agent and a model");
    let Some(agent) = agent else {
        return;
    };
    assert_eq!(state(&mut app).1, Some(3));

    // `/branch` opens a picker of the user's messages; choose "two".
    let picked = branch_before_two(&mut app, &wakes, agent, "/branch");
    assert!(picked, "a picker of the three messages");
    // The fork is written at the frame's end.
    run_until(&mut app, &wakes, |_| branched(&home).is_some());
    drop(app);
    let branched = branched(&home);
    assert!(branched.is_some(), "a branched session");
    let Some(branched) = branched else {
        return;
    };
    let parent = parent_of(&home, &branched);
    assert_eq!(
        parent.map(|parent| (parent.session, parent.text)),
        Some((id.to_string(), Some("two".to_owned())))
    );

    // The branch holds turn one and the count as of then; "two" waits in
    // the input.
    let (mut app, _) = start(&home, &branched);
    assert_eq!(
        state(&mut app),
        (vec!["one".to_owned(), "r1".to_owned()], Some(1))
    );
    assert_eq!(input(&mut app), "two");
    drop(app);
    // Only its first start fills the input.
    let (mut app, _) = start(&home, &branched);
    assert_eq!(input(&mut app), "");
    drop(app);

    // The original, reloaded from its log, still has all three turns.
    let (mut app, _) = start(&home, &id);
    let (messages, turns) = state(&mut app);
    assert_eq!(messages, ["one", "r1", "two", "r2", "three", "r3"]);
    assert_eq!(turns, Some(3));
    drop(app);
    fs::remove_dir_all(home.root()).ok();
}

/// The exits asked for.
#[cfg(feature = "checkpoints")]
#[derive(Resource, Default)]
struct Exits(Vec<ExitReason>);

#[cfg(feature = "checkpoints")]
fn exits(exit: On<Exit>, mut seen: ResMut<Exits>) {
    seen.0.push(exit.0.clone());
}

#[cfg(feature = "checkpoints")]
#[test]
fn a_branch_with_its_files_switches_once_they_are_back() {
    use corpus_checkpoint_rewind::CheckpointRewindPlugin;
    use rig_harness_test_support::{AnswerAsks, Answers, answer_open};

    let home = scratch_home("fork-session-files");
    let repo = home.root().join("repo");
    fs::create_dir_all(&repo).ok();
    let git = std::process::Command::new("git")
        .arg("-C")
        .arg(&repo)
        .args(["init", "-q"])
        .status();
    assert!(git.is_ok_and(|status| status.success()), "a git repository");
    let id = SessionId::generate();
    let (mut app, wakes) = start_with(&home, &id, |app| {
        // A front that answers nothing by itself: questions stay open.
        app.insert_resource(ProjectRoot(repo.clone()))
            .add_plugins((CheckpointRewindPlugin, AnswerAsks))
            .init_resource::<Exits>()
            .add_observer(exits);
    });
    // The user writes a.txt before each turn: "1", "2", "3".
    let file = repo.join("a.txt");
    let number = |text: &str| match text {
        "one" => "1",
        "two" => "2",
        _ => "3",
    };
    let agent = three_turns(&mut app, &wakes, |text| {
        fs::write(&file, number(text)).ok();
    });
    let Some(agent) = agent else {
        return;
    };
    let read = || fs::read_to_string(&file).unwrap_or_default();

    // Putting the files back before "two" undoes the user's edit before
    // "three": the switch waits for their answer.
    assert!(branch_before_two(
        &mut app,
        &wakes,
        agent,
        "/branch --files"
    ));
    let question = "Rewind to before turn 1?";
    run_until(&mut app, &wakes, |world| {
        world
            .resource::<Answers>()
            .asked
            .iter()
            .any(|asked| asked == question)
    });
    assert!(app.world().resource::<Exits>().0.is_empty());
    assert_eq!(read(), "3");
    assert!(answer_open(
        app.world_mut(),
        question,
        &Answer::choice("yes")
    ));
    run_until(&mut app, &wakes, |world| {
        !world.resource::<Exits>().0.is_empty()
    });
    assert_eq!(read(), "2");
    let to = branched(&home);
    assert!(to.is_some(), "a branched session");
    assert_eq!(app.world().resource::<Exits>().0, [ExitReason::Switch(to)]);
    drop(app);
    fs::remove_dir_all(home.root()).ok();
}

/// What the main input line holds.
fn input(app: &mut App) -> String {
    let world = app.world_mut();
    world
        .query_filtered::<&Editor, With<MainEditor>>()
        .single(world)
        .map(|editor| editor.text().to_owned())
        .unwrap_or_default()
}
