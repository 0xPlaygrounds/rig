use std::sync::mpsc::Receiver;

use rig_core::test_utils::MockCompletionModel;
use rig_harness::harness_protocol::SessionId;
use rig_harness::pick::PickPlugin;
use rig_harness_test_support::{
    Collected, Ended, app_in, collect, first_agent, log_ended, reply, run_until, scratch_home,
    scripted_models,
};

use super::*;

#[test]
fn stages_are_ordered_with_the_defaults() {
    rig_defaults_test_support::assert_stages_ordered(ConversationChaptersPlugin);
}

/// How many boundary calls settled.
#[derive(Resource, Default)]
struct Judged(usize);

fn count_judged(
    settled: On<Add<Settled>>,
    calls: Query<(), With<Boundary>>,
    mut judged: ResMut<Judged>,
) {
    if calls.contains(settled.entity) {
        judged.0 += 1;
    }
}

/// The scripted models: the agent's, the boundary model's, the summaries'.
struct Models3 {
    agent: MockCompletionModel,
    boundary: MockCompletionModel,
    summary: MockCompletionModel,
}

fn start(home: &Home, id: &SessionId, models: &Models3) -> (App, Receiver<()>, Entity) {
    let (mut app, wakes) = app_in(home, id);
    app.insert_resource(Settings::from_toml(
        "[chapters]\nboundary_model = \"test/boundary\"\nsummary_model = \"test/summary\"",
    ))
    .insert_resource(scripted_models([
        ("test/agent", &models.agent),
        ("test/boundary", &models.boundary),
        ("test/summary", &models.summary),
    ]))
    .add_plugins((
        AgentPlugin,
        SessionLogPlugin,
        PickPlugin,
        ConversationChaptersPlugin,
    ))
    .init_resource::<Ended>()
    .add_observer(log_ended)
    .init_resource::<Judged>()
    .add_observer(count_judged);
    collect::<Notice>(&mut app);
    collect::<Recalled>(&mut app);
    app.finish();
    app.update();
    let agent = first_agent(app.world_mut()).unwrap_or(Entity::PLACEHOLDER);
    app.world_mut().trigger(SetModel {
        entity: agent,
        model: "test/agent".to_owned(),
        kind: ChoiceKind::ForThisRun,
    });
    (app, wakes, agent)
}

/// Sends `text` and waits for its turn and the boundary call after it.
fn ask(app: &mut App, wakes: &Receiver<()>, agent: Entity, text: &str) {
    let judged = app.world().resource::<Judged>().0;
    app.world_mut()
        .trigger(Deliver::new(agent, text, DeliveryMode::Queue));
    run_until(app, wakes, |world| world.resource::<Judged>().0 > judged);
}

/// Runs the command `line` and waits until `done`.
fn run(
    app: &mut App,
    wakes: &Receiver<()>,
    agent: Entity,
    line: &str,
    done: impl Fn(&World) -> bool,
) {
    app.world_mut()
        .trigger(Deliver::new(agent, line, DeliveryMode::Queue));
    run_until(app, wakes, done);
}

fn lines(world: &World, agent: Entity) -> Vec<ChapterLine> {
    world
        .get::<Conversation>(agent)
        .map(outline)
        .unwrap_or_default()
}

/// Each chapter's title, turns and whether it is folded.
fn shape(world: &World, agent: Entity) -> Vec<(String, Option<(u64, u64)>, bool)> {
    let lines = lines(world, agent);
    let shape = |line: ChapterLine| {
        (
            line.chapter.title.clone(),
            line.turns,
            line.chapter.folded(),
        )
    };
    lines.into_iter().map(shape).collect()
}

fn folded(agent: Entity, n: usize) -> impl Fn(&World) -> bool {
    move |world| {
        lines(world, agent)
            .iter()
            .any(|line| line.number == n && line.chapter.folded())
    }
}

/// The messages of the agent's last request, as text.
fn last_request(model: &MockCompletionModel) -> String {
    let requests = model.requests();
    requests
        .last()
        .map(|request| format!("{:?}", request.chat_history))
        .unwrap_or_default()
}

fn noticed(world: &World, text: &str) -> bool {
    let notices = &world.resource::<Collected<Notice>>().0;
    notices.iter().any(|notice| notice.text.contains(text))
}

/// Whether a command line was refused for `why`.
fn refused(world: &World, why: &str) -> bool {
    let recalled = &world.resource::<Collected<Recalled>>().0;
    recalled
        .iter()
        .any(|line| line.why.as_deref().is_some_and(|text| text.contains(why)))
}

#[test]
fn chapters_split_fold_unfold_and_survive_a_restart() {
    let verdict = |new: bool, title: &str| {
        reply(&format!(
            "{{\"new_chapter\": {new}, \"title\": \"{title}\"}}"
        ))
    };
    let models = Models3 {
        agent: MockCompletionModel::from_stream_turns((1..=8).map(|n| reply(&format!("done {n}")))),
        boundary: MockCompletionModel::from_stream_turns([
            verdict(true, "Parser"),
            verdict(false, "Parser"),
            verdict(false, "Parser"),
            verdict(true, "Release"),
            verdict(false, "Release"),
            verdict(false, "Release"),
            verdict(false, "Release"),
            verdict(false, "Hotfix"),
        ]),
        summary: MockCompletionModel::from_stream_turns([
            reply(
                "## What was done\n- Pratt parser for expressions\n## Decisions\n- Pratt over LALR",
            ),
            reply("## What was done\n- Released v1.0"),
        ]),
    };
    let home = scratch_home("ct-conversation-chapters");
    let id = SessionId::generate();
    let (mut app, wakes, agent) = start(&home, &id, &models);
    for text in [
        "parser: fix the tokenizer",
        "parser: handle comments",
        "parser: add tests",
        "release: bump version",
        "release: write changelog",
        "release: tag v1",
    ] {
        ask(&mut app, &wakes, agent, text);
    }
    let parser = ("Parser".to_owned(), Some((1, 3)), false);
    let release = ("Release".to_owned(), Some((4, 6)), false);
    assert_eq!(
        shape(app.world(), agent),
        vec![parser.clone(), release.clone()]
    );

    // `/chapters`: the outline, as a picker.
    let picked = |world: &World| {
        world
            .try_query::<&Picker>()
            .and_then(|mut pickers| pickers.iter(world).next().cloned())
    };
    run(&mut app, &wakes, agent, "/chapters", |world| {
        picked(world).is_some()
    });
    let picker = picked(app.world()).unwrap_or_default();
    let labels: Vec<(&str, &str)> = picker
        .items
        .iter()
        .map(|item| {
            (
                item.label.as_str(),
                item.detail.as_deref().unwrap_or_default(),
            )
        })
        .collect();
    assert!(
        matches!(labels.as_slice(), [("1. Parser", one), ("2. Release", two)]
            if one.starts_with("turns 1–3 · open") && two.starts_with("turns 4–6 · current")),
        "{labels:?}"
    );

    // `/chapter Hotfix` while idle starts one with the next turn; `/fold 1`
    // gives the model the summary in place of turns 1–3.
    run(&mut app, &wakes, agent, "/chapter Hotfix", |world| {
        noticed(world, "Hotfix")
    });
    run(&mut app, &wakes, agent, "/fold 1", folded(agent, 1));
    ask(&mut app, &wakes, agent, "hotfix: crash on empty input");
    let sent = last_request(&models.agent);
    assert!(sent.contains("Pratt over LALR"), "{sent}");
    assert!(
        !sent.contains("fix the tokenizer") && !sent.contains("done 1"),
        "{sent}"
    );
    assert!(sent.contains("release: bump version"), "{sent}");
    let stored = app
        .world()
        .get::<Conversation>(agent)
        .map(|c| format!("{:?}", c.messages()));
    assert!(stored.is_some_and(|stored| stored.contains("fix the tokenizer")));
    // The boundary model said no; the user's chapter wins.
    let hotfix = ("Hotfix".to_owned(), Some((7, 7)), false);
    let parser_folded = ("Parser".to_owned(), Some((1, 3)), true);
    assert_eq!(
        shape(app.world(), agent),
        vec![parser_folded, release.clone(), hotfix.clone()]
    );

    // `/unfold 1`: the originals again.
    run(&mut app, &wakes, agent, "/unfold 1", |world| {
        !folded(agent, 1)(world)
    });
    ask(&mut app, &wakes, agent, "hotfix: add a regression test");
    let sent = last_request(&models.agent);
    assert!(
        sent.contains("fix the tokenizer") && !sent.contains("Pratt"),
        "{sent}"
    );

    // The current chapter cannot be folded; another can.
    run(&mut app, &wakes, agent, "/fold 3", |world| {
        refused(world, "current one")
    });
    assert!(refused(app.world(), "current one") && !folded(agent, 3)(app.world()));
    run(&mut app, &wakes, agent, "/fold 2", folded(agent, 2));
    let before = shape(app.world(), agent);
    assert_eq!(
        before,
        vec![
            parser,
            ("Release".to_owned(), Some((4, 6)), true),
            ("Hotfix".to_owned(), Some((7, 8)), false),
        ]
    );
    app.update();
    drop(app);

    let (restarted, _, agent) = start(&home, &id, &models);
    assert_eq!(shape(restarted.world(), agent), before);
}
