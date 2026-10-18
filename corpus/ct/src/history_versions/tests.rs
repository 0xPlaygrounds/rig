use std::sync::mpsc::Receiver;

use rig_basics::compaction::{CompactionDefault, CompactionPlugin, summary_author};
use rig_core::test_utils::MockCompletionModel;
use rig_harness::settings::Settings;
use rig_harness_test_support::{
    Collected, Ended, MemoryStore, app_on, collect, connect, first_agent, log_ended, reply,
    run_until,
};

use super::*;

/// The kernel, compaction, the plugin and a `/drop-turn N` command on
/// `store` with `settings`, after the first frame restored the session;
/// its agent answered by `model`.
fn start(
    store: &MemoryStore,
    model: &MockCompletionModel,
    settings: &str,
) -> (App, Receiver<()>, Entity) {
    let (mut app, wakes) = app_on(store);
    app.insert_resource(Settings::from_toml(settings));
    app.add_plugins((
        AgentPlugin,
        SessionLogPlugin,
        CompactionPlugin,
        HistoryVersionsPlugin,
    ))
    .add_command("drop-turn", "Remove turn N", drop_turn)
    .init_resource::<Ended>()
    .add_observer(log_ended);
    collect::<Notice>(&mut app);
    app.finish();
    app.update();
    let agent = first_agent(app.world_mut()).unwrap_or(Entity::PLACEHOLDER);
    if let Some(connection) = connect(model) {
        app.world_mut().entity_mut(agent).insert(connection);
    }
    (app, wakes, agent)
}

/// `/drop-turn N`: takes out the user's message N and its reply.
fn drop_turn(In(args): In<CommandArgs>, mut agents: Query<&mut Conversation>) {
    let (Ok(mut conversation), Ok(n)) = (agents.get_mut(args.agent), args.args.parse::<usize>())
    else {
        return;
    };
    let at = n.saturating_sub(1) * 2;
    conversation.splice(at..at + 2, [], Author::plugin("drop-turn"));
}

/// Sends `text` and runs frames until its turn ended.
fn ask(app: &mut App, wakes: &Receiver<()>, agent: Entity, text: &str) {
    let ended = app.world().resource::<Ended>().0.len();
    app.world_mut()
        .trigger(Deliver::new(agent, text, DeliveryMode::Steer));
    run_until(app, wakes, |world| {
        world.resource::<Ended>().0.len() > ended
    });
}

/// Runs `/line` and the frame after it.
fn run(app: &mut App, agent: Entity, line: &str) {
    app.world_mut().trigger(RunCommand::new(agent, line));
    app.update();
    app.update();
}

/// Each message of `agent`, as a short text; a summary as `SUMMARY`.
fn texts(app: &App, agent: Entity) -> Vec<String> {
    let Some(conversation) = app.world().get::<Conversation>(agent) else {
        return Vec::new();
    };
    conversation
        .entries()
        .map(|entry| match entry.message {
            _ if entry.author == Some(&summary_author()) => "SUMMARY".to_owned(),
            Message::Assistant(assistant) => assistant.text(),
            message => message.rag_text().unwrap_or_default(),
        })
        .collect()
}

/// Agent `agent`'s versions: number and author.
fn versions(app: &App, agent: Entity) -> Vec<(u32, String)> {
    let versions = app.world().get::<ConvVersions>(agent);
    let all = versions.map(|it| it.versions.iter());
    all.into_iter()
        .flatten()
        .map(|version| (version.n, version.by.label()))
        .collect()
}

fn last_notice(app: &App) -> String {
    let notices = &app.world().resource::<Collected<Notice>>().0;
    notices
        .last()
        .map(|notice| notice.text.clone())
        .unwrap_or_default()
}

fn owned(texts: &[&str]) -> Vec<String> {
    texts.iter().map(|text| (*text).to_owned()).collect()
}

#[test]
fn rewrites_are_logged_restored_and_undone() {
    let store = MemoryStore::default();
    let mut script: Vec<_> = (1..=5).map(|turn| reply(&format!("A{turn}"))).collect();
    script.extend([reply("The user asked Q1 and Q3."), reply("A6")]);
    let model = MockCompletionModel::from_stream_turns(script);
    let (mut app, wakes, agent) = start(&store, &model, "");
    for question in ["Q1", "Q2", "Q3", "Q4 TOKEN4", "Q5"] {
        ask(&mut app, &wakes, agent, question);
    }

    // A command drops turn 2, then a compaction condenses turns 1 and 3.
    run(&mut app, agent, "drop-turn 2");
    // It keeps the newest 20 estimated tokens: the last two turns.
    app.world_mut()
        .resource_mut::<CompactionDefault>()
        .0
        .keep_asked = 20;
    app.world_mut().trigger(RunCommand::new(agent, "compact"));
    run_until(&mut app, &wakes, |world| {
        let compacted = world
            .get::<Conversation>(agent)
            .is_some_and(|conversation| {
                conversation
                    .entries()
                    .any(|entry| entry.author == Some(&summary_author()))
            });
        compacted && world.get::<ActiveTurn>(agent).is_none()
    });
    ask(&mut app, &wakes, agent, "Q6");
    let compacted = owned(&["SUMMARY", "Q4 TOKEN4", "A4", "Q5", "A5", "Q6", "A6"]);
    assert_eq!(texts(&app, agent), compacted);

    run(&mut app, agent, "conv-log");
    let by = |n, who: &str| (n, format!("plugin {who}"));
    assert_eq!(
        versions(&app, agent),
        vec![by(1, "drop-turn"), by(2, "compaction")]
    );
    let log = last_notice(&app);
    assert!(
        log.starts_with("2 version(s)") && log.contains("#2") && log.contains("compaction"),
        "{log}"
    );
    run(&mut app, agent, "conv-diff 1");
    assert!(
        last_notice(&app).contains("- user: Q2"),
        "{}",
        last_notice(&app)
    );

    // Undo the compaction: turn 2 stays removed, turn 6 stays at the end.
    run(&mut app, agent, "conv-restore 2");
    let uncompacted = owned(&[
        "Q1",
        "A1",
        "Q3",
        "A3",
        "Q4 TOKEN4",
        "A4",
        "Q5",
        "A5",
        "Q6",
        "A6",
    ]);
    assert_eq!(texts(&app, agent), uncompacted);
    // Undo the undo.
    run(&mut app, agent, "conv-restore 3");
    assert_eq!(texts(&app, agent), compacted);

    let kept = versions(&app, agent);
    assert_eq!(kept.len(), 4, "{kept:?}");
    drop(app);

    // Restarted: the log is intact; with one kept, only the newest stays.
    let idle = MockCompletionModel::from_stream_turns(Vec::<
        Vec<rig_core::test_utils::MockStreamEvent>,
    >::new());
    let (mut app, _, agent) = start(&store, &idle, "");
    app.update();
    assert_eq!(versions(&app, agent), kept);
    assert_eq!(texts(&app, agent), compacted);
    drop(app);
    let (mut app, _, agent) = start(&store, &idle, "[conv-history]\nkeep = 1");
    app.update();
    assert_eq!(versions(&app, agent), kept.get(3..).unwrap_or_default());
    assert_eq!(texts(&app, agent), compacted);
}
