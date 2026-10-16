use std::sync::mpsc::Receiver;

use rig_core::message::{Message, ToolFunction, ToolName, ToolResultContent};
use rig_core::test_utils::MockCompletionModel;
use rig_harness::harness_protocol::{Home, SessionId};
use rig_harness::rig_ecs::session_log::SessionLogPlugin;
use rig_harness_test_support::{
    Collected, Ended, app_in, calls, collect, connect, first_agent, log_ended, reply, run_until,
    scratch_home, scripted_models,
};

use super::*;

/// A plugin's state saved with the session: the turns the agent ended.
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

/// A tool about the build cache.
fn cache_stats(
    called: On<ToolCalled<serde_json::Value>>,
    mut calls: Query<&mut Decision, ToolCalls>,
) {
    if let Ok(mut call) = calls.get_mut(called.call) {
        call.decide(
            Decider::plugin("cache_stats"),
            ToolResult::success("cache hit rate 12%".into()),
        );
    }
}

/// The app on session `id` of `home` with this plugin, summaries on the
/// scripted `cheap` model, after its first frame.
fn start(home: &Home, id: &SessionId, cheap: &MockCompletionModel) -> (App, Receiver<()>) {
    let (mut app, wakes) = app_in(home, id);
    app.insert_resource(Settings::from_toml("[split]\nmodel = \"test/cheap\"\n"))
        .insert_resource(scripted_models([("test/cheap", cheap)]))
        .add_plugins((AgentPlugin, SessionLogPlugin, SplitSessionPlugin))
        .add_open_tool(
            "cache_stats",
            "Build cache statistics.",
            ToolOptions::default(),
            cache_stats,
        )
        .register_required_components::<Agent, Turns>()
        .add_observer(count)
        .init_resource::<Ended>()
        .add_observer(log_ended);
    collect::<Notice>(&mut app);
    collect::<Recalled>(&mut app);
    app.finish();
    app.update();
    (app, wakes)
}

/// Types `text` for `agent` and runs until `done`.
fn say(
    app: &mut App,
    wakes: &Receiver<()>,
    agent: Entity,
    text: &str,
    done: impl Fn(&World) -> bool,
) {
    let line = Deliver::new(agent, text, DeliveryMode::Queue).with_author(Author::user());
    app.world_mut().trigger(line);
    run_until(app, wakes, done);
    app.update();
}

fn ended(count: usize) -> impl Fn(&World) -> bool {
    move |world| world.resource::<Ended>().0.len() >= count
}

/// The main model's six turns; turns 3 and 4 ask the cache tool.
fn six_turn_model(more: &[&str]) -> MockCompletionModel {
    let mut script = vec![
        reply("r1"),
        reply("r2"),
        calls("c3", "cache_stats", serde_json::json!({})),
        reply("the cache key holds a timestamp"),
        calls("c4", "cache_stats", serde_json::json!({})),
        reply("cache key fixed"),
        reply("r5"),
        reply("r6"),
    ];
    script.extend(more.iter().map(|text| reply(text)));
    MockCompletionModel::from_stream_turns(script)
}

/// Runs the six turns on a new agent; the agent.
fn six_turns(app: &mut App, wakes: &Receiver<()>, model: &MockCompletionModel) -> Option<Entity> {
    let agent = first_agent(app.world_mut())?;
    app.world_mut().entity_mut(agent).insert(connect(model)?);
    let lines = [
        "one",
        "two",
        "why does the build cache miss?",
        "fix the cache key",
        "five",
        "six",
    ];
    for (done, line) in (1..).zip(lines) {
        say(app, wakes, agent, line, ended(done));
    }
    Some(agent)
}

fn conversation(app: &App, agent: Entity) -> Conversation {
    app.world()
        .get::<Conversation>(agent)
        .cloned()
        .unwrap_or_default()
}

/// Every message as text: what it says and the tools it calls or answers.
fn texts(messages: &[std::sync::Arc<Message>]) -> Vec<String> {
    messages
        .iter()
        .map(|message| format!("{message:?}"))
        .collect()
}

/// Types `line` and runs until no split runs or waits.
fn split_with(app: &mut App, wakes: &Receiver<()>, agent: Entity, line: &str) {
    say(app, wakes, agent, line, |_| true);
    run_until(app, wakes, move |world| {
        world.get::<PendingSplit>(agent).is_none() && world.get::<Splitting>(agent).is_none()
    });
    app.update();
}

fn notices(app: &App) -> Vec<String> {
    app.world()
        .resource::<Collected<Notice>>()
        .0
        .iter()
        .map(|notice| notice.text.clone())
        .collect()
}

/// The split-off session's main conversation and saved values.
fn copy_of(home: &Home, name: &str, agent: &str) -> Option<(SessionId, SavedAgent)> {
    let id = home.find(name).ok()?;
    let store = StoreOpener::default().open(&home.session(&id)).ok()?;
    let session = Session::read(&*store).ok()?;
    Some((id.clone(), session.agent(agent)?.clone()))
}

#[test]
fn split_moves_turns_three_and_four_undo_puts_them_back_and_the_copy_resumes() {
    let home = scratch_home("ct-split-session");
    let id = SessionId::generate();
    let summaries = [
        reply("The user works on a feature."),
        reply("Topic: build cache\nThe cache key held a timestamp; it is fixed."),
    ];
    let cheap = MockCompletionModel::from_stream_turns(
        summaries
            .iter()
            .chain(&summaries)
            .cloned()
            .collect::<Vec<_>>(),
    );
    let (mut app, wakes) = start(&home, &id, &cheap);
    let main = six_turn_model(&["r7"]);
    let found = six_turns(&mut app, &wakes, &main);
    assert!(found.is_some(), "an agent and a model");
    let Some(agent) = found else {
        return;
    };
    let agent_id = app
        .world()
        .get::<AgentId>(agent)
        .cloned()
        .unwrap_or_default();
    let before = conversation(&app, agent);
    let before_entries: Vec<Entry> = (0..before.messages().len())
        .filter_map(|at| before.entry(at))
        .collect();
    assert_eq!(
        before.messages().len(),
        16,
        "{:#?}",
        texts(before.messages())
    );

    split_with(&mut app, &wakes, agent, "/split 3 4 --name cache");

    // The original: turns 1 and 2, the note, turns 5 and 6.
    let after = conversation(&app, agent);
    let messages = after.messages();
    assert_eq!(
        messages.len(),
        9,
        "{:#?}\n{:#?}",
        texts(messages),
        notices(&app)
    );
    assert_eq!(messages.get(..4), before.messages().get(..4));
    assert_eq!(messages.get(5..), before.messages().get(12..));
    let note = after.entries().nth(4);
    assert!(note.is_some_and(|note| note.author == Some(&Author::plugin(NAME))));
    let note = messages
        .get(4)
        .map(|note| format!("{note:?}"))
        .unwrap_or_default();
    assert!(
        note.contains("Turns 3–4 (build cache) were moved to session `cache`"),
        "{note}"
    );
    assert!(note.contains("timestamp; it is fixed"), "{note}");
    assert!(after.validate().is_ok());
    assert_eq!(app.world().get::<Turns>(agent), Some(&Turns(6)));

    // The new session: a context summary, then turns 3 and 4 verbatim,
    // listed as split from this one.
    let found = copy_of(&home, "cache", &agent_id.0);
    assert!(found.is_some(), "a session named cache");
    let Some((copy_id, copy)) = found else {
        return;
    };
    let copied = copy.conversation.messages();
    assert_eq!(copied.len(), 9, "{:#?}", texts(copied));
    let context = copied
        .first()
        .map(|context| format!("{context:?}"))
        .unwrap_or_default();
    assert!(
        context.contains("The user works on a feature."),
        "{context}"
    );
    assert_eq!(copied.get(1..), before.messages().get(4..12));
    assert!(copy.conversation.validate().is_ok());
    let detail = SessionEntry::read(&home.session(&copy_id), &StoreOpener::default())
        .map(|entry| entry.detail());
    assert!(
        detail
            .as_deref()
            .is_some_and(|detail| detail.contains(&format!("split from {id}"))),
        "{detail:?}"
    );

    // Undo: the original exactly as it was, and no `cache` session.
    say(&mut app, &wakes, agent, "/split undo", ended(6));
    let undone = conversation(&app, agent);
    let undone_entries: Vec<Entry> = (0..undone.messages().len())
        .filter_map(|at| undone.entry(at))
        .collect();
    assert_eq!(undone_entries, before_entries, "{:#?}", notices(&app));
    assert!(
        !undone
            .asides()
            .any(|(_, aside)| aside.kind == MOVED || aside.kind == SPLIT)
    );
    assert!(
        !home.session(&copy_id).path().exists(),
        "{:#?}",
        notices(&app)
    );

    // Split again; the next request reads the note, not turns 3 and 4.
    split_with(&mut app, &wakes, agent, "/split 3 4 --name cache");
    say(&mut app, &wakes, agent, "seven", ended(7));
    let sent = main
        .requests()
        .last()
        .map(|request| format!("{:?}", request.chat_history))
        .unwrap_or_default();
    assert!(sent.contains("were moved to session `cache`"), "{sent}");
    assert!(
        !sent.contains("c3") && !sent.contains("cache hit rate"),
        "{sent}"
    );

    // Resumed, the split-off session works.
    let found = copy_of(&home, "cache", &agent_id.0);
    assert!(found.is_some(), "a session named cache");
    let Some((copy_id, _)) = found else {
        return;
    };
    {
        let idle = MockCompletionModel::from_stream_turns([reply("unused")]);
        let (mut resumed, resumed_wakes) = start(&home, &copy_id, &idle);
        let found = first_agent(resumed.world_mut());
        assert!(found.is_some(), "the agent restored");
        let Some(copy_agent) = found else {
            return;
        };
        let model = MockCompletionModel::from_stream_turns([reply("resumed")]);
        let found = connect(&model);
        assert!(found.is_some(), "a model");
        let Some(connection) = found else {
            return;
        };
        resumed
            .world_mut()
            .entity_mut(copy_agent)
            .insert(connection);
        say(&mut resumed, &resumed_wakes, copy_agent, "go on", ended(1));
        let last = conversation(&resumed, copy_agent)
            .messages()
            .last()
            .cloned();
        assert_eq!(
            last.and_then(|last| last.final_answer()),
            Some("resumed".to_owned())
        );
    }

    // Both changed since: the split stays.
    say(&mut app, &wakes, agent, "/split undo", ended(7));
    // A command's error goes back to the input with the line.
    let refused = app
        .world()
        .resource::<Collected<Recalled>>()
        .0
        .iter()
        .filter_map(|recalled| recalled.why.clone())
        .collect::<Vec<_>>();
    assert!(
        refused
            .iter()
            .any(|why| why.contains("Cannot undo the split")),
        "{refused:#?}"
    );
    assert!(home.session(&copy_id).path().exists());
}

#[test]
fn a_boundary_inside_a_tool_exchange_moves_to_a_whole_turn() {
    let home = scratch_home("ct-split-session-boundary");
    let id = SessionId::generate();
    let cheap = MockCompletionModel::from_stream_turns([
        reply("The user said one."),
        reply("Topic: build cache\nThe cache is fine."),
    ]);
    let (mut app, wakes) = start(&home, &id, &cheap);
    let found = first_agent(app.world_mut());
    assert!(found.is_some(), "an agent");
    let Some(agent) = found else {
        return;
    };
    let model = MockCompletionModel::from_stream_turns([reply("unused")]);
    let found = connect(&model);
    assert!(found.is_some(), "a model");
    let Some(connection) = found else {
        return;
    };
    app.world_mut().entity_mut(agent).insert(connection);
    // Turn 3's record starts at the result of turn 2's call.
    let found = ToolName::new("cache_stats");
    assert!(found.is_ok(), "a tool name");
    let Ok(name) = found else {
        return;
    };
    let call = ToolCall::from_wire("c2", ToolFunction::new(name, serde_json::json!({})));
    let messages = [
        Message::user("one"),
        Message::assistant("r1"),
        Message::user("two"),
        Message::Assistant(AssistantMessage::new(vec![AssistantContent::ToolCall(
            call.clone(),
        )])),
        Message::tool_results(vec![
            call.result(vec![ToolResultContent::text("cache hit rate 12%")]),
        ]),
        Message::assistant("r3"),
        Message::user("four"),
        Message::assistant("r4"),
    ];
    if let Some(mut conversation) = app.world_mut().get_mut::<Conversation>(agent) {
        for (at, message) in messages.into_iter().enumerate() {
            let id = conversation.push(message);
            if [0, 2, 4, 6].contains(&at) {
                let record = TurnRecord {
                    index: at as u64 / 2,
                    start: id,
                    end: Some(TurnEnd::Answered { answer: None }),
                };
                let value = serde_json::to_value(record).unwrap_or_default();
                conversation.add_aside(
                    Aside::new(rig_core::conversation::TURN, Author::rig(), value).after(Some(id)),
                );
            }
        }
    }
    app.update();
    let before = conversation(&app, agent);

    split_with(&mut app, &wakes, agent, "/split 3 --name cache");

    let told = notices(&app);
    assert!(
        told.iter()
            .any(|text| text.contains("inside a tool exchange") && text.contains("turns 2–4")),
        "{told:#?}"
    );
    let after = conversation(&app, agent);
    assert_eq!(after.messages().len(), 3, "{:#?}", texts(after.messages()));
    assert_eq!(after.messages().get(..2), before.messages().get(..2));
    assert!(after.validate().is_ok());
    let agent_id = app
        .world()
        .get::<AgentId>(agent)
        .cloned()
        .unwrap_or_default();
    let found = copy_of(&home, "cache", &agent_id.0);
    assert!(found.is_some(), "a session named cache");
    let Some((_, copy)) = found else {
        return;
    };
    assert_eq!(
        copy.conversation.messages().get(1..),
        before.messages().get(2..)
    );
    assert!(copy.conversation.validate().is_ok());
}

#[test]
fn stages_are_ordered_with_the_defaults() {
    rig_defaults_test_support::assert_stages_ordered(SplitSessionPlugin);
}
