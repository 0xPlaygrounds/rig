use std::sync::Arc;
use std::sync::mpsc::Receiver;

use rig_basics::telemetry::effect_log::{EffectLog, EffectLogPlugin};
use rig_core::test_utils::MockCompletionModel;
use rig_harness::harness_protocol::SessionId;
use rig_harness::rig_core::tool::ToolOutput;
use rig_harness_test_support::MemoryStore;
use rig_harness_test_support::{
    Collected, Ended, app_in, app_on, calls, collect, commit_message, connect, first_agent,
    log_ended, reply, run_until, say, scratch_home, scripted_models,
};

use super::*;

/// A summary in the template, told apart by `marker`.
fn template(marker: &str) -> String {
    format!(
        "## Goal\nShip the API ({marker}).\n\n## Constraints the user gave\n- Keep the tests.\n\n\
         ## Decisions and why\n- **Dropped the cache**: it served stale data.\n\n\
         ## Files touched\n- `src/f1.rs`: read for the API.\n\n## Current state\nWorks.\n\n\
         ## Open questions / next steps\n1. Release."
    )
}

/// What the user says in turn `turn`.
fn says(turn: usize) -> String {
    format!("user says {turn}: do step {turn}")
}

/// What `read` is called with.
#[derive(Deserialize, schemars::JsonSchema)]
struct ReadArgs {
    path: String,
}

/// `read` answers with the path it was given.
fn read_file(called: On<ToolCalled<ReadArgs>>, mut calls: Query<&mut Decision, ToolCalls>) {
    let text = format!("contents of {}", called.args.path);
    if let Ok(mut call) = calls.get_mut(called.call) {
        call.decide(
            Decider::plugin("fh-structured-compaction"),
            ToolResult::success(ToolOutput::text(text)),
        );
    }
}

/// Runs frames until one more turn of `app` ended than `before`.
fn until_ended(app: &mut App, wakes: &Receiver<()>, before: usize) {
    run_until(app, wakes, |world| {
        world.resource::<Ended>().0.len() > before
    });
}

/// Each request `model` got, as JSON.
fn sent(model: &MockCompletionModel) -> Vec<String> {
    let requests = model.requests();
    let json = requests.iter().map(serde_json::to_string);
    json.map(Result::unwrap_or_default).collect()
}

/// Where `part` is in `text`, past the end when it is not.
fn at(text: &str, part: &str) -> usize {
    text.find(part).unwrap_or(usize::MAX)
}

#[test]
fn compact_writes_the_template_keeps_the_last_exchanges_and_builds_on_the_last_summary() {
    let home = scratch_home("fh-structured-compaction");
    let (mut app, wakes) = app_in(&home, &SessionId::generate());
    let mut script = Vec::new();
    for turn in 1..=6 {
        let path = serde_json::json!({ "path": format!("src/f{turn}.rs") });
        script.push(calls(&format!("c{turn}"), "read", path));
        script.push(reply(&format!("answer {turn}")));
    }
    script.push(reply("answer 7"));
    let model = MockCompletionModel::from_stream_turns(script);
    let summarizer = MockCompletionModel::from_stream_turns([
        reply(&template("FIRST-SUMMARY")),
        reply(&template("SECOND-SUMMARY")),
    ]);
    let summarizer_name = "test/summarizer";
    app.insert_resource(scripted_models([
        ("test/agent", &model),
        (summarizer_name, &summarizer),
    ]))
    .add_plugins((AgentPlugin, SessionLogPlugin, EffectLogPlugin))
    .add_plugins(StructuredCompactionPlugin)
    .insert_resource(StructuredCompaction {
        model: Some(summarizer_name.to_owned()),
        keep_user_messages: 3,
    })
    .add_open_tool("read", "Read a file.", ToolOptions::default(), read_file)
    .init_resource::<Ended>()
    .add_observer(log_ended);
    app.finish();
    app.update();
    let agent = first_agent(app.world_mut());
    assert!(agent.is_some(), "an agent");
    let Some(agent) = agent else {
        return;
    };
    let model_name = "test/agent".to_owned();
    app.world_mut().trigger(SetModel {
        entity: agent,
        model: model_name,
        kind: ChoiceKind::ForThisRun,
    });
    let deliver = |app: &mut App, text: String| say(app, &wakes, agent, &text);
    let compact = |app: &mut App, line: &str| {
        let before = app.world().resource::<Ended>().0.len();
        app.world_mut().trigger(RunCommand::new(agent, line));
        until_ended(app, &wakes, before);
    };
    for turn in 1..=6 {
        deliver(&mut app, says(turn));
    }

    compact(&mut app, "compact keep the API details");
    deliver(&mut app, says(7));
    let asked = sent(&summarizer);
    let first = asked.first().map(String::as_str).unwrap_or_default();
    assert!(first.contains("keep the API details"), "{first}");
    assert!(first.contains("## Decisions and why"), "{first}");
    let requests = sent(&model);
    let next = requests.last().map(String::as_str).unwrap_or_default();
    // The summary, then the user's last three messages, each where it was with its whole
    // exchange, and the new message.
    let order = [
        at(next, "FIRST-SUMMARY"),
        at(next, &says(4)),
        at(next, "contents of src/f4.rs"),
        at(next, &says(5)),
        at(next, "contents of src/f5.rs"),
        at(next, &says(6)),
        at(next, "contents of src/f6.rs"),
        at(next, &says(7)),
    ];
    assert!(
        order.is_sorted() && order.last() < Some(&usize::MAX),
        "{order:?} {next}"
    );
    for turn in 1..=3 {
        assert!(!next.contains(&says(turn)), "{next}");
        assert!(
            !next.contains(&format!("contents of src/f{turn}.rs")),
            "{next}"
        );
    }

    compact(&mut app, "compact");
    let asked = sent(&summarizer);
    let second = asked.get(1).map(String::as_str).unwrap_or_default();
    assert!(second.contains("<previous-summary>"), "{second}");
    assert!(second.contains("FIRST-SUMMARY"), "{second}");

    // The saved session still has every message the user sent.
    app.update();
    let log = app.world().get_resource::<EffectLog>();
    let records = log.and_then(|log| log.read().ok()).unwrap_or_default();
    let texts: Vec<String> = records
        .iter()
        .filter_map(|record| MessageRecord::read(&record.kind))
        .map(|message| message.text)
        .collect();
    for turn in 1..=6 {
        assert!(texts.contains(&says(turn)), "{texts:?}");
    }
}

#[test]
fn a_summary_without_its_headings_twice_falls_back_and_the_turn_goes_on() {
    let (mut app, wakes) = app_on(&MemoryStore::default());
    let summarizer =
        MockCompletionModel::from_stream_turns([reply("Just some prose."), reply("More prose.")]);
    // The fallback is the default summary of the messages the plugin planned.
    let model = MockCompletionModel::from_stream_turns([
        reply("## Goal\nThe default summary (DEFAULT-SUMMARY)."),
        reply("done"),
    ]);
    app.insert_resource(scripted_models([("test/summarizer", &summarizer)]))
        .add_plugins((AgentPlugin, SessionLogPlugin, StructuredCompactionPlugin))
        .insert_resource(StructuredCompaction {
            model: Some("test/summarizer".to_owned()),
            keep_user_messages: 3,
        })
        .init_resource::<Ended>()
        .add_observer(log_ended);
    collect::<Notice>(&mut app);
    app.finish();
    app.update();
    let agent = first_agent(app.world_mut());
    assert!(agent.is_some(), "an agent");
    let Some(agent) = agent else {
        return;
    };
    // A conversation near a 40k-token window.
    let connection = connect(&model);
    assert!(connection.is_some(), "the catalog lists the model");
    let Some(mut connection) = connection else {
        return;
    };
    connection.spec = Arc::new(connection.spec.as_ref().clone().with_context_window(40_000));
    app.world_mut().entity_mut(agent).insert(connection);
    let long = "word ".repeat(4_000);
    let mut messages = vec![Message::user("What does the crate do?")];
    for turn in 0..6 {
        messages.push(Message::assistant(format!("{turn}: {long}")));
        let mut question = Message::user(format!("question {turn}"));
        if let Message::User { content } = &mut question {
            content.push(rig_core::message::UserContent::image_base64(
                "aGk=", None, None,
            ));
        }
        messages.push(question);
    }
    messages.push(Message::assistant("The last answer."));
    for message in messages {
        assert!(commit_message(app.world_mut(), agent, message).is_ok());
    }

    let next = Deliver::new(agent, "And the next question?", DeliveryMode::Steer);
    app.world_mut().trigger(next);
    until_ended(&mut app, &wakes, 0);

    assert_eq!(sent(&summarizer).len(), 2, "the template, asked twice");
    let requests = sent(&model);
    let default = requests.first().map(String::as_str).unwrap_or_default();
    assert!(
        default.contains("You are a context summarization assistant"),
        "{default}"
    );
    assert!(
        default.contains("question 3") && !default.contains("question 4"),
        "the last three of the user's messages stay: {default}"
    );
    assert!(
        matches!(
            app.world().resource::<Ended>().0.as_slice(),
            [TurnOutcome::Answered(_)]
        ),
        "{:?}",
        app.world().resource::<Ended>().0
    );
    let last = requests.last().map(String::as_str).unwrap_or_default();
    assert!(last.contains("DEFAULT-SUMMARY"), "{last}");
    assert!(last.contains("aGk="), "the kept messages' images stay");
    let notices = &app.world().resource::<Collected<Notice>>().0;
    assert!(
        notices
            .iter()
            .any(|notice| notice.text.contains("default way")),
        "{notices:?}"
    );
}

/// A compaction replaces only the messages before the ones kept: with nothing before them it is
/// refused, and a summary that comes back after the conversation changed goes in nowhere. Either
/// way the entries stay the ones they were, the image of a kept message included.
#[test]
fn a_refused_or_failed_compaction_changes_nothing() {
    /// Rewrites the first entry while a summary is reviewed, once switched on.
    #[derive(Resource, Default)]
    struct EditMeanwhile(bool);
    let edit = |calls: Query<&ForAgent, (ReadyToSettle, With<Summarizing>)>,
                mut agents: Query<&mut Conversation>,
                mut on: ResMut<EditMeanwhile>| {
        for &ForAgent(agent) in &calls {
            if let (true, Ok(mut conversation)) = (on.0, agents.get_mut(agent)) {
                on.0 = false;
                conversation.edit_each(Author::plugin("test"), |entry| {
                    (entry.index == 0).then(|| Message::user("edited meanwhile"))
                });
            }
        }
    };
    let (mut app, wakes) = app_on(&MemoryStore::default());
    let model = MockCompletionModel::from_stream_turns([reply("answer 3")]);
    let summarizer = MockCompletionModel::from_stream_turns([reply(&template("SUMMARY"))]);
    app.insert_resource(scripted_models([
        ("test/agent", &model),
        ("test/summarizer", &summarizer),
    ]))
    .add_plugins((AgentPlugin, SessionLogPlugin, StructuredCompactionPlugin))
    .insert_resource(StructuredCompaction {
        model: Some("test/summarizer".to_owned()),
        keep_user_messages: 2,
    })
    .init_resource::<EditMeanwhile>()
    .add_systems(AgentPass, edit.in_set(ReviewSystems::Edit))
    .init_resource::<Ended>()
    .add_observer(log_ended);
    app.finish();
    app.update();
    let agent = first_agent(app.world_mut());
    assert!(agent.is_some(), "an agent");
    let Some(agent) = agent else {
        return;
    };
    let model = "test/agent".to_owned();
    let kind = ChoiceKind::ForThisRun;
    let entity = agent;
    app.world_mut().trigger(SetModel {
        entity,
        model,
        kind,
    });
    let mut pictured = Message::user(says(2));
    if let Message::User { content } = &mut pictured {
        let png = Some(rig_core::message::ImageMediaType::PNG);
        content.push(rig_core::message::UserContent::image_base64(
            "aGk=", png, None,
        ));
    }
    let long = || Message::assistant("word ".repeat(16_001));
    for message in [Message::user(says(1)), long(), pictured, long()] {
        assert!(commit_message(app.world_mut(), agent, message).is_ok());
    }
    let entries = |app: &App| -> Vec<Entry> {
        let conversation = app.world().get::<Conversation>(agent);
        let all = conversation.map(|it| (0..it.messages().len()).filter_map(|at| it.entry(at)));
        all.into_iter().flatten().collect()
    };
    let compact = |app: &mut App| {
        let before = app.world().resource::<Ended>().0.len();
        app.world_mut().trigger(RunCommand::new(agent, "compact"));
        until_ended(app, &wakes, before);
    };

    // Nothing comes before the two kept messages: refused, no summary asked for.
    let before = entries(&app);
    compact(&mut app);
    assert_eq!(entries(&app), before);
    assert_eq!(sent(&summarizer).len(), 0);

    // A summary whose messages changed while it was written goes in nowhere.
    say(&mut app, &wakes, agent, &says(3));
    let before = entries(&app);
    app.world_mut().resource_mut::<EditMeanwhile>().0 = true;
    compact(&mut app);
    let after = entries(&app);
    assert_eq!(sent(&summarizer).len(), 1);
    assert_eq!(after.get(1..), before.get(1..));
    let json = serde_json::to_string(&after).unwrap_or_default();
    let kept = json.contains("edited meanwhile") && json.contains("aGk=");
    assert!(kept && !json.contains("SUMMARY"), "{json}");
}

rig_defaults_test_support::stages_ordered_test!(StructuredCompactionPlugin);
