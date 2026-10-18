use std::sync::mpsc::Receiver;

use rig_core::message::Message;
use rig_core::test_utils::MockCompletionModel;
use rig_harness::settings::Settings;
use rig_harness_test_support::{
    AnswerAsks, Answers, MemoryStore, app_on, calls, first_agent, reply, run_until, scripted_models,
};
use serde_json::json;

use super::*;

/// The project file the interviewer reads.
const FILE: &str = "fn serve() { /* every route */ }";

/// The brief the scripted interviewer writes.
const BRIEF: &str =
    "Goal: rate limiting.\nConstraints: none.\nDecisions: see answers.\nOpen points: none.";

/// `read`, answered with [`FILE`].
fn read(called: On<ToolCalled<serde_json::Value>>, mut calls: Query<&mut Decision, ToolCalls>) {
    if let Ok(mut call) = calls.get_mut(called.call) {
        call.decide(Decider::plugin("test"), ToolResult::success(FILE.into()));
    }
}

/// An interviewer that reads a file, asks two questions and writes
/// [`BRIEF`].
fn interviewer() -> MockCompletionModel {
    MockCompletionModel::from_stream_turns([
        calls("r", "read", json!({ "path": "src/server.rs" })),
        calls(
            "q1",
            CLARIFY_ASK,
            json!({ "question": "Which routes?", "default": "all routes" }),
        ),
        calls(
            "q2",
            CLARIFY_ASK,
            json!({ "question": "What limit?", "default": "100 per minute", "options": ["10 per second"] }),
        ),
        reply(BRIEF),
    ])
}

/// The kernel, this plugin (which adds subagents itself) and `read`, with `settings`, the
/// models `test/main` and `test/interviewer`, and a front answering
/// `answers` unless it is `None`; and the main agent, on `test/main`.
fn start(
    settings: &str,
    main: &MockCompletionModel,
    interviewer: &MockCompletionModel,
    answers: Option<Vec<Answer>>,
) -> (App, Receiver<()>, Entity) {
    let (mut app, wakes) = app_on(&MemoryStore::default());
    let reads = ToolOptions {
        footprint: Some(Touches::Reads(Scope::Anything)),
    };
    app.insert_resource(Settings::from_toml(settings))
        .insert_resource(scripted_models([
            ("test/main", main),
            ("test/interviewer", interviewer),
        ]))
        .add_plugins((
            AgentPlugin,
            SessionLogPlugin,
            ClarifyFirstPlugin,
        ))
        .add_open_tool("read", "Read a file.", reads, read)
        .init_resource::<Asked>()
        .add_observer(count_asks);
    if let Some(answers) = answers {
        app.insert_resource(Answers::new(answers))
            .add_plugins(AnswerAsks);
    }
    app.finish();
    app.update();
    let agent = first_agent(app.world_mut()).unwrap_or(Entity::PLACEHOLDER);
    app.world_mut().trigger(SetModel {
        entity: agent,
        model: "test/main".to_owned(),
    });
    (app, wakes, agent)
}

/// The title of every question asked.
#[derive(Resource, Default)]
struct Asked(Vec<String>);

fn count_asks(added: On<Add<Ask>>, asks: Query<&Ask>, mut asked: ResMut<Asked>) {
    if let Ok(ask) = asks.get(added.entity) {
        asked.0.push(ask.title.clone());
    }
}

/// The interviewer's settings.
const SETTINGS: &str = "[clarify]\nmodel = \"test/interviewer\"";

/// Sends `line` from the user and runs until the main agent answered.
fn run(app: &mut App, wakes: &Receiver<()>, agent: Entity, main: &MockCompletionModel, line: &str) {
    app.world_mut()
        .trigger(Deliver::new(agent, line, DeliveryMode::Steer));
    run_until(app, wakes, |world| {
        main.request_count() >= 1 && world.get::<ActiveTurn>(agent).is_none()
    });
}

/// What the main agent's first request sent, and how many messages besides
/// the system prompt; `clarify_ask` is never offered to it.
fn first_request(main: &MockCompletionModel) -> (String, usize) {
    let requests = main.requests();
    let first = requests.first();
    let offered = first.is_some_and(|request| {
        (request.tools.iter()).any(|tool| tool.name.as_str() == CLARIFY_ASK)
    });
    assert!(!offered, "the main agent is not offered {CLARIFY_ASK}");
    let text = first.map(|request| format!("{:?}", request.chat_history));
    let count = first.map_or(0, |request| {
        let history = request.chat_history.iter();
        history
            .filter(|message| !matches!(&***message, Message::System { .. }))
            .count()
    });
    (text.unwrap_or_default(), count)
}

#[test]
fn the_agent_starts_with_the_task_and_a_brief_of_the_answers() {
    let main = MockCompletionModel::from_stream_turns([reply("On it.")]);
    let interviewer = interviewer();
    let answers = vec![
        Answer::text("only /api routes"),
        Answer::choice("default"),
        Answer::choice("start"),
    ];
    let (mut app, wakes, agent) = start(SETTINGS, &main, &interviewer, Some(answers));
    run(&mut app, &wakes, agent, &main, "/clarify add rate limiting");

    assert_eq!(interviewer.request_count(), 4, "read, two questions, brief");
    let asked = &app.world().resource::<Asked>().0;
    assert_eq!(
        asked,
        &["Which routes?", "What limit?", "Start with this brief?"]
    );
    let (sent, messages) = first_request(&main);
    assert_eq!(messages, 1, "one user message, no interview: {sent}");
    for part in [
        "add rate limiting",
        "Goal: rate limiting.",
        "Which routes? -> only /api routes",
        "What limit? -> 100 per minute (the suggested default)",
    ] {
        assert!(sent.contains(part), "{part} in {sent}");
    }
    let conversation = app.world().get::<Conversation>(agent);
    assert_eq!(conversation.map(|it| it.messages().len()), Some(2));
}

#[test]
fn just_start_ends_the_interview_with_the_defaults() {
    let main = MockCompletionModel::from_stream_turns([reply("On it.")]);
    let interviewer = interviewer();
    let (mut app, wakes, agent) = start(
        SETTINGS,
        &main,
        &interviewer,
        Some(vec![Answer::choice("start")]),
    );
    run(&mut app, &wakes, agent, &main, "/clarify add rate limiting");

    let asked = &app.world().resource::<Asked>().0;
    assert_eq!(
        asked,
        &["Which routes?"],
        "nothing asked after just start, no review"
    );
    let (sent, _) = first_request(&main);
    for part in [
        "Which routes? -> all routes (default; the user said just start)",
        "What limit? -> 100 per minute (default, not asked)",
        "the user said just start",
    ] {
        assert!(sent.contains(part), "{part} in {sent}");
    }
}

#[test]
fn a_run_nobody_answers_asks_nothing_and_says_so() {
    let main = MockCompletionModel::from_stream_turns([reply("On it.")]);
    let interviewer = interviewer();
    let (mut app, wakes, agent) = start(SETTINGS, &main, &interviewer, None);
    run(&mut app, &wakes, agent, &main, "/clarify add rate limiting");

    assert!(
        app.world().resource::<Asked>().0.is_empty(),
        "no question shown"
    );
    let (sent, _) = first_request(&main);
    assert!(
        sent.contains("questions were skipped and their defaults used"),
        "{sent}"
    );
    assert!(
        sent.contains("Which routes? -> all routes (default, not asked)"),
        "{sent}"
    );
}

#[test]
fn auto_mode_lets_a_short_prompt_straight_through() {
    let main = MockCompletionModel::from_stream_turns([reply("Fixed.")]);
    let interviewer = interviewer();
    let settings = "[clarify]\nmode = \"auto\"\nmodel = \"test/interviewer\"";
    let (mut app, wakes, agent) = start(settings, &main, &interviewer, Some(Vec::new()));
    run(&mut app, &wakes, agent, &main, "fix the typo in README");

    assert_eq!(interviewer.request_count(), 0);
    let (sent, messages) = first_request(&main);
    assert_eq!(messages, 1, "{sent}");
    assert!(
        sent.contains("fix the typo in README") && !sent.contains("Brief"),
        "{sent}"
    );
    let long = ClarifySettings {
        mode: ClarifyMode::Auto,
        ..ClarifySettings::default()
    };
    assert!(long.wants_interview("Please build an export button"));
}

#[test]
fn stages_are_ordered_with_the_defaults() {
    rig_defaults_test_support::assert_stages_ordered(ClarifyFirstPlugin);
}
