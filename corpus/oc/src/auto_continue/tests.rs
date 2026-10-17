use std::path::PathBuf;

use rig_basics::coding_tools::WriteTool;
use rig_basics::telemetry::{TurnUsage, UsagePlugin};
use rig_core::completion::{FinishReason, Usage};
use rig_core::message::Message as Said;
use rig_core::operation::Finish;
use rig_core::test_utils::{MockCompletionModel, MockStreamEvent};
use rig_harness_test_support::{
    Collected, Ended, MemoryStore, ToolOutputs, app_on, calls, collect, log_ended,
    log_tool_outputs, reply, run_until, scripted_models, tui_view,
};
use rig_tui::{MainPane, ShowAgent, TuiTestScreen};
use serde_json::json;

use super::*;

/// The end of a reply, stopped for `reason`, with `input` and `output`
/// tokens.
fn end(reason: FinishReason, input: u64, output: u64) -> MockStreamEvent {
    MockStreamEvent::FinalResponse(Finish {
        usage: Usage::new().input_tokens(input).output_tokens(output),
        reason: Some(reason),
        ..Finish::default()
    })
}

/// A reply of `text` cut at the output limit.
fn cut_text(text: &str) -> Vec<MockStreamEvent> {
    vec![
        MockStreamEvent::text(text),
        end(FinishReason::Length, 100, 20),
    ]
}

/// One turn asking `ask` of the scripted `model`, run to its end in a
/// project at `root`.
fn run_turn(model: &MockCompletionModel, root: PathBuf, ask: &str) -> (App, Entity) {
    let (mut app, wakes) = app_on(&MemoryStore::default());
    app.insert_resource(scripted_models([("test/model", model)]))
        .insert_resource(ProjectRoot(root))
        .add_plugins((AgentPlugin, SessionLogPlugin, WriteTool, UsagePlugin))
        .add_plugins(AutoContinuePlugin)
        .init_resource::<ToolOutputs>()
        .add_observer(log_tool_outputs)
        .init_resource::<Ended>()
        .add_observer(log_ended)
        .insert_resource(TuiTestScreen::new(80, 30));
    tui_view(&mut app);
    collect::<Notice>(&mut app);
    let agent = app
        .world_mut()
        .spawn((Agent, ModelChoice("test/model".to_owned())))
        .id();
    app.finish();
    app.cleanup();
    app.update();
    app.world_mut()
        .trigger(Deliver::new(agent, ask, DeliveryMode::Steer));
    run_until(&mut app, &wakes, |world| {
        !world.resource::<Ended>().0.is_empty()
    });
    (app, agent)
}

/// The assistant messages of `agent`'s conversation.
fn answers(world: &World, agent: Entity) -> Vec<AssistantMessage> {
    let messages = world.get::<Conversation>(agent).map(|it| {
        it.messages()
            .iter()
            .map(|m| (**m).clone())
            .collect::<Vec<_>>()
    });
    let messages = messages.unwrap_or_default().into_iter();
    messages
        .filter_map(|message| match message {
            Said::Assistant(answer) => Some(answer),
            _ => None,
        })
        .collect()
}

/// Where the seams' marks are: in a saved message's text, at a byte.
fn marks(world: &World) -> Vec<Option<usize>> {
    let mut main = world.try_query_filtered::<&TranscriptView, With<MainPane>>();
    let view = main.as_mut().and_then(|main| main.single(world).ok());
    let marks = view.into_iter().flat_map(|view| view.marks.iter());
    let at = |mark: &TextMark| matches!(mark.r#in, MarkIn::Message { .. }).then_some(mark.at);
    marks.map(at).collect()
}

/// The text blocks of `answer`, joined.
fn text_of(answer: &AssistantMessage) -> String {
    answer
        .content
        .iter()
        .filter_map(|block| match block {
            AssistantContent::Text(text) => Some(text.text.as_str()),
            _ => None,
        })
        .collect()
}

/// A scratch project directory beside the test binary.
fn project(name: &str) -> PathBuf {
    let exe = std::env::current_exe().unwrap_or_default();
    let root = exe.with_file_name(name);
    std::fs::remove_dir_all(&root).ok();
    std::fs::create_dir_all(&root).ok();
    root
}

#[test]
fn a_cut_reply_is_continued_and_joined_at_the_seam() {
    let model = MockCompletionModel::from_stream_turns([
        cut_text("The three steps are: 1. parse 2. val"),
        vec![
            MockStreamEvent::text("2. validate 3. emit."),
            end(FinishReason::Stop, 130, 10),
        ],
    ]);
    let (mut app, agent) = run_turn(&model, project("auto-continue-text"), "Steps?");

    let requests = model.requests();
    let second = format!("{:?}", requests.get(1));
    assert_eq!(requests.len(), 2);
    assert!(second.contains("1. parse 2. val"), "{second}");
    assert!(
        second.contains("Continue exactly where you stopped"),
        "{second}"
    );
    let answers = answers(app.world(), agent);
    let texts: Vec<String> = answers.iter().map(text_of).collect();
    assert_eq!(
        texts,
        ["The three steps are: 1. parse 2. validate 3. emit."]
    );
    // The turn's usage is both requests'.
    let mut spent = app.world_mut().query::<&TurnUsage>();
    let spent: Vec<_> = spent.iter(app.world()).map(|it| it.0).collect();
    let tokens = spent
        .first()
        .map(|it| (it.calls, it.tokens.input_tokens, it.tokens.output_tokens));
    assert_eq!(tokens, Some((2, Some(230), Some(30))), "{spent:?}");
    // The seam is marked where the continuation began, its repeat said once.
    assert_eq!(marks(app.world()), [Some(30)]);
    app.world_mut().trigger(ShowAgent { entity: agent });
    app.update();
    let screen = app.world().resource::<TuiTestScreen>().rows().join("\n");
    assert!(
        screen.contains("parse  (continued) 2. validate 3. emit."),
        "{screen}"
    );
}

#[test]
fn a_cut_write_is_never_run_and_asked_for_again() {
    let content = "line 1\nline 2\nline 3\n";
    let model = MockCompletionModel::from_stream_turns([
        vec![
            // A complete call before the cut one is kept and runs once.
            MockStreamEvent::tool_call("w0", "write", json!({ "path": "a.txt", "content": "a" })),
            MockStreamEvent::tool_call_name_delta("w1", "write"),
            MockStreamEvent::tool_call_arguments_delta(
                "w1",
                r#"{"path": "big.txt", "content": "line 1\nli"#,
            ),
            end(FinishReason::Length, 100, 20),
        ],
        calls(
            "w2",
            "write",
            json!({ "path": "big.txt", "content": content }),
        ),
        reply("Wrote big.txt."),
    ]);
    let root = project("auto-continue-write");
    let (app, agent) = run_turn(&model, root.clone(), "Write big.txt.");

    let written = std::fs::read_to_string(root.join("big.txt")).unwrap_or_default();
    let first = std::fs::read_to_string(root.join("a.txt")).unwrap_or_default();
    std::fs::remove_dir_all(&root).ok();
    assert_eq!((written.as_str(), first.as_str()), (content, "a"));
    let mut ran: Vec<&String> = app.world().resource::<ToolOutputs>().0.keys().collect();
    ran.sort();
    assert_eq!(ran, ["w0", "w2"]);
    let stored: Vec<Vec<String>> = answers(app.world(), agent)
        .iter()
        .map(|answer| {
            let calls = answer.content.iter().filter_map(|block| match block {
                AssistantContent::ToolCall(call) => Some(call.id.to_string()),
                _ => None,
            });
            calls.collect()
        })
        .collect();
    // One message for the joined reply, the kept call first; then the answer.
    assert_eq!(stored, [vec!["w0", "w2"], vec![]]);
    let second = format!("{:?}", model.requests().get(1));
    assert!(second.contains("several `edit` calls"), "{second}");
    assert!(second.contains("(write) were kept"), "{second}");
}

#[test]
fn a_reply_cut_too_often_is_kept_with_a_notice() {
    let model = MockCompletionModel::from_stream_turns(
        ["alpha ", "bravo ", "charlie ", "delta "].map(cut_text),
    );
    let (app, agent) = run_turn(&model, project("auto-continue-limit"), "Go on.");

    assert_eq!(model.request_count(), 4);
    let texts: Vec<String> = answers(app.world(), agent).iter().map(text_of).collect();
    assert_eq!(texts, ["alpha bravo charlie delta "]);
    let notices = &app.world().resource::<Collected<Notice>>().0;
    let told = format!("{notices:?}");
    assert!(told.contains("kept as it is"), "{told}");
    assert_eq!(marks(app.world()), [Some(6), Some(12), Some(20)]);
}

#[test]
fn stages_are_ordered_with_the_defaults() {
    rig_defaults_test_support::assert_stages_ordered(AutoContinuePlugin);
}
