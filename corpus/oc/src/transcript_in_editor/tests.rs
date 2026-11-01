use std::path::Path;
use std::sync::Arc;
use std::sync::mpsc::Receiver;

use rig_core::completion::{CompletionResponse, FinishReason, Usage};
use rig_core::effect::Answered;
use rig_core::message::{Origin, ToolFunction, ToolName};
use rig_core::test_utils::{MockCompletionModel, MockStreamEvent};
use rig_core::tool::ExitStatus;
use rig_harness_test_support::{
    AnswerAsks, Answers, Collected, Ended, MemoryStore, app_on, calls, collect, first_agent,
    log_ended, reply, run_until, scratch_home, scripted_models, tui_view,
};

use super::*;

/// How many lines the build log has.
const LOG_LINES: usize = 5_000;

/// The arguments of `shell`.
#[derive(serde::Deserialize, schemars::JsonSchema)]
struct Args {
    /// The command.
    #[expect(dead_code, reason = "the log is the same for every command")]
    command: String,
    /// The exit status to publish, if any.
    #[serde(default)]
    exit: Option<i32>,
}

/// A `shell` that prints a long build log, and publishes the exit status
/// its call asks for.
fn shell(
    called: On<ToolCalled<Args>>,
    mut calls: Query<&mut Decision, ToolCalls>,
    mut commands: Commands,
) {
    let log: Vec<String> = (1..=LOG_LINES)
        .map(|n| format!("build log line {n}"))
        .collect();
    if let Ok(mut call) = calls.get_mut(called.call) {
        call.decide(
            Decider::plugin("shell"),
            ToolResult::success(log.join("\n").into()),
        );
    }
    if let Some(code) = called.args.exit {
        let mut context = ToolContext::new();
        context.insert_result(ExitStatus::Exited(code)).ok();
        let published = Published(context.result_context());
        commands.entity(called.call).insert(published);
    }
}

/// The kernel, the terminal view without a terminal, a front that answers
/// questions from a script, and the plugin, on `store`, answered by
/// `model`; files under `home`.
fn start(
    store: &MemoryStore,
    model: &MockCompletionModel,
    home: &Path,
) -> Option<(App, Receiver<()>, Entity)> {
    let (mut app, wakes) = app_on(store);
    app.insert_resource(scripted_models([("test/model", model)]))
        .insert_resource(RigHome(Home::at(home.to_path_buf())))
        .insert_resource(Scratch(Some(home.join("scratch"))))
        .insert_resource(EditorTerminal)
        .add_plugins((
            AgentPlugin,
            SessionLogPlugin,
            TranscriptInEditorPlugin,
            AnswerAsks,
        ))
        .add_open_tool("shell", "Run a command.", ToolOptions::default(), shell)
        .init_resource::<Ended>()
        .add_observer(log_ended);
    tui_view(&mut app);
    collect::<Notice>(&mut app);
    app.finish();
    app.update();
    let agent = first_agent(app.world_mut())?;
    let model = "test/model".to_owned();
    app.world_mut().trigger(SetModel {
        entity: agent,
        model,
        kind: ChoiceKind::ForThisRun,
    });
    Some((app, wakes, agent))
}

/// Asks `agent` something and runs frames until its turn ended.
fn ask(app: &mut App, wakes: &Receiver<()>, agent: Entity, text: &str) {
    let ended = app.world().resource::<Ended>().0.len();
    app.world_mut()
        .trigger(Deliver::new(agent, text, DeliveryMode::Queue));
    run_until(app, wakes, |world| {
        world.resource::<Ended>().0.len() > ended
    });
}

/// Types `line` with `editor` as the editor and runs frames until a
/// notice says `until`; the notices written meanwhile.
fn typed(
    app: &mut App,
    wakes: &Receiver<()>,
    (agent, line): (Entity, &str),
    editor: &str,
    until: &str,
) -> String {
    let world = app.world_mut();
    world.resource_mut::<Collected<Notice>>().0.clear();
    world.insert_resource(EditHistorySettings {
        editor: Some(editor.to_owned()),
    });
    world.trigger(Deliver::new(agent, line, DeliveryMode::Queue));
    let seen = |world: &World| {
        let notices = &world.resource::<Collected<Notice>>().0;
        notices.iter().any(|notice| notice.text.contains(until))
    };
    run_until(app, wakes, seen);
    let notices = &app.world().resource::<Collected<Notice>>().0;
    notices
        .iter()
        .map(|notice| notice.text.clone())
        .collect::<Vec<_>>()
        .join("\n")
}

fn messages(app: &App, agent: Entity) -> String {
    let conversation = app.world().get::<Conversation>(agent);
    let messages = conversation.map_or_else(Vec::new, |it| it.messages().to_vec());
    serde_json::to_string(&messages).unwrap_or_default()
}

fn ids(app: &App, agent: Entity) -> Vec<MessageId> {
    let conversation = app.world().get::<Conversation>(agent);
    conversation.map_or_else(Vec::new, |it| it.entries().map(|entry| entry.id).collect())
}

fn paired(app: &App, agent: Entity) -> bool {
    let conversation = app.world().get::<Conversation>(agent);
    conversation.is_some_and(|it| it.validate().is_ok())
}

/// A shell script at `home/name` running `body` on the file `$f`; the
/// editor command that runs it.
fn script(home: &Path, name: &str, body: &str) -> String {
    let path = home.join(name);
    let text = format!("f=\"$1\"\nd=\"$(dirname \"$0\")\"\n{body}\n");
    assert!(std::fs::write(&path, text).is_ok());
    format!("sh '{}'", path.display())
}

/// Rewrites the file with awk `program`.
fn awk(program: &str) -> String {
    format!("awk '{program}' \"$f\" > \"$f.new\" && mv \"$f.new\" \"$f\"")
}

#[cfg(unix)]
#[test]
fn the_conversation_is_edited_in_the_editor_validated_kept_and_undone() {
    let home = scratch_home("oc-transcript-in-editor");
    let home = home.root().to_path_buf();
    assert!(std::fs::create_dir_all(&home).is_ok());
    let model = MockCompletionModel::from_stream_turns([
        reply("Hello."),
        calls(
            "c1",
            "shell",
            serde_json::json!({ "command": "cat build.log" }),
        ),
        reply("The bug is in parser.rs."),
        reply("Turn three answer."),
        reply("Fourth answer."),
    ]);
    let store = MemoryStore::default();
    let started = start(&store, &model, &home);
    assert!(started.is_some(), "the session has one agent");
    let Some((mut app, wakes, agent)) = started else {
        return;
    };
    ask(&mut app, &wakes, agent, "First question.");
    ask(&mut app, &wakes, agent, "Find the bug.");
    ask(&mut app, &wakes, agent, "Third question.");
    let before = messages(&app, agent);
    let before_ids = ids(&app, agent);
    assert!(
        before.contains("Turn three answer") && before.contains("build log line 4999"),
        "{before}"
    );

    // Invalid JSON in a call reopens the editor with the error; quitting
    // that unchanged leaves the conversation as it was.
    let invalid = script(
        &home,
        "invalid.sh",
        &format!(
            "if [ -e \"$d/count\" ]; then head -n 1 \"$f\" > \"$d/seen\"; exit 0; fi\ntouch \"$d/count\"\n{}",
            awk(r#"{ sub(/"cat build.log"/, "oops"); print }"#)
        ),
    );
    let said = typed(
        &mut app,
        &wakes,
        (agent, "/edit-history"),
        &invalid,
        "nothing changed",
    );
    let seen = std::fs::read_to_string(home.join("seen")).unwrap_or_default();
    assert!(
        seen.starts_with("# ERROR:") && seen.contains("not valid JSON"),
        "{seen}\n{said}"
    );
    assert_eq!(messages(&app, agent), before, "{said}");

    // The claim rewritten, the log trimmed, turn 3 deleted.
    let edit = script(
        &home,
        "edit.sh",
        &awk(r#"/^## \[/ { skip = 0 }
/^## \[5\] tool-result shell c1/ { print; print ""; print "(log trimmed)"; print ""; skip = 1; next }
/^## \[7\]/ { exit }
skip { next }
{ gsub(/parser\.rs/, "lexer.rs"); print }"#),
    );
    let said = typed(
        &mut app,
        &wakes,
        (agent, "/edit-history"),
        &edit,
        "Edited the conversation",
    );
    assert!(said.contains("6 messages, 8 before"), "{said}");
    ask(&mut app, &wakes, agent, "Fourth question.");
    let sent = model
        .requests()
        .last()
        .map(|it| serde_json::to_string(it).unwrap_or_default());
    let sent = sent.unwrap_or_default();
    for wanted in [
        "lexer.rs",
        "(log trimmed)",
        "cat build.log",
        "Fourth question",
    ] {
        assert!(sent.contains(wanted), "{wanted}: {sent}");
    }
    for gone in [
        "parser.rs",
        "Third question",
        "Turn three answer",
        "build log line",
    ] {
        assert!(!sent.contains(gone), "{gone}: {sent}");
    }
    assert!(paired(&app, agent), "every call has its result");
    drop(app);

    // The edit is in the session's log; undo, after a restart, brings
    // turn 3 back and keeps the turn since.
    let restarted = start(
        &store,
        &MockCompletionModel::from_stream_turns([reply("Unused.")]),
        &home,
    );
    assert!(restarted.is_some(), "the agent was restored");
    let Some((mut app, wakes, agent)) = restarted else {
        return;
    };
    let restored = messages(&app, agent);
    assert!(
        restored.contains("lexer.rs") && !restored.contains("Third question"),
        "{restored}"
    );
    let said = typed(
        &mut app,
        &wakes,
        (agent, "/edit-history undo"),
        "true",
        "Restored",
    );
    let undone = messages(&app, agent);
    assert!(
        ids(&app, agent).starts_with(&before_ids),
        "the messages put back keep their ids"
    );
    for wanted in [
        "Third question",
        "Turn three answer",
        "parser.rs",
        "build log line 4999",
        "Fourth question",
    ] {
        assert!(undone.contains(wanted), "{wanted}: {undone}\n{said}");
    }
    assert!(paired(&app, agent));

    // Deleting only the result asks; "delete both" takes the call too.
    app.world_mut()
        .insert_resource(Answers::new([Answer::choice(DELETE_BOTH)]));
    let drop_result = script(
        &home,
        "drop-result.sh",
        &awk(r#"/^## \[/ { skip = 0 } /^## \[5\] tool-result/ { skip = 1 } !skip { print }"#),
    );
    let said = typed(
        &mut app,
        &wakes,
        (agent, "/edit-history"),
        &drop_result,
        "Edited the conversation",
    );
    let asked = &app.world().resource::<Answers>().asked;
    assert!(
        asked.iter().any(|title| title == DIALOG_TITLE),
        "{asked:?}\n{said}"
    );
    let left = messages(&app, agent);
    assert!(
        !left.contains("cat build.log") && !left.contains("build log line"),
        "{left}"
    );
    assert!(
        left.contains("Turn three answer") && paired(&app, agent),
        "{left}"
    );

    // The editor line is shell syntax, quotes and all; a failed editor
    // changes nothing and says how it ended.
    let quoted = r#"sh -c 'sed -i.bak "s/Turn three answer/MARKER/" "$1"' sh"#;
    let said = typed(
        &mut app,
        &wakes,
        (agent, "/edit-history"),
        quoted,
        "Edited the conversation",
    );
    assert!(messages(&app, agent).contains("MARKER"), "{said}");
    let marked = messages(&app, agent);
    let said = typed(
        &mut app,
        &wakes,
        (agent, "/edit-history"),
        r#"sh -c 'echo junk >> "$1"; exit 3' sh"#,
        "nothing changed",
    );
    assert!(
        said.contains("/edit-history: the editor failed (exit status 3); nothing changed."),
        "{said}"
    );
    assert_eq!(messages(&app, agent), marked);
    std::fs::remove_dir_all(&home).ok();
}

/// Deleting the first of a reply's two calls moves the second into its
/// place, so the reply gets a new id: what was kept at the old places,
/// such as a call's run time or a stop's cut list, matches nothing.
#[cfg(unix)]
#[test]
fn a_reply_whose_calls_an_edit_moved_gets_a_new_id() {
    let home = scratch_home("oc-transcript-in-editor-moved");
    let home = home.root().to_path_buf();
    assert!(std::fs::create_dir_all(&home).is_ok());
    let command = |command: &str| serde_json::json!({ "command": command });
    let model = MockCompletionModel::from_stream_turns([
        vec![
            MockStreamEvent::tool_call("c1", "shell", command("one")),
            MockStreamEvent::tool_call("c2", "shell", command("two")),
            MockStreamEvent::final_response_with_default_usage(),
        ],
        reply("Both ran."),
    ]);
    let store = MemoryStore::default();
    let started = start(&store, &model, &home);
    assert!(started.is_some(), "the session has one agent");
    let Some((mut app, wakes, agent)) = started else {
        return;
    };
    ask(&mut app, &wakes, agent, "Run both.");
    let reply = |app: &App| {
        let conversation = app.world().get::<Conversation>(agent);
        let entries = conversation.into_iter().flat_map(|it| it.entries());
        let calls = |entry: &EntryRef<'_>| matches!(entry.message, Message::Assistant(turn) if turn.tool_calls().next().is_some());
        entries
            .filter(calls)
            .map(|entry| entry.id)
            .collect::<Vec<_>>()
    };
    let before = reply(&app);
    assert_eq!(before.len(), 1, "{}", messages(&app, agent));

    app.world_mut()
        .insert_resource(Answers::new([Answer::choice(DELETE_BOTH)]));
    let drop_first = script(
        &home,
        "drop-first.sh",
        &awk(
            r#"/^## \[/ { skip = 0 } /^## \[3\] tool-result shell c1/ { skip = 1 } !skip { print }"#,
        ),
    );
    let said = typed(
        &mut app,
        &wakes,
        (agent, "/edit-history"),
        &drop_first,
        "Edited the conversation",
    );
    let left = messages(&app, agent);
    assert!(
        !left.contains("\"one\"") && left.contains("\"two\""),
        "{left}\n{said}"
    );
    assert!(paired(&app, agent));
    let after = reply(&app);
    assert_eq!(after.len(), 1, "{left}");
    assert_ne!(after, before, "the reply got a new id");
    std::fs::remove_dir_all(&home).ok();
}

/// Taking out a reply's only call, and its result, merges the next reply
/// into it: that reply's call then sits where the taken call was, so the
/// merged reply gets a new id and nothing kept for the old place is its,
/// also when the two calls have the same id, as replies may reuse one.
#[cfg(unix)]
#[test]
fn a_reply_a_merge_moves_a_call_into_gets_a_new_id() {
    merge_moves_a_call("c2");
    merge_moves_a_call("c1");
}

/// [`a_reply_a_merge_moves_a_call_into_gets_a_new_id`], with `second` the
/// id of the second reply's call.
#[cfg(unix)]
fn merge_moves_a_call(second: &str) {
    let home = scratch_home(&format!("oc-transcript-in-editor-merged-{second}"));
    let home = home.root().to_path_buf();
    assert!(std::fs::create_dir_all(&home).is_ok());
    let command = |command: &str| serde_json::json!({ "command": command });
    let model = MockCompletionModel::from_stream_turns([
        vec![
            MockStreamEvent::text("Looking."),
            MockStreamEvent::tool_call("c1", "shell", command("one")),
            MockStreamEvent::final_response_with_default_usage(),
        ],
        vec![
            MockStreamEvent::tool_call(second, "shell", command("two")),
            MockStreamEvent::final_response_with_default_usage(),
        ],
        reply("Both ran."),
    ]);
    let store = MemoryStore::default();
    let started = start(&store, &model, &home);
    assert!(started.is_some(), "the session has one agent");
    let Some((mut app, wakes, agent)) = started else {
        return;
    };
    ask(&mut app, &wakes, agent, "Run both.");
    let replies = |app: &App| {
        let conversation = app.world().get::<Conversation>(agent);
        let entries = conversation.into_iter().flat_map(|it| it.entries());
        let replies = entries.filter(|entry| matches!(entry.message, Message::Assistant(_)));
        replies.map(|entry| entry.id).collect::<Vec<_>>()
    };
    let before = replies(&app);
    assert_eq!(before.len(), 3, "{}", messages(&app, agent));

    app.world_mut()
        .insert_resource(Answers::new([Answer::choice(DELETE_BOTH)]));
    let drop_first = script(
        &home,
        "drop-first.sh",
        &awk(r#"/^## \[/ { skip = 0 }
/^```call shell c1$/ && !gone { fence = 1; gone = 1; next }
fence { if (/^```$/) fence = 0; next }
/^## \[3\] tool-result shell c1$/ && !cut { skip = 1; cut = 1 }
!skip { print }"#),
    );
    let said = typed(
        &mut app,
        &wakes,
        (agent, "/edit-history"),
        &drop_first,
        "Edited the conversation",
    );
    let left = messages(&app, agent);
    assert!(
        !left.contains("\"one\"") && left.contains("\"two\"") && left.contains("Looking."),
        "{left}\n{said}"
    );
    let after = replies(&app);
    assert_eq!(after.len(), 2, "{left}");
    let merged = after.first().copied();
    assert!(
        merged.is_some_and(|id| !before.contains(&id)),
        "{before:?} {after:?}"
    );
    std::fs::remove_dir_all(&home).ok();
}

/// Answers the first model call with a reply asking for `asked`, each a
/// `shell` call's arguments, under one id: a live stream renames a
/// repeated id, an answer decided for the call does not.
fn answer_once(
    asked: Vec<(&'static str, serde_json::Value)>,
) -> impl FnMut(Query<&mut Decision, (With<TurnCall>, ReadyToStart)>, Local<bool>) {
    move |mut calls, mut answered| {
        for mut decision in &mut calls {
            let call = |(id, args): &(&str, serde_json::Value)| {
                let function = ToolFunction::new(ToolName::new("shell").ok()?, args.clone());
                Some(AssistantContent::ToolCall(ToolCall::from_wire(
                    *id, function,
                )))
            };
            let choice: Option<Vec<_>> = asked.iter().map(call).collect();
            let origin = Origin::new("test.api", "test", "model");
            let reply = choice.map(|choice| {
                let reply = CompletionResponse::new(choice, Usage::default(), origin, Value::Null);
                Ok(Answered::Completion(Arc::new(
                    reply.with_finish_reason(FinishReason::ToolCalls),
                )))
            });
            if let Some(reply) = reply.filter(|_| !*answered) {
                *answered = decision.decide(Decider::plugin("replay"), reply);
            }
        }
    }
}

/// An edit that takes out one of two calls sharing an id, or merges their
/// reply into another, is refused, as the file can't say which call is
/// which: the editor opens again with why, and the calls keep what each
/// published. Taking out a call at that reply's end is no such edit.
#[cfg(unix)]
#[test]
fn an_edit_is_refused_when_it_moves_merges_or_takes_out_calls_sharing_an_id() {
    let home = scratch_home("oc-transcript-in-editor-repeated");
    let home = home.root().to_path_buf();
    assert!(std::fs::create_dir_all(&home).is_ok());
    let model = MockCompletionModel::from_stream_turns([reply("Hello."), reply("Both ran.")]);
    let store = MemoryStore::default();
    let started = start(&store, &model, &home);
    assert!(started.is_some(), "the session has one agent");
    let Some((mut app, wakes, agent)) = started else {
        return;
    };
    let asked = vec![
        ("c9", serde_json::json!({ "command": "one" })),
        ("c9", serde_json::json!({ "command": "two", "exit": 7 })),
        ("c10", serde_json::json!({ "command": "three", "exit": 5 })),
    ];
    ask(&mut app, &wakes, agent, "Hi.");
    app.add_systems(AgentPass, answer_once(asked).in_set(Stage::Gate));
    ask(&mut app, &wakes, agent, "Run both.");
    // The id of each reply with calls, and the exit status each call
    // published.
    let exits = |app: &App| {
        let conversation = app.world().get::<Conversation>(agent);
        let entries: Vec<_> = conversation
            .into_iter()
            .flat_map(|it| it.entries())
            .collect();
        let replies = entries.windows(2).filter_map(|pair| {
            let [reply, results] = pair else { return None };
            let Message::Assistant(turn) = reply.message else {
                return None;
            };
            let calls: Vec<_> = CallRef::calls(reply.id, turn).collect();
            let facts = results
                .facts
                .of(calls.iter().map(|(at, call)| (*at, &call.id)));
            let exits = facts
                .into_iter()
                .map(|facts| facts.and_then(|facts| facts.get::<ExitStatus>().ok().flatten()));
            (!calls.is_empty()).then(|| (reply.id, exits.collect::<Vec<_>>()))
        });
        replies.collect::<Vec<_>>()
    };
    let before = exits(&app);
    let ran: Vec<_> = before.iter().map(|(_, exits)| exits.clone()).collect();
    assert_eq!(
        ran,
        [vec![
            None,
            Some(ExitStatus::Exited(7)),
            Some(ExitStatus::Exited(5))
        ]]
    );

    let drop_first = script(
        &home,
        "drop-first.sh",
        &format!(
            "if [ -e \"$d/count\" ]; then head -n 1 \"$f\" > \"$d/seen\"; exit 0; fi\ntouch \"$d/count\"\n{}",
            awk(r#"/^## \[/ { skip = 0 }
/^```call shell c9$/ && !gone { fence = 1; gone = 1; next }
fence { if (/^```$/) fence = 0; next }
/^## \[5\] tool-result shell c9$/ && !cut { skip = 1; cut = 1 }
!skip { print }"#)
        ),
    );
    app.world_mut()
        .insert_resource(Answers::new([Answer::choice(DELETE_BOTH)]));
    let said = typed(
        &mut app,
        &wakes,
        (agent, "/edit-history"),
        &drop_first,
        "nothing changed",
    );
    let seen = std::fs::read_to_string(home.join("seen")).unwrap_or_default();
    assert!(
        seen.starts_with("# ERROR:") && seen.contains("`c9`"),
        "{seen}\n{said}"
    );
    assert_eq!(exits(&app), before, "{said}");

    let drop_unique = script(
        &home,
        "drop-unique.sh",
        &awk(r#"/^## \[/ { skip = 0 }
/^```call shell c10$/ { fence = 1; next }
fence { if (/^```$/) fence = 0; next }
/^## \[5\] tool-result shell c10$/ { skip = 1 }
!skip { print }"#),
    );
    let said = typed(
        &mut app,
        &wakes,
        (agent, "/edit-history"),
        &drop_unique,
        "Edited the conversation",
    );
    // Nothing moved, so the reply keeps its id and each call its own.
    let after = exits(&app);
    let ran: Vec<_> = after.iter().map(|(_, exits)| exits.clone()).collect();
    assert_eq!(ran, [vec![None, Some(ExitStatus::Exited(7))]], "{said}");
    let ids = |replies: &[(MessageId, Vec<Option<ExitStatus>>)]| {
        replies.iter().map(|(id, _)| *id).collect::<Vec<_>>()
    };
    assert_eq!(ids(&after), ids(&before), "{said}");

    // Taking out the question before the reply would merge it into the
    // answer before: refused too.
    std::fs::remove_file(home.join("count")).ok();
    let drop_question = script(
        &home,
        "drop-question.sh",
        &format!(
            "if [ -e \"$d/count\" ]; then head -n 1 \"$f\" > \"$d/seen\"; exit 0; fi\ntouch \"$d/count\"\n{}",
            awk(r#"/^## \[/ { skip = 0 } /^## \[3\] user$/ { skip = 1 } !skip { print }"#)
        ),
    );
    let said = typed(
        &mut app,
        &wakes,
        (agent, "/edit-history"),
        &drop_question,
        "nothing changed",
    );
    let seen = std::fs::read_to_string(home.join("seen")).unwrap_or_default();
    assert!(
        seen.starts_with("# ERROR:") && seen.contains("`c9`"),
        "{seen}\n{said}"
    );
    assert_eq!(exits(&app), after, "{said}");
    std::fs::remove_dir_all(&home).ok();
}
