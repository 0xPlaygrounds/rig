use bevy_ecs::system::RunSystemOnce;
use rig_harness::rig_core::effect::Answered;
use rig_harness::rig_core::error::{ErrorKind, ErrorReport};
use rig_harness::rig_core::serde_json::json;
use rig_harness::rig_core::test_utils::{MockCompletionModel, MockStreamEvent};
use rig_harness::rig_core::tool::{FileChange, Lines, ToolOutput, ToolResultContext};
use rig_harness_test_support::{
    Collected, Ended, MemoryStore, app_on, calls, collect, log_ended, run_until, scripted_models,
};

use super::*;
use crate::coding_tools::{AttachPlugin, EditTool, ShellTool};
use crate::telemetry::effect_log::EffectLogPlugin;

/// Pieces of the secrets [`a_session_with_the_plugin_holds_no_key`] puts
/// in, each long enough that no other text holds it.
const FRAGMENTS: [&str; 7] = [
    // Typed in the message.
    "abcdefghijklmnop1234",
    // In a file the message names.
    "hunter2hunter2",
    // Printed by a command: a token and a private key block's body.
    "printedprintedprinted",
    "MIIEsecretbody",
    // On a line next to the one an edit changed.
    "edited12345678",
    // Added to every tool result by a reviewer, in place.
    "reviewed12345678",
    // Streamed by the model in pieces.
    "uvwxyz0123456789",
];

/// What the session's agent ends up holding, and what its two logs and the
/// streamed text hold, after a turn that meets a secret everywhere one
/// comes in: the message, a file it names, a command's output, an edit's
/// diff, a reviewer's edit and the model's reply, streamed in pieces.
fn a_session_meeting_secrets(redacting: bool) -> (String, String) {
    let exe = std::env::current_exe().unwrap_or_default();
    let dir = exe.with_file_name(format!("redact-{redacting}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).ok();
    std::fs::write(dir.join("secret.txt"), "PASSWORD=hunter2hunter2\n").ok();
    std::fs::write(dir.join("notes.txt"), "one\nTOKEN=edited12345678\nthree\n").ok();
    // It prints secrets its own text does not hold: the model's call is
    // redacted as it streams, so a key in it would run redacted.
    let printed = "printf 'GITHUB_TO''KEN=ghp_''print''edprint''edprinted00000\\n\
        -----BEG''IN RSA PRIVATE KEY-----\\nMIIE''secretbody\\n-----END RSA PRIVATE KEY-----\\n\
        after\\n'";
    let edits = json!([{ "old_text": "one", "new_text": "ONE" }]);
    let pieces = [
        "Use GITHUB_TOKEN=gh",
        "p_abcdefghijklmnopqrst",
        "uvwxyz0123456789 now.",
    ];
    let mut streamed: Vec<_> = pieces
        .iter()
        .map(|piece| MockStreamEvent::text(*piece))
        .collect();
    streamed.push(MockStreamEvent::final_response_with_default_usage());
    let model = MockCompletionModel::from_stream_turns([
        calls("s1", "shell", json!({ "command": printed })),
        calls("e1", "edit", json!({ "path": "notes.txt", "edits": edits })),
        streamed,
    ]);
    let store = MemoryStore::default();
    let (mut app, wakes) = app_on(&store);
    app.insert_resource(scripted_models([("test/model", &model)]))
        .add_plugins((
            AgentPlugin,
            SessionLogPlugin,
            EffectLogPlugin,
            AttachPlugin,
            ShellTool,
            EditTool,
        ))
        .init_resource::<Ended>()
        .add_observer(log_ended)
        .add_systems(AgentPass, review.in_set(ReviewSystems::Edit));
    collect::<StreamDelta>(&mut app);
    if redacting {
        app.add_plugins(RedactPlugin);
    }
    app.finish();
    app.cleanup();
    app.update();
    let world = app.world_mut();
    let agent = (
        Agent,
        ModelChoice("test/model".to_owned()),
        WorkDir(dir.clone()),
    );
    let agent = world.spawn(agent).id();
    let said = "Go, with my key sk-proj-abcdefghijklmnop1234 and @secret.txt";
    world.trigger(Deliver::new(agent, said, DeliveryMode::Queue));
    run_until(&mut app, &wakes, |world| {
        !world.resource::<Ended>().0.is_empty()
    });
    app.update();
    std::fs::remove_dir_all(&dir).ok();
    let world = app.world();
    let shown: String = world
        .resource::<Collected<StreamDelta>>()
        .0
        .iter()
        .filter_map(|delta| match &delta.event {
            StreamEvent::Text { text, .. } => Some(text.as_str()),
            _ => None,
        })
        .collect();
    let held = world
        .get::<Conversation>(agent)
        .map(|conversation| format!("{:?}", conversation.messages()))
        .unwrap_or_default();
    let logs = store.logs().unwrap_or_default();
    let mut saved: Vec<u8> = logs
        .iter()
        .flat_map(|log| store.read_log(log).unwrap_or_default())
        .collect();
    let effects = store.read_file(rig_cassette::effect_log::jsonl::SESSION_EFFECT_LOG);
    saved.extend(effects.unwrap_or_default());
    let all = format!("{shown}\n{held}\n{}", String::from_utf8_lossy(&saved));
    (shown, all)
}

/// A reviewer that adds a key to every tool result, in place.
fn review(mut answers: Query<&mut Outcome, (ToolCalls, Added<Outcome>)>) {
    for mut outcome in &mut answers {
        if let Ok(Answered::ToolResult { result }) = &outcome.0 {
            let text = result.output().render();
            let edited = format!("{text}\nAPI_KEY=reviewed12345678");
            let result = result.clone().with_output(ToolOutput::text(edited));
            outcome.0 = Ok(Answered::ToolResult { result });
        }
    }
}

/// With the plugin, no secret reaches the streamed text, the conversation,
/// the session log or the effect log; without it, every one does.
#[test]
fn a_session_with_the_plugin_holds_no_key() {
    let (shown, all) = a_session_meeting_secrets(true);
    assert_eq!(shown, "Use GITHUB_TOKEN=[REDACTED] now.");
    for fragment in FRAGMENTS {
        assert!(!all.contains(fragment), "{fragment} in {all}");
    }
    assert!(all.contains("TOKEN=[REDACTED]"), "{all}");
    let (shown, all) = a_session_meeting_secrets(false);
    assert_eq!(
        shown,
        "Use GITHUB_TOKEN=ghp_abcdefghijklmnopqrstuvwxyz0123456789 now."
    );
    for fragment in FRAGMENTS {
        assert!(all.contains(fragment), "{fragment} not in {all}");
    }
}

/// A text event of `part`.
fn text(part: u32, text: &str) -> Relayed {
    Relayed::Item(Item::Event(StreamEvent::Text {
        part: Part::new(part),
        text: text.to_owned(),
    }))
}

/// A reply's stream comes out redacted: a key in a text part stays held
/// while another part opens and streams, a tool call's arguments go out
/// whole at its end, and what a part holds goes out before its end.
#[test]
fn a_reply_streams_redacted_however_its_parts_interleave() {
    let mut redacting = redact_stream();
    let event = |event: StreamEvent| Relayed::Item(Item::Event(event));
    let end = |part: u32| {
        event(StreamEvent::End {
            part: Part::new(part),
            content: AssistantContent::text("ended"),
        })
    };
    let items = [
        text(0, "Here: ghp_abcdefghijklmnopqrst"),
        event(StreamEvent::Reasoning {
            part: Part::new(1),
            text: "thinking ".to_owned(),
        }),
        text(0, "uvwxyz0123456789 done"),
        event(StreamEvent::Arguments {
            part: Part::new(2),
            json: r#"{"command":"echo \nGITHUB_TOKEN=ghp_abcdefghij"#.to_owned(),
        }),
        event(StreamEvent::Arguments {
            part: Part::new(2),
            json: r#"klmnopqrstuvwxyz0123456789"}"#.to_owned(),
        }),
        end(2),
        end(0),
    ];
    let out: Vec<StreamEvent> = items
        .into_iter()
        .flat_map(&mut redacting)
        .filter_map(|item| match item {
            Relayed::Item(Item::Event(event)) => Some(event),
            _ => None,
        })
        .collect();
    let joined = |wanted: fn(&StreamEvent) -> Option<&str>| -> String {
        out.iter().filter_map(wanted).collect()
    };
    let shown = joined(|event| match event {
        StreamEvent::Text { text, .. } => Some(text),
        _ => None,
    });
    let thought = joined(|event| match event {
        StreamEvent::Reasoning { text, .. } => Some(text),
        _ => None,
    });
    assert_eq!(
        (shown.as_str(), thought.as_str()),
        ("Here: [REDACTED] done", "thinking ")
    );
    let arguments: Vec<usize> = (0..out.len())
        .filter(|&at| matches!(out.get(at), Some(StreamEvent::Arguments { .. })))
        .collect();
    assert_eq!(arguments.len(), 1, "{out:?}");
    let ends_after = arguments.first().and_then(|&at| out.get(at + 1));
    assert!(matches!(ends_after, Some(StreamEvent::End { part, .. }) if *part == Part::new(2)));
    let arguments = joined(|event| match event {
        StreamEvent::Arguments { json, .. } => Some(json),
        _ => None,
    });
    assert_eq!(arguments, r#"{"command":"echo \nGITHUB_TOKEN=[REDACTED]"}"#);
    // The last of the text goes out before its part's end.
    let last_text = out
        .iter()
        .rposition(|event| matches!(event, StreamEvent::Text { .. }));
    let text_end = out
        .iter()
        .rposition(|event| matches!(event, StreamEvent::End { part, .. } if *part == Part::new(0)));
    assert!(last_text < text_end, "{out:?}");
}

/// An answer, the one a stop would give, the facts a call published and a
/// line a view hands back are redacted as they come; a changed file's old
/// text is not.
#[test]
fn whatever_comes_in_is_redacted_as_it_comes() {
    let mut world = World::new();
    world.add_observer(redact_outcome);
    world.add_observer(redact_if_stopped);
    world.add_observer(redact_facts);
    let key = "sk-proj-abcdefghijklmnop1234";
    let failed = Outcome(Err(ErrorReport::new(
        ErrorKind::Cancelled,
        format!("no {key}"),
    )));
    let printed = ToolResult::success(ToolOutput::text(format!("printed {key}")));
    let block = "-----BEGIN RSA PRIVATE KEY-----\nMIIEow\n-----END RSA PRIVATE KEY-----\n";
    let hunk = |lines: Vec<HunkLine>| Hunk {
        old: Lines { start: 1, count: 1 },
        new: Lines { start: 1, count: 1 },
        lines,
    };
    let same = hunk(vec![
        HunkLine::Same(format!("{key}\n")),
        HunkLine::Added("new\n".to_owned()),
    ]);
    let keyed = hunk(
        block
            .lines()
            .map(|line| HunkLine::Added(format!("{line}\n")))
            .collect(),
    );
    let changes = FilesChanged(vec![
        FileChange::modified("a.rs", 1, 1)
            .with_hunks(vec![same])
            .with_before_text(format!("old {key}\n")),
        FileChange::created("id_rsa", 3).with_hunks(vec![keyed]),
    ]);
    let mut context = ToolContext::new();
    assert!(context.insert_result(changes).is_ok());
    let entities = [
        world.spawn(failed).id(),
        world.spawn(IfStopped(Outcome::from(printed))).id(),
        world.spawn(Published(context.result_context())).id(),
    ];
    let [failed, stopped, published] = entities;
    let error = world
        .get::<Outcome>(failed)
        .and_then(|outcome| outcome.0.as_ref().err());
    assert_eq!(
        error.map(|report| report.message.as_str()),
        Some("no [REDACTED]")
    );
    let stopped = world
        .get::<IfStopped>(stopped)
        .map(|stopped| stopped.0.tool_result());
    let stopped = stopped.map(|result| result.output().render());
    assert_eq!(stopped.as_deref(), Some("printed [REDACTED]"));
    let facts: Option<ToolResultContext> = world
        .get::<Published>(published)
        .map(|facts| facts.0.clone());
    let changed = facts.and_then(|facts| facts.get::<FilesChanged>().ok().flatten());
    let changed = changed.unwrap_or_default();
    let first = changed.0.first();
    assert_eq!(
        first.and_then(|change| change.before_text.as_deref()),
        Some(format!("old {key}\n").as_str())
    );
    let line = first
        .and_then(|change| change.hunks.first())
        .and_then(|hunk| hunk.lines.first());
    assert_eq!(line.map(HunkLine::text), Some("[REDACTED]\n"));
    // A key block is in no one line: the change keeps no hunks.
    assert_eq!(changed.0.get(1).map(|change| change.hunks.len()), Some(0));

    // A refused command's line, handed back.
    world.init_resource::<Messages<Recalled>>();
    world.write_message(Recalled {
        agent: Entity::PLACEHOLDER,
        text: format!("/missing {key}"),
        why: Some("no such command".to_owned()),
    });
    assert!(world.run_system_once(redact_recalled).is_ok());
    let mut recalled = world.resource_mut::<Messages<Recalled>>();
    let texts: Vec<String> = recalled.drain().map(|recalled| recalled.text).collect();
    assert_eq!(texts, ["/missing [REDACTED]"]);
}

/// A command's output loses its secrets line by line, a private key block
/// through its end line, before the tool sees it.
#[test]
fn a_private_key_block_is_left_out_through_its_end_line() {
    let mut in_key = false;
    let lines = [
        "GITHUB_TOKEN=ghp_abcdefghijklmnopqrstuvwxyz0123456789\n",
        "-----BEGIN RSA PRIVATE KEY-----\n",
        "MIIEsecretbody\n",
        "-----END RSA PRIVATE KEY-----\n",
        "after\n",
    ];
    let kept: Vec<String> = lines
        .iter()
        .map(|line| redact_line(line, &mut in_key))
        .collect();
    assert_eq!(
        kept,
        [
            "GITHUB_TOKEN=[REDACTED]\n",
            "[REDACTED]\n",
            "",
            "",
            "after\n"
        ]
    );
}

/// A message going to its agent is redacted, text and attached file; one a
/// gate took, such as a line answering a question, reaches the gate as typed.
#[test]
fn a_message_is_redacted_unless_a_gate_took_it() {
    let mut world = World::new();
    let key = "sk-proj-abcdefghijklmnop1234";
    let mut message = Incoming::new(format!("my {key}"), DeliveryMode::Queue);
    message.parts.push(MessagePart::Attachment(Attachment {
        label: "secret.txt".to_owned(),
        content: UserContent::text(format!("PASSWORD={key}")),
    }));
    let sent = world.spawn((message, Decision::default())).id();
    let mut taken = Decision::default();
    taken.decide(Decider::Person, Outcome::custom("answer"));
    let answer = Incoming::new(format!("here: {key}"), DeliveryMode::Queue);
    let answer = world.spawn((answer, taken)).id();
    assert!(world.run_system_once(redact_messages).is_ok());
    let texts = |entity| {
        let parts = world
            .get::<Incoming>(entity)
            .map(|message| message.parts.clone());
        let texts = parts
            .unwrap_or_default()
            .into_iter()
            .map(|part| match part {
                MessagePart::Text(text) => text,
                MessagePart::Attachment(attachment) => format!("{:?}", attachment.content),
            });
        texts.collect::<Vec<_>>().join(" ")
    };
    let sent = texts(sent);
    assert!(
        sent.starts_with("my [REDACTED] ") && sent.contains("PASSWORD=[REDACTED]"),
        "{sent}"
    );
    assert!(!sent.contains(key), "{sent}");
    assert_eq!(texts(answer), format!("here: {key}"));
}
