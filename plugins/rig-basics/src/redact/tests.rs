use bevy_ecs::system::RunSystemOnce;
use rig_harness::rig_core::completion::{CompletionRequest, ToolDefinition};
use rig_harness::rig_core::effect::Answered;
use rig_harness::rig_core::error::{ErrorKind, ErrorReport};
use rig_harness::rig_core::message::ToolName;
use rig_harness::rig_core::serde_json::json;
use rig_harness::rig_core::test_utils::{MockCompletionModel, MockStreamEvent};
use rig_harness::rig_core::tool::{FileChange, Lines, ToolOutput, ToolResultContext};
use rig_harness::rig_ecs::tools::ResultFilter;
use rig_harness_test_support::{
    Collected, Ended, MemoryStore, app_on, calls, collect, log_ended, reply, run_until,
    scripted_models,
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
    assert!(
        shown.starts_with("Use GITHUB_TOKEN=‹SECRET_") && shown.ends_with("› now."),
        "{shown}"
    );
    for fragment in FRAGMENTS {
        assert!(!all.contains(fragment), "{fragment} in {all}");
    }
    assert!(all.contains("TOKEN=‹SECRET_"), "{all}");
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
    let mut redacting = redact_stream(Secrets::default());
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
        ("Here: ‹SECRET_1› done", "thinking ")
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
    assert_eq!(arguments, r#"{"command":"echo \nGITHUB_TOKEN=‹SECRET_1›"}"#);
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
/// text, and a plugin's own answer, such as a typed answer to its question,
/// are not.
#[test]
fn whatever_comes_in_is_redacted_as_it_comes() {
    let mut world = World::new();
    world.insert_resource(Secrets::default());
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
    let asked = Outcome(Ok(Answered::Custom {
        payload: json!({ "text": key }),
    }));
    let asked = world.spawn(asked).id();
    let answer = world
        .get::<Outcome>(asked)
        .map(|outcome| format!("{:?}", outcome.0));
    assert!(answer.is_some_and(|answer| answer.contains(key)));
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
        Some("no ‹SECRET_1›")
    );
    let stopped = world
        .get::<IfStopped>(stopped)
        .map(|stopped| stopped.0.tool_result());
    let stopped = stopped.map(|result| result.output().render());
    assert_eq!(stopped.as_deref(), Some("printed ‹SECRET_1›"));
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
    assert_eq!(line.map(HunkLine::text), Some("‹SECRET_1›\n"));
    // A key block is in no one line: the change keeps no hunks.
    assert_eq!(changed.0.get(1).map(|change| change.hunks.len()), Some(0));

    // A refused command's line, handed back.
    world.init_resource::<Messages<Recalled>>();
    world.write_message(Recalled {
        agent: Entity::PLACEHOLDER,
        text: format!("/missing {key}"),
        why: Some(format!("No command `/missing {key}`.")),
    });
    assert!(world.run_system_once(redact_recalled).is_ok());
    let mut recalled = world.resource_mut::<Messages<Recalled>>();
    let texts: Vec<_> = recalled
        .drain()
        .map(|recalled| (recalled.text, recalled.why))
        .collect();
    let refused = Some("No command `/missing ‹SECRET_1›`.".to_owned());
    assert_eq!(texts, [("/missing ‹SECRET_1›".to_owned(), refused)]);
}

/// A command's output loses its secrets line by line; a private key block
/// goes out as one placeholder with its first line, and the whole block is
/// what a call naming it gets.
#[test]
fn a_private_key_block_is_one_placeholder_for_the_whole_block() {
    let (mut block, secrets) = (None, Secrets::default());
    let lines = [
        "GITHUB_TOKEN=ghp_abcdefghijklmnopqrstuvwxyz0123456789\n",
        "-----BEGIN RSA PRIVATE KEY-----\n",
        "MIIEsecretbody\n",
        "-----END RSA PRIVATE KEY-----\n",
        "after\n",
    ];
    let kept: Vec<String> = lines
        .iter()
        .map(|line| redact_line(line, &mut block, &secrets))
        .collect();
    let shown = [
        "GITHUB_TOKEN=‹SECRET_1›\n",
        "‹SECRET_2›\n",
        "",
        "",
        "after\n",
    ];
    assert_eq!(kept, shown);
    let key = lines.get(1..4).map(|key| key.concat()).unwrap_or_default();
    assert_eq!(secrets.reveal("‹SECRET_2›"), key);
    // A block that never ends stops holding lines past its most.
    let opening = "-----BEGIN RSA PRIVATE KEY-----\n";
    assert_eq!(redact_line(opening, &mut block, &secrets), "‹SECRET_3›\n");
    let body = format!("{}\n", "A".repeat(63));
    let held = (0..BLOCK_MOST / body.len()).map(|_| redact_line(&body, &mut block, &secrets));
    assert_eq!(held.collect::<String>(), "");
    assert_eq!(redact_line("ls\n", &mut block, &secrets), "ls\n");
    assert!(secrets.reveal("‹SECRET_3›").len() <= BLOCK_MOST + body.len());
}

/// A message going to its agent is redacted, text, attached file and its
/// label; one a gate took, such as a line answering a question, reaches the
/// gate as typed.
#[test]
fn a_message_is_redacted_unless_a_gate_took_it() {
    let mut world = World::new();
    world.insert_resource(Secrets::default());
    let key = "sk-proj-abcdefghijklmnop1234";
    let mut message = Incoming::new(format!("my {key}"), DeliveryMode::Queue);
    message.parts.push(MessagePart::Attachment(Attachment {
        label: "secret.txt".to_owned(),
        content: UserContent::text(format!("PASSWORD={key}")),
    }));
    let document = message::Document {
        data: message::DocumentData::Text(format!("token {key}")),
        media_type: None,
        additional_params: None,
    };
    message.parts.push(MessagePart::Attachment(Attachment {
        label: format!("notes-{key}.md"),
        content: UserContent::Document(document),
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
                MessagePart::Attachment(attachment) => {
                    format!("{} {:?}", attachment.label, attachment.content)
                }
            });
        texts.collect::<Vec<_>>().join(" ")
    };
    let sent = texts(sent);
    assert!(
        sent.starts_with("my ‹SECRET_1› ") && sent.contains("PASSWORD=‹SECRET_1›"),
        "{sent}"
    );
    assert!(!sent.contains(key), "{sent}");
    assert_eq!(texts(answer), format!("here: {key}"));
}

/// A command's output is redacted line by line before the tool spills,
/// shows or keeps it, and a call that never settles is committed through
/// the plugin's result filter; without the plugin the spilled key shows.
#[test]
fn a_spilled_commands_output_holds_no_key() {
    for redacting in [true, false] {
        let exe = std::env::current_exe().unwrap_or_default();
        let dir = exe.with_file_name(format!("redact-spill-{redacting}-{}", std::process::id()));
        std::fs::remove_dir_all(&dir).ok();
        let (mut app, wakes) = app_on(&MemoryStore::default());
        app.insert_resource(Scratch(Some(dir.clone())))
            .add_plugins((AgentPlugin, ShellTool));
        if redacting {
            app.add_plugins(RedactPlugin);
        }
        app.finish();
        app.update();
        let world = app.world_mut();
        let agent = world.spawn(Agent).id();
        // Its own text holds no key: the model's call would be redacted.
        let printed =
            "printf 'GITHUB_TO''KEN=ghp_''abcdefghijklmnopqrstuvwxyz0123456789\\n'; seq 1 3000";
        let args = json!({ "command": printed });
        let call = rig_harness_test_support::start_call(world, agent, ("shell", "s1"), args);
        let answered =
            |world: &World| call.is_some_and(|call| world.get::<Outcome>(call).is_some());
        run_until(&mut app, &wakes, answered);
        let spilled = std::fs::read_to_string(dir.join("spill").join("shell-1.txt"));
        std::fs::remove_dir_all(&dir).ok();
        let spilled = spilled.unwrap_or_default();
        assert!(spilled.contains("TOKEN="), "{spilled:.200}");
        assert_eq!(spilled.contains("ghp_"), !redacting, "{spilled:.200}");
        if redacting {
            let filter = app.world().resource::<ResultFilter>();
            let printed = ToolResult::success(ToolOutput::text("GITHUB_TOKEN=abcdefgh12345678"));
            let filtered = (filter.0)(&Outcome::from(printed));
            let filtered = filtered.map(|answer| answer.tool_result().output().render());
            assert!(filtered.is_some_and(|text| text.starts_with("GITHUB_TOKEN=‹SECRET_")));
        }
    }
}

/// A key block across a diff's lines takes its hunks, whatever the lines
/// end with, and an old last line with no ending runs into nothing.
#[test]
fn a_key_block_takes_the_hunks_whatever_its_lines_end_with() {
    for end in ["\n", "\r\n", "\r"] {
        let block = [
            "-----BEGIN RSA PRIVATE KEY-----",
            "MIIEowIBAAKCAQEAabc",
            "-----END RSA PRIVATE KEY-----",
        ];
        let old = HunkLine::Removed("old".to_owned());
        let added = block.map(|line| HunkLine::Added(format!("{line}{end}")));
        let mut hunks = vec![Hunk {
            old: Lines { start: 1, count: 1 },
            new: Lines { start: 1, count: 3 },
            lines: [old].into_iter().chain(added).collect(),
        }];
        assert!(redact_hunks(&mut hunks, &Secrets::default()));
        assert!(hunks.is_empty(), "{end:?}");
    }
}

/// A prompt the terminal view keeps is kept redacted, for Up and the file
/// other sessions read, its secrets and placeholders as `[REDACTED]`,
/// which names none.
#[cfg(feature = "tui")]
#[test]
fn a_prompt_kept_for_up_is_kept_redacted() {
    let mut world = World::new();
    let mut history = rig_tui::PromptHistory::default();
    history.entries = vec![
        "deploy with GITHUB_TOKEN=abcdefgh12345678".to_owned(),
        "then hunter2hunter2 and ‹SECRET_1› again".to_owned(),
    ];
    world.insert_resource(history);
    let secrets = Secrets::default();
    let _ = secrets.text("PASSWORD=hunter2hunter2");
    world.insert_resource(secrets);
    assert!(world.run_system_once(redact_history).is_ok());
    let entries = &world.resource::<rig_tui::PromptHistory>().entries;
    assert_eq!(
        entries,
        &[
            "deploy with GITHUB_TOKEN=[REDACTED]",
            "then [REDACTED] and [REDACTED] again"
        ]
    );
}

/// A tool call that names a placeholder runs with its secret, found again
/// in its output though it shows there bare, while the model's requests,
/// the conversation and the call every view reads keep the placeholder;
/// one naming a placeholder this run does not know is refused.
#[test]
fn a_call_naming_a_placeholder_runs_with_its_secret() {
    let exe = std::env::current_exe().unwrap_or_default();
    let dir = exe.with_file_name(format!("redact-reveal-{}", std::process::id()));
    std::fs::create_dir_all(&dir).ok();
    // A password: no shape finds it again, only its name, which the call's
    // output leaves out.
    let key = "hunter2hunter2";
    let command = "printf %s ‹SECRET_1› > out.txt && cat out.txt";
    let model = MockCompletionModel::from_stream_turns([
        calls("s1", "shell", json!({ "command": command })),
        calls("s2", "shell", json!({ "command": "echo ‹SECRET_7›" })),
        calls("s3", "shell", json!({ "command": "echo ‹SECRET_2›" })),
        reply("Done."),
    ]);
    let (mut app, wakes) = app_on(&MemoryStore::default());
    app.insert_resource(scripted_models([("test/model", &model)]))
        .add_plugins((AgentPlugin, ShellTool, RedactPlugin))
        .init_resource::<Ended>()
        .add_observer(log_ended);
    app.finish();
    app.update();
    let world = app.world_mut();
    let agent = (
        Agent,
        ModelChoice("test/model".to_owned()),
        WorkDir(dir.clone()),
    );
    let agent = world.spawn(agent).id();
    // A key in the system prompt, such as in an AGENTS.md, is sent as a
    // placeholder too.
    world.spawn(PromptSection::new(
        400,
        "deploy",
        "Deploy with TOKEN=prompted12345678.",
    ));
    let said = format!("Save DB_PASSWORD={key} to out.txt; API_KEY=\"two words here\"");
    world.trigger(Deliver::new(agent, said, DeliveryMode::Queue));
    run_until(&mut app, &wakes, |world| {
        !world.resource::<Ended>().0.is_empty()
    });
    let written = std::fs::read_to_string(dir.join("out.txt")).unwrap_or_default();
    std::fs::remove_dir_all(&dir).ok();
    assert_eq!(written, key);
    let world = app.world_mut();
    let held = world
        .get::<Conversation>(agent)
        .map(|conversation| format!("{:?}", conversation.messages()))
        .unwrap_or_default();
    assert!(held.contains("‹SECRET_1›") && !held.contains(key), "{held}");
    assert!(
        held.contains("‹SECRET_7› is not a secret this run knows"),
        "{held}"
    );
    // A secret a shell would read as code is not put in a command.
    let special = "‹SECRET_2› holds characters a shell reads as code";
    assert!(
        held.contains(special) && !held.contains("two words"),
        "{held}"
    );
    let mut requests = world.query::<&ToolRequest>();
    let asked: Vec<String> = requests
        .iter(world)
        .map(|request| format!("{:?}", request.call.function.arguments))
        .collect();
    assert!(asked.iter().all(|args| !args.contains(key)), "{asked:?}");
    // What the model was sent: placeholders only.
    let sent = format!("{:?}", model.requests());
    assert!(
        !sent.contains(key) && !sent.contains("prompted12345678"),
        "{sent}"
    );
    assert!(sent.contains("TOKEN=‹SECRET_"), "{sent}");
}

/// A run after a restart hands out numbers after the session's last, so an
/// old placeholder names no new secret.
#[test]
fn a_later_run_never_gives_an_old_number_to_a_new_secret() {
    let mut world = World::new();
    world.insert_resource(Secrets::default());
    world.insert_resource(Issued(3));
    assert!(world.run_system_once(keep_count).is_ok());
    let secrets = world.resource::<Secrets>().clone();
    let text = secrets.text("TOKEN=abcdefgh12345678").into_owned();
    assert_eq!(text, "TOKEN=‹SECRET_4›");
    assert_eq!(
        secrets.reveal("‹SECRET_2› ‹SECRET_4›"),
        "‹SECRET_2› abcdefgh12345678"
    );
    assert!(world.run_system_once(keep_count).is_ok());
    assert_eq!(world.resource::<Issued>().0, 4);
}

/// A secret the run learned is found again where it shows bare, without the
/// name that made it one, such as a password a call printed.
#[test]
fn a_known_secret_is_found_again_where_it_shows_bare() {
    let secrets = Secrets::default();
    assert_eq!(
        secrets.text("DB_PASSWORD=hunter2hunter2"),
        "DB_PASSWORD=‹SECRET_1›"
    );
    assert_eq!(secrets.text("hunter2hunter2\n"), "‹SECRET_1›\n");
    let printed = json!({ "output": "it is hunter2hunter2" });
    let redacted = secrets.redacted(&printed);
    assert_eq!(redacted, Some(json!({ "output": "it is ‹SECRET_1›" })));
}

/// A message delivered after the agent loop of the frame the app exits on,
/// which the exit keeps for the restart, and what a running call answers if
/// stopped, changed in place that frame, are redacted before the exit
/// keeps them; the count of numbers handed out is saved with them.
#[test]
fn what_the_exit_frame_keeps_is_redacted() {
    let store = MemoryStore::default();
    let (mut app, _) = app_on(&store);
    app.add_plugins((AgentPlugin, SessionLogPlugin, RedactPlugin));
    app.finish();
    app.update();
    let world = app.world_mut();
    let agent = world.spawn(Agent).id();
    let quiet = ToolResult::success(ToolOutput::text("so far nothing"));
    let call = world.spawn(IfStopped(Outcome::from(quiet))).id();
    // After `AgentLoop`, in the frame that exits.
    app.add_systems(
        PostUpdate,
        move |mut stopped: Query<&mut IfStopped>,
              mut commands: Commands,
              mut exits: MessageWriter<AppExit>,
              mut done: Local<bool>| {
            if std::mem::replace(&mut *done, true) {
                return;
            }
            let said = "deploy with DB_PASSWORD=hunter2hunter2";
            commands.trigger(Deliver::new(agent, said, DeliveryMode::Queue));
            if let Ok(mut stopped) = stopped.get_mut(call) {
                let printed =
                    ToolResult::success(ToolOutput::text("GITHUB_TOKEN=abcdefgh12345678"));
                stopped.0 = Outcome::from(printed);
            }
            exits.write(AppExit::Success);
        },
    );
    app.update();
    let stopped = app
        .world()
        .get::<IfStopped>(call)
        .map(|stopped| stopped.0.tool_result());
    let stopped = stopped
        .map(|result| result.output().render())
        .unwrap_or_default();
    assert!(stopped.starts_with("GITHUB_TOKEN=‹SECRET_"), "{stopped}");
    let logs = store.logs().unwrap_or_default();
    let saved: Vec<u8> = logs
        .iter()
        .flat_map(|log| store.read_log(log).unwrap_or_default())
        .collect();
    let saved = String::from_utf8_lossy(&saved);
    assert!(
        saved.contains("DB_PASSWORD=‹SECRET_") && !saved.contains("hunter2"),
        "{saved}"
    );
    let issued = saved
        .lines()
        .filter_map(|line| serde_json::from_str::<Value>(line).ok())
        .rfind(|line| line.get("key") == Some(&json!("rig_basics::redact::Issued")))
        .and_then(|line| line.get("value").cloned());
    assert_eq!(issued, Some(json!(2)));
}

/// A secret the run knows is found again though the stream cuts it at its
/// punctuation; a known original that is also a field's tag, or that a
/// JSON string holds escaped, hides no other secret.
#[test]
fn a_known_secret_is_found_however_it_arrives() {
    let secrets = Secrets::default();
    let _ = secrets.text("PASSWORD='hunter2!hunter2' and SECRET=tool_result and TOKEN=abc\\defgh1");
    let streamed = |pieces: [&str; 2]| -> String {
        let end = Relayed::Item(Item::Event(StreamEvent::End {
            part: Part::new(0),
            content: AssistantContent::text("ended"),
        }));
        pieces
            .map(|piece| text(0, piece))
            .into_iter()
            .chain([end])
            .flat_map(redact_stream(secrets.clone()))
            .filter_map(|item| match item {
                Relayed::Item(Item::Event(StreamEvent::Text { text, .. })) => Some(text),
                _ => None,
            })
            .collect()
    };
    assert_eq!(
        streamed(["say hunter2!", "hunter2 now."]),
        "say ‹SECRET_1› now."
    );
    // Whole in one piece, though its end could start it again.
    assert_eq!(
        streamed(["it is hunter2!hunter2", " ok"]),
        "it is ‹SECRET_1› ok"
    );
    // An original that is, or is part of, a field's tag: the tag is kept,
    // and the output is redacted, whatever finds its secrets.
    let output = |printed: &str, secrets: &Secrets| {
        let printed = Outcome::from(ToolResult::success(ToolOutput::text(printed)));
        let redacted = redacted_answer(&printed, secrets).map(Outcome);
        redacted.map(|answer| answer.tool_result().output().render())
    };
    let shaped = output(
        "GITHUB_TOKEN=ghp_abcdefghijklmnopqrstuvwxyz0123456789",
        &secrets,
    );
    assert_eq!(shaped.as_deref(), Some("GITHUB_TOKEN=‹SECRET_4›"));
    let bare = output("hunter2!hunter2 and tool_result", &secrets);
    assert_eq!(bare.as_deref(), Some("‹SECRET_1› and ‹SECRET_2›"));
    assert_eq!(
        output("tool_result", &secrets).as_deref(),
        Some("‹SECRET_2›")
    );
    let part = Secrets::default();
    let _ = part.text("SECRET=tool_res and PASSWORD=hunter2hunter2");
    let inside = output("value hunter2hunter2", &part);
    assert_eq!(inside.as_deref(), Some("value ‹SECRET_2›"));
    // An original a JSON string holds escaped, as a call's arguments do.
    let args = json!({ "args": serde_json::to_string(&json!({ "cmd": "echo abc\\defgh1" })).unwrap_or_default() });
    let shown = secrets
        .redacted(&args)
        .map(|args| args.to_string())
        .unwrap_or_default();
    assert!(
        shown.contains("‹SECRET_3›") && !shown.contains("defgh1"),
        "{shown}"
    );
}

/// A secret that reads like a placeholder's inside is never found in one:
/// redacting again changes nothing, and revealing gives it back.
#[test]
fn a_secret_like_a_placeholder_leaves_placeholders_alone() {
    let secrets = Secrets::default();
    let once = secrets.text("DB_PASSWORD=SECRET_1 and more").into_owned();
    assert_eq!(once, "DB_PASSWORD=‹SECRET_1› and more");
    assert_eq!(secrets.text(&once), once);
    assert_eq!(secrets.reveal("‹SECRET_1›"), "SECRET_1");
}

/// A secret that is also a name a model request gives, its model's or a
/// tool's, is never found as that name: the request keeps it, and so does
/// a call to that tool the reply streams.
#[test]
fn a_secret_like_a_name_leaves_the_name_alone() {
    let secrets = Secrets::default();
    let _ = secrets.text("DB_PASSWORD=get_weather and API_KEY=big-model-9");
    let parameters = json!({ "type": "object" });
    let Ok(name) = ToolName::new("get_weather") else {
        return;
    };
    let tool = ToolDefinition::new(name, "", parameters);
    let request = CompletionRequest::new("call it")
        .model("big-model-9")
        .tool(tool);
    let kind = EffectKind::Completion {
        request: Arc::new(request),
        stream: true,
    };
    let layer = AsSent(secrets.clone());
    let before = std::pin::pin!(layer.before(EffectId::from_raw(1), &kind));
    let mut context = std::task::Context::from_waker(std::task::Waker::noop());
    let ready = before.poll(&mut context);
    assert!(matches!(
        ready,
        std::task::Poll::Ready(BeforeServe::Proceed)
    ));
    assert_eq!(
        secrets.text("I will call get_weather"),
        "I will call get_weather"
    );
    let call = json!({ "name": "get_weather", "model": "big-model-9" });
    assert_eq!(secrets.redacted(&call), None);
}
