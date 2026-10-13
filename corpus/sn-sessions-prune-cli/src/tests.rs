use std::fs;
use std::io::Write as _;
use std::path::PathBuf;

use rig_core::message::{Message, UserContent};
use rig_core::test_utils::{MockCompletionModel, MockStreamEvent};
use rig_harness::harness_protocol::{Invocation, SessionId};
use rig_harness::host::exit::ExitPlugin;
use rig_harness_test_support::{
    Ended, app_in, calls, first_agent, log_ended, reply, run_until, scratch_home, scripted_models,
};
use rig_sessions::{NameSession, SessionsPlugin, write_meta};

use super::*;

const DAY: u64 = 86_400_000;

/// A tool whose large output a plugin keeps whole in the session's spill.
fn dump(
    called: On<ToolCalled<serde_json::Value>>,
    scratch: Res<Scratch>,
    mut calls: Query<&mut Decision, ToolCalls>,
) {
    let Some((path, mut file)) = scratch.spill().and_then(|spill| spill.create("dump").ok()) else {
        return;
    };
    file.write_all(&vec![b'x'; 2_000_000]).ok();
    if let Ok(mut call) = calls.get_mut(called.call) {
        let text = format!("2 MB, whole at {}", path.display());
        call.decide(Decider::plugin("dump"), ToolResult::success(text.into()));
    }
}

/// A new session of `home` run through `turns`, one prompt each, answered
/// by `script`; its app, still open, and its agent.
fn run(home: &Home, turns: &[&str], script: Vec<Vec<MockStreamEvent>>) -> (SessionId, App, Entity) {
    let id = home
        .new_session()
        .map(|(id, _)| id)
        .unwrap_or_else(|_| SessionId::generate());
    let model = MockCompletionModel::from_stream_turns(script);
    let (mut app, wakes) = app_in(home, &id);
    app.insert_resource(scripted_models([("test/model", &model)]))
        .add_plugins((AgentPlugin, JournalPlugin, SessionsPlugin))
        .add_open_tool("dump", "Dump a lot.", ToolOptions::default(), dump)
        .init_resource::<Ended>()
        .add_observer(log_ended);
    app.finish();
    app.update();
    let agent = first_agent(app.world_mut()).unwrap_or(Entity::PLACEHOLDER);
    let model = "test/model".to_owned();
    app.world_mut().trigger(SetModel {
        entity: agent,
        model,
    });
    for (done, prompt) in turns.iter().enumerate() {
        app.world_mut()
            .trigger(Deliver::new(agent, *prompt, DeliveryMode::Steer));
        run_until(&mut app, &wakes, |world| {
            world.resource::<Ended>().0.len() > done
        });
    }
    app.update();
    (id, app, agent)
}

/// A session of one exchange.
fn simple(home: &Home, prompt: &str) -> SessionId {
    run(home, &[prompt], vec![reply("ok")]).0
}

/// Makes `id` last changed `days` ago, as if it ran then.
fn age(home: &Home, id: &SessionId, days: u64) {
    let Ok(store) = SessionStores::default().open(&home.session(id)) else {
        return;
    };
    let mut meta = Meta::of(&*store).unwrap_or_default();
    meta.updated = millis(SystemTime::now()).saturating_sub(days * DAY);
    write_meta(&*store, &meta).ok();
}

/// Runs `rig sessions <words>` without a terminal: how it exits and what it
/// reported.
fn prune(home: &Home, words: &[&str]) -> (Option<ExitReason>, PruneReport) {
    let id = home
        .new_session()
        .map(|(id, _)| id)
        .unwrap_or_else(|_| SessionId::generate());
    let (mut app, _wakes) = app_in(home, &id);
    app.insert_resource(Invoked {
        args: Invocation {
            command: Some("sessions".to_owned()),
            arguments: words.iter().map(|word| (*word).to_owned()).collect(),
            ..Invocation::default()
        },
        terminal: false,
    })
    .add_plugins((ExitPlugin, AgentPlugin, JournalPlugin, SessionsPrunePlugin));
    app.finish();
    app.update();
    let world = app.world();
    let report = world
        .get_resource::<PruneReport>()
        .cloned()
        .unwrap_or_default();
    (world.get_resource::<ExitReason>().cloned(), report)
}

/// The first agent's messages in session `id`, as `--resume` reads them.
fn history(home: &Home, id: &SessionId) -> Vec<Message> {
    let saved = SessionStores::default()
        .open(&home.session(id))
        .and_then(|store| Session::read(&*store))
        .unwrap_or_default();
    let main = saved
        .agents
        .into_iter()
        .find(|agent| agent.parent.is_none());
    main.map(|agent| agent.conversation.messages().to_vec())
        .unwrap_or_default()
}

/// Whether `messages` has a user message saying `text`.
fn says(messages: &[Message], text: &str) -> bool {
    messages.iter().any(|message| {
        matches!(message, Message::User { content }
        if content.iter().any(|item| matches!(item, UserContent::Text(said) if said.text == text)))
    })
}

fn ids(sessions: &[PlannedSession]) -> Vec<String> {
    let mut ids: Vec<String> = sessions.iter().map(|session| session.id.clone()).collect();
    ids.sort();
    ids
}

#[test]
fn prunes_old_sessions_but_what_is_named_open_or_needed() {
    let home = scratch_home("sn-sessions-prune-cli");
    // S1: a large spilled tool output.
    let (s1, app, _) = run(
        &home,
        &["Dump it."],
        vec![calls("c1", "dump", serde_json::json!({})), reply("dumped")],
    );
    let spill = app
        .world()
        .resource::<Scratch>()
        .spill()
        .map(|spill| spill.0);
    drop(app);
    let spilled: Vec<PathBuf> = spill
        .and_then(|dir| fs::read_dir(dir).ok())
        .into_iter()
        .flatten()
        .filter_map(|entry| Some(entry.ok()?.path()))
        .collect();
    assert_eq!(spilled.len(), 1, "S1 spilled its output");
    let s2 = simple(&home, "Second.");
    // S3: named "release".
    let (s3, mut app, _) = run(&home, &["Cut the release."], vec![reply("ok")]);
    app.world_mut().trigger(NameSession {
        session: s3.clone(),
        name: "release".to_owned(),
    });
    app.update();
    drop(app);
    let s4 = simple(&home, "Held open.");
    // S5, and S6 forked from it before its second message.
    let (s5, mut app, agent) = run(
        &home,
        &["Before the fork.", "After the fork."],
        vec![reply("one"), reply("two")],
    );
    let world = app.world_mut();
    let before = world.get::<AgentId>(agent).cloned().zip(
        world
            .get::<Conversation>(agent)
            .and_then(|conversation| conversation.id(2)),
    );
    world.trigger(ForkSession {
        before,
        asker: None,
    });
    app.update();
    drop(app);
    let s7 = simple(&home, "Recent.");
    for old in [&s1, &s2, &s3, &s4, &s5] {
        age(&home, old, 60);
    }
    age(&home, &s7, 1);
    let s6 = fs::read_dir(home.sessions())
        .into_iter()
        .flatten()
        .filter_map(|entry| entry.ok()?.file_name().to_str()?.parse::<SessionId>().ok())
        .find(|id| {
            let store = SessionStores::default().open(&home.session(id));
            store
                .ok()
                .and_then(|store| ForkedFrom::of(&*store))
                .is_some_and(|from| from.session == s5)
        });
    assert!(s6.is_some(), "S5 was forked");
    let Some(s6) = s6 else {
        return;
    };
    // S4 is open in a running rig.
    let lock = fs::File::create(home.session(&s4).launcher_lock());
    assert!(lock.as_ref().is_ok_and(|lock| lock.try_lock().is_ok()));

    // The plan.
    let (exit, plan) = prune(
        &home,
        &["prune", "--older-than", "30d", "--dry-run", "--json"],
    );
    assert_eq!(exit, Some(ExitReason::Quit));
    assert!(!plan.done);
    let mut old = vec![s1.to_string(), s2.to_string()];
    old.sort();
    assert_eq!(ids(&plan.deleted), old);
    let kept: HashMap<String, KeptBecause> = plan
        .kept
        .iter()
        .map(|kept| (kept.session.id.clone(), kept.because.clone()))
        .collect();
    assert_eq!(kept.get(s3.as_str()), Some(&KeptBecause::Named));
    assert_eq!(kept.get(s4.as_str()), Some(&KeptBecause::InUse));
    assert_eq!(
        kept.get(s5.as_str()),
        Some(&KeptBecause::NeededBy(s6.to_string()))
    );
    assert_eq!(kept.len(), 3, "{kept:?}");
    assert!(plan.bytes_freed > 2_000_000);
    assert!(home.session(&s1).is_saved() && spilled.iter().all(|path| path.exists()));

    // No terminal, no --yes: refused.
    let (exit, _) = prune(&home, &["prune", "--older-than", "30d"]);
    assert_eq!(exit, Some(ExitReason::Failed(2)));
    assert!(home.session(&s1).path().exists());

    // Deleted.
    let (exit, done) = prune(&home, &["prune", "--older-than", "30d", "--yes"]);
    assert_eq!(exit, Some(ExitReason::Quit));
    assert!(done.done && done.failed.is_empty(), "{:?}", done.failed);
    assert!(done.bytes_freed > 2_000_000);
    assert!(!home.session(&s1).path().exists() && !home.session(&s2).path().exists());
    assert!(spilled.iter().all(|path| !path.exists()));
    for kept in [&s3, &s4, &s5, &s6, &s7] {
        assert!(home.session(kept).is_saved(), "{kept} kept");
    }
    // S6 resumes with what came before the fork.
    let forked = history(&home, &s6);
    assert!(
        says(&forked, "Before the fork.") && !says(&forked, "After the fork."),
        "{forked:?}"
    );
    drop(lock);

    // Keep the newest of a project.
    let project = home.root().join("project-p");
    fs::create_dir_all(&project).ok();
    let mut in_project = Vec::new();
    for (prompt, days) in [("P one.", 60), ("P two.", 50), ("P three.", 40)] {
        let id = simple(&home, prompt);
        fs::write(
            home.session(&id).directory(),
            project.to_string_lossy().as_bytes(),
        )
        .ok();
        age(&home, &id, days);
        in_project.push(id);
    }
    let path = project.to_string_lossy().into_owned();
    let words = [
        "prune",
        "--keep-last",
        "1",
        "--project",
        &path,
        "--yes",
        "--include-named",
    ];
    assert_eq!(prune(&home, &words).0, Some(ExitReason::Quit));
    let left: Vec<bool> = in_project
        .iter()
        .map(|id| home.session(id).is_saved())
        .collect();
    assert_eq!(left, [false, false, true]);
    assert!(home.session(&s7).is_saved(), "other projects untouched");

    // No rule: refused.
    assert_eq!(
        prune(&home, &["prune", "--dry-run"]).0,
        Some(ExitReason::Failed(2))
    );
}
