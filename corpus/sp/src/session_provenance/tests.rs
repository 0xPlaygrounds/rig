use std::fs;
use std::path::Path;
use std::process::Command;
use std::sync::mpsc::Receiver;

use rig_basics::models::ModelsPlugin;
use rig_core::test_utils::MockCompletionModel;
use rig_harness::harness_protocol::SessionId;
use rig_harness::load;
use rig_harness_test_support::{
    Ended, app_in, first_agent, log_ended, reply, run_until, scratch_home, scripted_models,
    tui_view,
};
use rig_tui::TuiTestScreen;

use super::*;

/// The test plugin whose version changes between runs.
#[derive(Default)]
struct AlphaPlugin;

impl Plugin for AlphaPlugin {
    fn build(&self, _app: &mut App) {}
}

/// Runs `git args` in `dir`; its output, trimmed.
fn git(dir: &Path, args: &[&str]) -> String {
    let output = Command::new("git")
        .args([
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@t",
            "-c",
            "commit.gpgsign=false",
        ])
        .args(args)
        .current_dir(dir)
        .output();
    assert!(
        output.as_ref().is_ok_and(|o| o.status.success()),
        "git {args:?}: {output:?}"
    );
    output
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_owned())
        .unwrap_or_default()
}

/// Commits a change to `file` in `repo`; the new HEAD.
fn commit(repo: &Path, file: &str) -> String {
    assert!(fs::write(repo.join(file), file).is_ok());
    git(repo, &["add", "."]);
    git(repo, &["commit", "-q", "-m", "change"]);
    git(repo, &["rev-parse", "HEAD"])
}

/// An app on session `id` of `home` in `repo`, with `alpha` at `alpha`
/// (and the provenance plugin when `provenance`), after its first frame;
/// a new session's agent starts on `model`.
fn start(
    (home, repo): (&Home, &Path),
    id: &str,
    models: Models,
    alpha: &str,
    provenance: bool,
) -> Option<(App, Receiver<()>, Entity)> {
    let id: SessionId = id.parse().ok()?;
    let new = !home.session(&id).is_saved();
    let (mut app, wakes) = app_in(home, &id);
    app.insert_resource(models)
        .insert_resource(ProjectRoot(repo.to_owned()))
        .insert_resource(TuiTestScreen::new(120, 60))
        .init_resource::<Ended>()
        .add_observer(log_ended)
        .add_plugins((AgentPlugin, SessionLogPlugin, ModelsPlugin));
    load::<AlphaPlugin>(&mut app, "alpha", &format!("version {alpha}"));
    if provenance {
        load::<SessionProvenancePlugin>(&mut app, "sp-session-provenance", "path corpus");
    }
    tui_view(&mut app);
    if new {
        let model = ModelChoice("test/scripted-a".to_owned());
        let agent = (Name::new("main"), Agent, AgentId("main".to_owned()), model);
        app.world_mut().spawn(agent);
    }
    app.finish();
    app.cleanup();
    app.update();
    let agent = first_agent(app.world_mut())?;
    app.insert_resource(ShownAgent(Some(agent)));
    app.update();
    Some((app, wakes, agent))
}

/// Sends `line` as the user typed it: a command runs, a message's turn
/// runs to its end.
fn send(app: &mut App, wakes: &Receiver<()>, agent: Entity, line: &str) {
    let ended = app.world().resource::<Ended>().0.len();
    let typed = Deliver::new(agent, line, DeliveryMode::Queue).with_author(Author::user());
    app.world_mut().trigger(typed);
    if line.starts_with('/') {
        app.update();
        app.update();
    } else {
        run_until(app, wakes, |world| {
            world.resource::<Ended>().0.len() > ended
        });
    }
}

/// What the test screen shows.
fn screen(app: &App) -> String {
    app.world().resource::<TuiTestScreen>().rows().join("\n")
}

/// The agent's record.
fn record(app: &App, agent: Entity) -> Provenance {
    app.world()
        .get::<Provenance>(agent)
        .cloned()
        .unwrap_or_default()
}

/// The kinds of the record's entries.
fn kinds(record: &Provenance) -> Vec<EntryKind> {
    record.entries.iter().map(|entry| entry.kind).collect()
}

#[test]
fn the_setup_and_its_changes_are_recorded_resumed_and_never_sent() {
    let home = scratch_home("sp-session-provenance-home");
    let repo = home.root().join("repo");
    assert!(fs::create_dir_all(&repo).is_ok());
    git(&repo, &["init", "-q", "-b", "main"]);
    let first = commit(&repo, "a.txt");
    let a = MockCompletionModel::from_stream_turns([reply("Old session.")]);
    let b = MockCompletionModel::from_stream_turns([reply("Done."), reply("Again.")]);
    let models = || scripted_models([("test/scripted-a", &a), ("test/scripted-b", &b)]);
    let at = (&home, repo.as_path());

    // A new session: `/session-info` shows the start record.
    let started = start(at, "1000-1", models(), "0.1", true);
    assert!(started.is_some(), "a session with an agent");
    let Some((mut app, wakes, agent)) = started else {
        return;
    };
    send(&mut app, &wakes, agent, "/session-info");
    let shown = screen(&app);
    for expected in [
        VERSION,
        "alpha 0.1",
        "test/scripted-a",
        short(&first),
        "clean",
    ] {
        assert!(shown.contains(expected), "{expected} in\n{shown}");
    }
    let started_with = record(&app, agent).start;
    assert!(started_with.is_some());

    // `/model` appends an entry; the start record stays.
    send(&mut app, &wakes, agent, "/model test/scripted-b");
    let after_model = record(&app, agent);
    assert_eq!(after_model.start, started_with);
    assert_eq!(kinds(&after_model), [EntryKind::Model]);

    // A commit outside the agent is noticed as the next turn starts.
    let second = commit(&repo, "b.txt");
    send(&mut app, &wakes, agent, "Do it.");
    let after_turn = record(&app, agent);
    assert_eq!(kinds(&after_turn), [EntryKind::Model, EntryKind::Git]);
    let git_line = after_turn
        .entries
        .get(1)
        .map(|entry| entry.lines.join("\n"));
    assert!(git_line.is_some_and(|line| line.contains(short(&second))));
    drop(app);

    // Restarted with alpha 0.2 and resumed: only that is new.
    let resumed = start(at, "1000-1", models(), "0.2", true);
    assert!(resumed.is_some(), "the session was resumed");
    let Some((mut app, wakes, agent)) = resumed else {
        return;
    };
    let after_resume = record(&app, agent);
    assert_eq!(after_resume.start, started_with);
    let last = after_resume.entries.last();
    assert_eq!(last.map(|entry| entry.kind), Some(EntryKind::Resumed));
    assert_eq!(
        last.map(|entry| entry.lines.clone()),
        Some(vec!["plugin alpha 0.1 → 0.2".to_owned()])
    );

    // Another model after a second turn appends one more entry.
    send(&mut app, &wakes, agent, "Again.");
    send(&mut app, &wakes, agent, "/model test/scripted-a");
    assert_eq!(record(&app, agent).entries.len(), 4);
    drop(app);

    // A session saved before the plugin ran says so, and is recorded from
    // its resume on.
    let old = start(at, "1000-9", models(), "0.2", false);
    assert!(old.is_some(), "an old session");
    let Some((mut app, wakes, agent)) = old else {
        return;
    };
    send(&mut app, &wakes, agent, "Old.");
    drop(app);
    let old = start(at, "1000-9", models(), "0.2", true);
    assert!(old.is_some(), "the old session was resumed");
    let Some((mut app, wakes, agent)) = old else {
        return;
    };
    send(&mut app, &wakes, agent, "/session-info");
    let shown = screen(&app);
    assert!(shown.contains("no provenance recorded"), "{shown}");
    assert_eq!(kinds(&record(&app, agent)), [EntryKind::Resumed]);
    drop(app);

    // No model read any of it.
    let requests = [a.requests(), b.requests()].concat();
    assert_eq!(requests.len(), 3);
    for request in &requests {
        let sent = serde_json::to_string(request).unwrap_or_default();
        for secret in [
            first.as_str(),
            short(&first),
            &second,
            short(&second),
            "alpha",
        ] {
            assert!(!sent.contains(secret), "{secret} in {sent}");
        }
    }
    fs::remove_dir_all(home.root()).ok();
}
