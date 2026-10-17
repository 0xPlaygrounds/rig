use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::mpsc::Receiver;

use rig_basics::coding_tools::ShellTool;
use rig_core::test_utils::MockCompletionModel;
use rig_harness::approval::ALLOW;
use rig_harness::ask::AskPlugin;
use rig_harness::harness_protocol::SessionId;
use rig_harness_test_support::{
    AnswerAsks, Answers, Ended, ToolOutputs, app_in, calls, connect, first_agent, log_ended,
    log_tool_outputs, reply, run_until, scratch_home,
};
use serde_json::json;

use super::*;

/// What `git args` printed in `dir`, trimmed; empty when it failed.
fn git(dir: &Path, args: &[&str]) -> String {
    let output = Command::new("git").current_dir(dir).args(args).output();
    output
        .ok()
        .filter(|output| output.status.success())
        .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_owned())
        .unwrap_or_default()
}

/// `name/remote.git`, a bare remote, and `name/work`, a clone on `main`
/// with a `pf` alias for `push --force`, three commits pushed and then the last one dropped locally (so a
/// force push would change the remote), a `feature` branch, a `sub/`
/// directory and the user's untracked `draft.md`.
fn repo(name: &str) -> (PathBuf, PathBuf) {
    let root = std::env::current_exe()
        .unwrap_or_default()
        .with_file_name(name);
    std::fs::remove_dir_all(&root).ok();
    assert!(std::fs::create_dir_all(root.join("work/sub")).is_ok());
    let root = std::fs::canonicalize(&root).unwrap_or(root);
    let (remote, work) = (root.join("remote.git"), root.join("work"));
    git(&root, &["init", "-q", "--bare", "-b", "main", "remote.git"]);
    git(&work, &["init", "-q", "-b", "main"]);
    for (key, value) in [
        ("user.email", "test@example.com"),
        ("user.name", "Test"),
        ("commit.gpgsign", "false"),
    ] {
        git(&work, &["config", key, value]);
    }
    git(&work, &["config", "alias.pf", "push --force"]);
    git(&work, &["remote", "add", "origin", "../remote.git"]);
    for (file, message) in [
        ("sub/keep.txt", "start"),
        ("one.txt", "one"),
        ("two.txt", "two"),
    ] {
        assert!(std::fs::write(work.join(file), message).is_ok());
        git(&work, &["add", file]);
        git(&work, &["commit", "-q", "-m", message]);
    }
    git(&work, &["push", "-q", "-u", "origin", "main"]);
    git(&work, &["reset", "-q", "--hard", "HEAD^"]);
    git(&work, &["branch", "feature"]);
    assert!(std::fs::write(work.join("draft.md"), "the user's notes\n").is_ok());
    (remote, work)
}

/// An app on the project `work` with the real `shell` tool, checkpoints,
/// questions and the guard, its agent answered by `model`; `answers`
/// adds someone who answers questions.
fn start(
    name: &str,
    work: &Path,
    model: &MockCompletionModel,
    config: GitGuardConfig,
    answers: Option<Answers>,
) -> (App, Receiver<()>) {
    let home = scratch_home(&format!("{name}-home"));
    let (mut app, wakes) = app_in(&home, &SessionId::generate());
    if let Some(answers) = answers {
        app.add_plugins(AnswerAsks).insert_resource(answers);
    }
    app.insert_resource(ProjectRoot(work.to_path_buf()))
        .insert_resource(config)
        .add_plugins((
            AgentPlugin,
            SessionLogPlugin,
            AskPlugin,
            ShellTool,
            GitGuardPlugin,
        ))
        .init_resource::<Ended>()
        .add_observer(log_ended)
        .init_resource::<ToolOutputs>()
        .add_observer(log_tool_outputs)
        .init_resource::<Bodies>()
        .add_observer(keep_bodies);
    app.finish();
    app.update();
    let agent = first_agent(app.world_mut()).unwrap_or(Entity::PLACEHOLDER);
    if let Some(connection) = connect(model) {
        app.world_mut().entity_mut(agent).insert(connection);
    }
    app.world_mut().trigger(Deliver::new(
        agent,
        "Tidy up the repo.",
        DeliveryMode::Steer,
    ));
    run_until(&mut app, &wakes, |world| {
        !world.resource::<Ended>().0.is_empty()
    });
    (app, wakes)
}

/// The body of every question asked.
#[derive(Resource, Default)]
struct Bodies(Vec<String>);

fn keep_bodies(added: On<Add<Ask>>, asks: Query<&Ask>, mut bodies: ResMut<Bodies>) {
    bodies
        .0
        .extend(asks.get(added.entity).map(|ask| ask.body.clone()));
}

fn output(app: &App, id: &str) -> String {
    let outputs = &app.world().resource::<ToolOutputs>().0;
    outputs.get(id).cloned().unwrap_or_default()
}

fn shell(id: &str, command: &str) -> Vec<rig_core::test_utils::MockStreamEvent> {
    calls(id, "shell", json!({ "command": command }))
}

#[test]
fn unsafe_git_is_refused_and_safe_git_runs() {
    let (remote, work) = repo("ct-git-guard-deny");
    let remote_main = git(&remote, &["rev-parse", "main"]);
    let model = MockCompletionModel::from_stream_turns([
        shell("force", "git push --force origin main"),
        shell("sub-force", "cd sub && git push -f"),
        shell("alias", "true; (git -C sub pf origin HEAD:main)"),
        shell("no-verify", "git commit --no-verify -m x"),
        shell("amend-pushed", "git commit --amend -m y"),
        shell(
            "own",
            "echo n > notes.txt; echo a > agent.txt && git add agent.txt && git commit -q -m mine",
        ),
        shell("amend-own", "git commit --amend -q -m mine-amended"),
        shell("reset", "git status --short | cat && git reset --hard"),
        shell("lease", "git push -q --force-with-lease origin feature"),
        reply("done"),
    ]);
    let (app, _wakes) = start(
        "ct-git-guard-deny",
        &work,
        &model,
        GitGuardConfig::default(),
        None,
    );

    for (id, rule) in [
        ("force", "force-push"),
        ("sub-force", "force-push"),
        ("alias", "force-push"),
        ("no-verify", "no-verify"),
        ("amend-pushed", "rewrite-pushed"),
        ("reset", "discard-changes"),
    ] {
        let said = output(&app, id);
        assert!(said.contains(&format!("rule {rule}")), "{id}: {said}");
    }
    assert!(output(&app, "force").contains("`main`"));
    assert!(output(&app, "amend-pushed").contains("Commit a fixup instead"));
    assert_eq!(
        git(&remote, &["rev-parse", "main"]),
        remote_main,
        "remote unchanged"
    );

    // The agent's own commit, not pushed, may be amended.
    assert!(!output(&app, "amend-own").contains(NAME));
    assert_eq!(git(&work, &["log", "-1", "--format=%s"]), "mine-amended");

    // The user's file is listed and kept; the agent's own is not theirs.
    let reset = output(&app, "reset");
    assert!(
        reset.contains("draft.md") && !reset.contains("notes.txt"),
        "{reset}"
    );
    let draft = std::fs::read_to_string(work.join("draft.md")).unwrap_or_default();
    assert_eq!(draft, "the user's notes\n");

    let lease = output(&app, "lease");
    assert!(!lease.contains(NAME), "{lease}");
    assert!(!git(&remote, &["rev-parse", "--verify", "refs/heads/feature"]).is_empty());
}

#[test]
fn an_ask_rule_runs_once_the_user_allows() {
    let (_remote, work) = repo("ct-git-guard-ask");
    let model = MockCompletionModel::from_stream_turns([
        shell("amend", "git commit --amend -q -m z"),
        reply("done"),
    ]);
    let mut config = GitGuardConfig::default();
    config.rules.rewrite_pushed = Level::Ask;
    let answers = Answers::new([Answer::choice(ALLOW)]);
    let (app, _wakes) = start("ct-git-guard-ask", &work, &model, config, Some(answers));

    let asked = &app.world().resource::<Answers>().asked;
    assert_eq!(asked, &["Allow `shell`?"]);
    let body = app.world().resource::<Bodies>().0.join("\n");
    assert!(
        body.contains("rule rewrite-pushed") && body.contains("git commit --amend"),
        "{body}"
    );
    assert_eq!(git(&work, &["log", "-1", "--format=%s"]), "z");
}

#[test]
fn stages_are_ordered_with_the_defaults() {
    rig_defaults_test_support::assert_stages_ordered(GitGuardPlugin);
}
