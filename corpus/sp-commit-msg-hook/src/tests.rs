use std::fs;
use std::os::unix::fs::PermissionsExt as _;
use std::path::{Path, PathBuf};
use std::process::Command;

use rig_core::test_utils::{MockCompletionModel, MockError, MockStreamEvent};
use rig_harness::harness_protocol::{Home, Invocation, SessionId};
use rig_harness::host::exit::ExitPlugin;
use rig_harness::settings::Settings;
use rig_harness_test_support::{app_in, reply, run_until, scratch_home, scripted_models};

use super::*;

/// What git writes in the message file before the hook runs.
const TEMPLATE: &str = "\n# Please enter the commit message for your changes. Lines starting\n\
                        # with '#' will be ignored, and an empty message aborts the commit.\n";

/// The settings of a run on the scripted model.
const SETTINGS: &str = "[commit_msg]\nmodel = \"test/model\"\n";

/// Variables that keep git away from this machine's own configuration.
const ISOLATED: [(&str, &str); 6] = [
    ("GIT_CONFIG_GLOBAL", "/dev/null"),
    ("GIT_CONFIG_NOSYSTEM", "1"),
    ("GIT_AUTHOR_NAME", "Test"),
    ("GIT_AUTHOR_EMAIL", "test@example.com"),
    ("GIT_COMMITTER_NAME", "Test"),
    ("GIT_COMMITTER_EMAIL", "test@example.com"),
];

/// Runs git in `repo` with `args`, isolated; whether it succeeded.
fn git_in(repo: &Path, args: &[&str], path: Option<&str>) -> bool {
    let mut command = Command::new("git");
    command.args(args).current_dir(repo).envs(ISOLATED);
    if let Some(path) = path {
        command.env("PATH", path);
    }
    command.status().is_ok_and(|status| status.success())
}

/// A repository in `home` with two commits and a change to `lib.rs`
/// staged.
fn repository(home: &Home) -> PathBuf {
    let repo = home.root().join("repo");
    fs::create_dir_all(&repo).ok();
    assert!(git_in(&repo, &["init", "-q"], None));
    for (text, subject) in [
        ("pub fn parse() {}\n", "feat: add the parser"),
        ("pub fn parse(input: &str) {}\n", "refactor: take the input"),
    ] {
        fs::write(repo.join("lib.rs"), text).ok();
        assert!(git_in(&repo, &["add", "lib.rs"], None));
        assert!(git_in(&repo, &["commit", "-q", "-m", subject], None));
    }
    let fixed =
        "pub fn parse(input: &str) {\n    if input.is_empty() {\n        return;\n    }\n}\n";
    fs::write(repo.join("lib.rs"), fixed).ok();
    assert!(git_in(&repo, &["add", "lib.rs"], None));
    repo
}

/// What `rig <command> <args>` did in `repo`, its model scripted and its
/// settings `settings`, once its app exited.
fn rig(
    home: &Home,
    repo: &Path,
    command: &str,
    args: &[&str],
    settings: &str,
    model: &MockCompletionModel,
) -> CommandRun {
    let invocation = Invocation {
        command: Some(command.to_owned()),
        arguments: args.iter().map(|arg| (*arg).to_owned()).collect(),
        ..Invocation::default()
    };
    let processes = Processes::default();
    let isolated = ISOLATED
        .iter()
        .fold(processes.launcher(), |launcher, (name, value)| {
            launcher.with_env(name, value)
        });
    processes.set(isolated);
    let (mut app, wakes) = app_in(home, &SessionId::generate());
    app.insert_resource(Invoked {
        args: invocation,
        terminal: false,
    })
    .insert_resource(Settings::from_toml(settings))
    .insert_resource(scripted_models([("test/model", model)]))
    .insert_resource(ProjectRoot(repo.to_owned()))
    .insert_resource(processes)
    .add_plugins((
        AgentPlugin,
        SessionLogPlugin,
        ExitPlugin,
        CommitMsgHookPlugin,
    ));
    app.finish();
    run_until(&mut app, &wakes, |world| {
        world.contains_resource::<CommandRun>()
    });
    run_until(&mut app, &wakes, |world| {
        world
            .get_resource::<Messages<AppExit>>()
            .is_some_and(|exits| !exits.is_empty())
    });
    app.world_mut()
        .remove_resource::<CommandRun>()
        .unwrap_or(CommandRun {
            status: 255,
            ..CommandRun::default()
        })
}

/// What the model was sent, as JSON text.
fn sent(model: &MockCompletionModel) -> String {
    let requests = model.requests();
    let history = requests.iter().map(|request| &request.chat_history);
    let history: Vec<_> = history.collect();
    serde_json::to_string(&history).unwrap_or_default()
}

/// The entries of the sessions directory.
fn sessions(home: &Home) -> usize {
    fs::read_dir(home.sessions()).map_or(0, |dir| dir.count())
}

#[test]
fn a_draft_goes_above_git_s_comments_and_never_stops_a_commit() {
    let home = scratch_home("sp-commit-msg-hook-draft");
    let repo = repository(&home);
    let file = repo.join(".git/COMMIT_EDITMSG");
    fs::write(&file, TEMPLATE).ok();
    let read = |file: &Path| fs::read_to_string(file).unwrap_or_default();

    // Drafted: the message above the comments, from the staged diff and
    // both earlier subjects, in one call without tools.
    let model = MockCompletionModel::from_stream_turns([reply("fix: handle empty input")]);
    let run = rig(
        &home,
        &repo,
        COMMIT_MSG_COMMAND,
        &[".git/COMMIT_EDITMSG"],
        SETTINGS,
        &model,
    );
    assert_eq!(run, CommandRun::default());
    let drafted = read(&file);
    let expected = format!("fix: handle empty input\n\n{}", TEMPLATE.trim_start());
    assert_eq!(drafted, expected);
    assert_eq!(model.request_count(), 1);
    assert!(
        model
            .requests()
            .iter()
            .all(|request| request.tools.is_empty())
    );
    let request = sent(&model);
    for seen in [
        "if input.is_empty()",
        "feat: add the parser",
        "refactor: take the input",
    ] {
        assert!(request.contains(seen), "{seen} not in {request}");
    }
    assert!(!request.contains("was cut"), "{request}");

    // A message git already has: nothing touched, no call.
    fs::write(&file, TEMPLATE).ok();
    let model = MockCompletionModel::from_stream_turns([reply("never asked")]);
    let args = [".git/COMMIT_EDITMSG", "message"];
    let run = rig(&home, &repo, COMMIT_MSG_COMMAND, &args, SETTINGS, &model);
    assert_eq!(run, CommandRun::default());
    assert_eq!((read(&file).as_str(), model.request_count()), (TEMPLATE, 0));

    // The model failing: exit 0, the file as it was, one line on stderr.
    let failing = vec![MockStreamEvent::Error(MockError::provider("overloaded"))];
    let model = MockCompletionModel::from_stream_turns([failing]);
    let run = rig(
        &home,
        &repo,
        COMMIT_MSG_COMMAND,
        &[".git/COMMIT_EDITMSG"],
        SETTINGS,
        &model,
    );
    assert_eq!((run.status, run.stdout.as_str()), (0, ""));
    assert_eq!(run.stderr.lines().count(), 1, "{}", run.stderr);
    assert_eq!(read(&file), TEMPLATE);

    // A diff over the limit is cut with every file's summary, and said so;
    // `--dry-run` prints the draft.
    let settings = format!("{SETTINGS}max_diff_bytes = 40\n");
    let body = "fix: handle empty input\n\nReturn early when there is nothing to parse.";
    let model = MockCompletionModel::from_stream_turns([reply(&format!("```\n{body}\n```"))]);
    let run = rig(
        &home,
        &repo,
        COMMIT_MSG_COMMAND,
        &["--dry-run"],
        &settings,
        &model,
    );
    assert_eq!((run.status, run.stdout), (0, format!("{body}\n")));
    let request = sent(&model);
    assert!(
        request.contains("was cut") && request.contains("lib.rs |"),
        "{request}"
    );
    assert!(!request.contains("return;"), "{request}");

    // No run left a session behind.
    assert_eq!(sessions(&home), 0);
}

#[test]
fn install_keeps_a_foreign_hook_only_when_chained_and_uninstall_never_removes_it() {
    let home = scratch_home("sp-commit-msg-hook-install");
    let repo = repository(&home);
    let model = MockCompletionModel::from_stream_turns(Vec::<Vec<MockStreamEvent>>::new());
    let hook = repo.join(".git/hooks/prepare-commit-msg");
    let hook_cmd = |args: &[&str]| rig(&home, &repo, HOOK_COMMAND, args, "", &model);

    assert_eq!(hook_cmd(&["install"]).status, 0);
    let mode = fs::metadata(&hook)
        .map(|meta| meta.permissions().mode())
        .ok();
    assert!(mode.is_some_and(|mode| mode & 0o111 == 0o111), "{mode:?}");
    assert_eq!(hook_cmd(&["uninstall"]).status, 0);
    assert!(!hook.exists());

    // A hook of the user's own: refused, kept, and chained with --chain.
    let foreign = "#!/bin/sh\necho \"$@\" > old-hook-ran\n";
    fs::write(&hook, foreign).ok();
    fs::set_permissions(&hook, fs::Permissions::from_mode(0o755)).ok();
    let refused = hook_cmd(&["install"]);
    assert_eq!(refused.status, 1, "{refused:?}");
    assert_eq!(fs::read_to_string(&hook).unwrap_or_default(), foreign);
    assert_eq!(hook_cmd(&["uninstall"]).status, 1);
    assert_eq!(fs::read_to_string(&hook).unwrap_or_default(), foreign);
    assert_eq!(hook_cmd(&["install", "--chain"]).status, 0);

    // A commit runs both: the old hook first, then `rig commit-msg` (a
    // stand-in `rig` on the PATH records how it was called).
    let bin = home.root().join("bin");
    fs::create_dir_all(&bin).ok();
    let stand_in = bin.join("rig");
    fs::write(&stand_in, "#!/bin/sh\necho \"$@\" > rig-ran\n").ok();
    fs::set_permissions(&stand_in, fs::Permissions::from_mode(0o755)).ok();
    let path = format!(
        "{}:{}",
        bin.display(),
        std::env::var("PATH").unwrap_or_default()
    );
    assert!(git_in(
        &repo,
        &["commit", "-q", "-m", "fix: handle empty input"],
        Some(&path)
    ));
    let ran = |name: &str| fs::read_to_string(repo.join(name)).unwrap_or_default();
    assert_eq!(ran("old-hook-ran"), ".git/COMMIT_EDITMSG message\n");
    assert_eq!(ran("rig-ran"), "commit-msg .git/COMMIT_EDITMSG message\n");

    // Uninstalling puts the user's hook back.
    assert_eq!(hook_cmd(&["uninstall"]).status, 0);
    assert_eq!(fs::read_to_string(&hook).unwrap_or_default(), foreign);
}
