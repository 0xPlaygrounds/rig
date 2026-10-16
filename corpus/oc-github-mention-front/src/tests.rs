use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpListener;
use std::process::Command;
use std::sync::Mutex;

use rig_coding_tools::WriteTool;
use rig_core::test_utils::MockCompletionModel;
use rig_harness::harness_protocol::{Invocation, SessionId};
use rig_harness::services::ServicesPlugin;
use rig_harness_test_support::{
    app_in, calls, first_agent, reply, run_until, scratch_home, scripted_models,
};

use super::*;

/// A request the fake API got: method, path and body.
type Request = (String, String, String);

/// A fake GitHub REST API for the repository `o/r` on a local port, which
/// records every request: `writer` may write, anyone else only read; pull
/// request 3 is the branch `feature` of `o/r`.
fn fake_api() -> Option<(String, Arc<Mutex<Vec<Request>>>)> {
    let listener = TcpListener::bind("127.0.0.1:0").ok()?;
    let url = format!("http://{}", listener.local_addr().ok()?);
    let log = Arc::new(Mutex::new(Vec::new()));
    let seen = log.clone();
    std::thread::spawn(move || {
        for mut stream in listener.incoming().flatten() {
            let mut reader = BufReader::new(&mut stream);
            let mut head = Vec::new();
            let mut line = String::new();
            while reader.read_line(&mut line).is_ok_and(|n| n > 2) {
                head.push(line.trim().to_owned());
                line.clear();
            }
            let header = |name: &str| {
                head.iter()
                    .find_map(|h| {
                        h.to_lowercase()
                            .strip_prefix(&format!("{name}: "))
                            .map(str::to_owned)
                    })
                    .unwrap_or_default()
            };
            let mut body = vec![0; header("content-length").parse().unwrap_or(0)];
            reader.read_exact(&mut body).ok();
            let first = head.first().cloned().unwrap_or_default();
            let mut words = first.split(' ');
            let (method, path) = (
                words.next().unwrap_or("").to_owned(),
                words.next().unwrap_or("").to_owned(),
            );
            let answer = route(&method, &path, &header("accept"));
            if let Ok(mut seen) = seen.lock() {
                seen.push((method, path, String::from_utf8_lossy(&body).into_owned()));
            }
            let text = format!(
                "HTTP/1.1 {}\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                answer.0,
                answer.1.len(),
                answer.1
            );
            stream.write_all(text.as_bytes()).ok();
        }
    });
    Some((url, log))
}

/// The fake API's answer to `method path`.
fn route(method: &str, path: &str, accept: &str) -> (&'static str, String) {
    let path = path.strip_prefix("/repos/o/r").unwrap_or(path);
    let path = path.split_once('?').map_or(path, |(path, _)| path);
    let parts: Vec<&str> = path.trim_matches('/').split('/').collect();
    match (method, parts.as_slice()) {
        ("GET", ["collaborators", user, "permission"]) => {
            let permission = if *user == "writer" { "write" } else { "read" };
            ("200 OK", json!({ "permission": permission }).to_string())
        }
        ("POST", ["issues", _, "comments"]) => ("201 Created", json!({ "id": 42 }).to_string()),
        ("PATCH", ["issues", "comments", _]) => ("200 OK", "{}".to_owned()),
        ("GET", ["issues", _, "comments"]) => ("200 OK", "[]".to_owned()),
        ("GET", ["pulls", "3", "comments"]) => (
            "200 OK",
            json!([
                { "id": 5, "body": "This can be None.", "user": { "login": "reviewer" } },
                { "id": 6, "in_reply_to_id": 5, "body": "@rig handle the None case here", "user": { "login": "writer" } },
                { "id": 8, "body": "Elsewhere.", "user": { "login": "reviewer" } },
            ])
            .to_string(),
        ),
        ("GET", ["pulls", "3"]) if accept.contains("diff") => (
            "200 OK",
            "diff --git a/src/x.rs b/src/x.rs\n+pub fn x() -> Option<u8> { None }\n".to_owned(),
        ),
        ("GET", ["pulls", "3"]) => (
            "200 OK",
            json!({ "head": { "ref": "feature", "repo": { "full_name": "o/r" } }, "base": { "ref": "main" } })
                .to_string(),
        ),
        _ => ("404 Not Found", "{}".to_owned()),
    }
}

/// Runs `git args` in `dir`; its output.
fn git(dir: &Path, args: &[&str]) -> String {
    let output = Command::new("git").args(args).current_dir(dir).output();
    output
        .map(|out| String::from_utf8_lossy(&out.stdout).into_owned())
        .unwrap_or_default()
}

/// A bare `remote.git` with `main` and `feature`, and its clone `work`.
fn fixture(name: &str) -> Option<PathBuf> {
    let exe = std::env::current_exe().ok()?;
    let dir = exe
        .parent()?
        .join(format!("oc-github-{name}-{}", std::process::id()));
    fs::remove_dir_all(&dir).ok();
    let seed = dir.join("seed");
    fs::create_dir_all(seed.join("src")).ok()?;
    fs::write(seed.join("README.md"), "# r\n").ok()?;
    fs::write(seed.join("src/x.rs"), "pub fn x() -> Option<u8> { None }\n").ok()?;
    git(&dir, &["init", "-q", "--bare", "-b", "main", "remote.git"]);
    for args in [
        &["init", "-q", "-b", "main"][..],
        &[
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@t",
            "-c",
            "commit.gpgsign=false",
            "add",
            "-A",
        ],
        &[
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@t",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "-q",
            "-m",
            "start",
        ],
        &["push", "-q", "../remote.git", "main", "main:feature"],
    ] {
        git(&seed, args);
    }
    git(&dir, &["clone", "-q", "remote.git", "work"]);
    git(&dir.join("work"), &["config", "commit.gpgsign", "false"]);
    Some(dir)
}

/// The test's `shell`, which never runs anything.
#[derive(Deserialize, schemars::JsonSchema)]
struct ShellArgs {
    /// The command.
    #[allow(dead_code)]
    command: String,
}

/// The end of a run.
#[derive(Resource, Default)]
struct Over(bool);

/// One `rig github` run in `dir/work` for `event`, answered by `model`.
fn run(dir: &Path, home: &Home, api: &str, event: &Value, model: &MockCompletionModel) {
    let event_path = dir.join("event.json");
    fs::write(&event_path, event.to_string()).ok();
    let (mut app, wakes) = app_in(home, &SessionId::generate());
    let Ok(runtime) = rig_reqwest::runtime() else {
        return;
    };
    let invocation = Invocation {
        command: Some(FRONT.to_owned()),
        model: Some("test/model".to_owned()),
        ..Invocation::default()
    };
    app.insert_resource(Invoked {
        args: invocation,
        terminal: false,
    })
    .insert_resource(GithubEnv {
        event_path,
        api_url: api.to_owned(),
        token: "t".to_owned(),
        server_url: "https://github.example".to_owned(),
        run_id: Some("9".to_owned()),
    })
    .insert_resource(ProjectRoot(dir.join("work")))
    .insert_resource(TokioRuntime(runtime))
    .insert_resource(scripted_models([("test/model", model)]))
    .add_plugins((
        ServicesPlugin,
        AgentPlugin,
        SessionLogPlugin,
        WriteTool,
        GithubMentionPlugin,
    ))
    .add_open_tool(
        "shell",
        "Run a command.",
        ToolOptions {
            rules: &[],
            footprint: Some(Touches::Writes(Scope::Anything)),
        },
        |called: On<ToolCalled<ShellArgs>>, mut calls: Query<&mut Decision, ToolCalls>| {
            if let Ok(mut call) = calls.get_mut(called.call) {
                call.decide(Decider::plugin("test"), ToolResult::success("ran".into()));
            }
        },
    )
    .init_resource::<Over>()
    .add_observer(|_: On<HeadlessEnded>, mut over: ResMut<Over>| over.0 = true);
    app.finish();
    app.update();
    let agent = first_agent(app.world_mut()).unwrap_or(Entity::PLACEHOLDER);
    app.world_mut().trigger(SetModel {
        entity: agent,
        model: "test/model".to_owned(),
    });
    run_until(&mut app, &wakes, |world| world.resource::<Over>().0);
}

/// An issue comment event on issue 7 by `user`.
fn issue_comment(user: &str, text: &str) -> Value {
    json!({
        "action": "created",
        "comment": { "body": text, "user": { "login": user } },
        "issue": { "number": 7, "title": "A badge", "body": "The README needs a badge." },
        "repository": { "full_name": "o/r" },
    })
}

/// Everything the model was sent, as text.
fn sent(model: &MockCompletionModel) -> String {
    format!("{:?}", model.requests())
}

#[test]
fn a_mention_on_an_issue_works_on_its_branch_replies_and_continues_the_thread() {
    let (dir, server) = (fixture("issue"), fake_api());
    assert!(dir.is_some() && server.is_some(), "no fixture");
    let (Some(dir), Some((api, log))) = (dir, server) else {
        return;
    };
    let home = scratch_home("oc-github-issue");
    let remote = dir.join("remote.git");
    let requests = || log.lock().map(|log| log.clone()).unwrap_or_default();

    // A reader mentions the agent: nothing happens.
    let unused = MockCompletionModel::from_stream_turns([reply("no")]);
    run(
        &dir,
        &home,
        &api,
        &issue_comment("reader", "@rig add a README badge"),
        &unused,
    );
    assert!(
        requests().iter().all(|(method, ..)| method == "GET"),
        "{:?}",
        requests()
    );
    assert!(git(&remote, &["branch", "--list", "rig/issue-7"]).is_empty());

    // A writer does: the reply, the branch, and a denied command.
    let content = json!({ "path": "README.md", "content": "# r\n![badge](b.svg)\n" });
    let model = MockCompletionModel::from_stream_turns([
        calls("c1", "write", content),
        calls("c2", "shell", json!({ "command": "rm -rf /" })),
        reply("Added the badge."),
    ]);
    run(
        &dir,
        &home,
        &api,
        &issue_comment("writer", "@rig add a README badge"),
        &model,
    );
    let writes: Vec<Request> = requests()
        .into_iter()
        .filter(|(method, ..)| method != "GET")
        .collect();
    let posted = writes.iter().position(|(method, path, body)| {
        method == "POST" && path.ends_with("/issues/7/comments") && body.contains("Working on it")
    });
    assert_eq!(posted, Some(0), "{writes:?}");
    let last = writes.last().cloned().unwrap_or_default();
    assert!(
        last.0 == "PATCH" && last.1.ends_with("/issues/comments/42"),
        "{last:?}"
    );
    assert!(
        last.2.contains("Added the badge.") && last.2.contains("rm -rf /"),
        "{last:?} {}",
        sent(&model)
    );
    let changed = git(
        &remote,
        &["log", "-1", "--name-only", "--format=", "rig/issue-7"],
    );
    assert!(changed.contains("README.md"), "{changed:?}");

    // A later comment continues the conversation.
    let next = MockCompletionModel::from_stream_turns([reply("Done too.")]);
    run(
        &dir,
        &home,
        &api,
        &issue_comment("writer", "@rig also add a license badge"),
        &next,
    );
    let text = sent(&next);
    assert!(
        text.contains("Added the badge.") && text.contains("also add a license badge"),
        "{text}"
    );
}

#[test]
fn a_review_comment_gets_the_diff_and_its_place_and_commits_to_the_pull_request() {
    let (dir, server) = (fixture("review"), fake_api());
    assert!(dir.is_some() && server.is_some(), "no fixture");
    let (Some(dir), Some((api, _log))) = (dir, server) else {
        return;
    };
    let home = scratch_home("oc-github-review");
    let event = json!({
        "action": "created",
        "comment": { "id": 6, "in_reply_to_id": 5, "body": "@rig handle the None case here", "user": { "login": "writer" }, "path": "src/x.rs", "line": 12 },
        "pull_request": { "number": 3, "title": "Add x", "body": "" },
        "repository": { "full_name": "o/r" },
    });
    let content =
        json!({ "path": "src/x.rs", "content": "pub fn x() -> Option<u8> { Some(0) }\n" });
    let model: MockCompletionModel = MockCompletionModel::from_stream_turns([
        calls("c1", "write", content),
        reply("Handled the None case."),
    ]);
    run(&dir, &home, &api, &event, &model);
    let first = model
        .requests()
        .first()
        .map(|request| format!("{request:?}"))
        .unwrap_or_default();
    assert!(
        first.contains("diff --git a/src/x.rs")
            && first.contains("`src/x.rs` line 12")
            && first.contains("This can be None.")
            && !first.contains("Elsewhere."),
        "{first}"
    );
    let changed = git(
        &dir.join("remote.git"),
        &["log", "-1", "--name-only", "--format=", "feature"],
    );
    assert!(changed.contains("src/x.rs"), "{changed:?}");
}

#[test]
fn stages_are_ordered_with_the_defaults() {
    rig_defaults_test_support::assert_stages_ordered(GithubMentionPlugin);
}
