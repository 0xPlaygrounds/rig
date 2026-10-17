use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

use rig_core::completion::CompletionRequest;
use rig_core::test_utils::{MockCompletionModel, MockStreamEvent};
use rig_harness::harness_protocol::{Invocation, SessionId};
use rig_harness_test_support::{
    AnswerAsks, Answers, Collected, Ended, ToolOutputs, app_in, calls, collect, connect,
    first_agent, log_ended, log_tool_outputs, reply, run_until, scratch_home,
};

use super::*;

/// The team's rules, as the spec's test has them, and a session and a
/// prompt hook that add context.
const PROJECT_HOOKS: &str = r#"
[[hook]]
event = "before_tool"
matcher = "shell"
command = "sh .rig/block-curl.sh"

[[hook]]
event = "after_tool"
matcher = "edit"
command = "sh .rig/lint.sh"

[[hook]]
event = "turn_end"
command = "sh .rig/needs-tests.sh"

[[hook]]
event = "session_start"
command = '''echo '{"context":"Team rule: no network in tests."}' '''

[[hook]]
event = "prompt_submit"
command = '''echo '{"context":"Ticket: ABC-1"}' '''
"#;

/// Refuses `curl`, and has every other command say it was hooked.
const BLOCK_CURL: &str = r#"
command=$(sed -n 's/.*"command":"\([^"]*\)".*/\1/p')
case "$command" in *curl*) echo "network not allowed" >&2; exit 2;; esac
printf '{"args":{"command":"%s && echo hooked"}}' "$command"
"#;

/// Finds one warning in each edited file.
const LINT: &str = r#"
path=$(sed -n 's/.*"arguments":{"path":"\([^"]*\)".*/\1/p')
printf '{"context":"lint: 1 warning in %s"}' "$path"
"#;

/// Carries the turn on while `NEEDS_TESTS` is there, taking it away.
const NEEDS_TESTS: &str = r#"
cat > /dev/null
if [ -f NEEDS_TESTS ]; then rm NEEDS_TESTS; echo '{"decision":"continue","reason":"run the tests"}'; fi
"#;

/// A project in `home` with the hooks above.
fn project(home: &Home) -> std::path::PathBuf {
    let dir = home.root().join("project");
    std::fs::create_dir_all(dir.join(".rig")).ok();
    for (file, text) in [
        ("hooks.toml", PROJECT_HOOKS),
        ("block-curl.sh", BLOCK_CURL),
        ("lint.sh", LINT),
        ("needs-tests.sh", NEEDS_TESTS),
    ] {
        std::fs::write(dir.join(".rig").join(file), text).ok();
    }
    std::fs::write(dir.join("a.txt"), "a\n").ok();
    dir
}

/// What a run did.
struct Run {
    outputs: HashMap<String, String>,
    notices: Vec<Notice>,
    requests: Vec<CompletionRequest>,
    asked: Vec<String>,
    took: Duration,
}

impl Run {
    fn output(&self, id: &str) -> &str {
        self.outputs.get(id).map(String::as_str).unwrap_or_default()
    }

    fn request(&self, at: usize) -> String {
        self.requests
            .get(at)
            .map(|request| format!("{request:?}"))
            .unwrap_or_default()
    }

    fn warned(&self, text: &str) -> bool {
        self.notices.iter().any(|notice| notice.text.contains(text))
    }
}

/// Runs one prompt in `dir` against `turns`, with the trust flag when
/// `trust`, and someone answering `answers` when given, until its turn
/// ended.
fn run(
    home: &Home,
    dir: &Path,
    trust: bool,
    answers: Option<Answers>,
    turns: Vec<Vec<MockStreamEvent>>,
) -> Run {
    let model = MockCompletionModel::from_stream_turns(turns);
    let (mut app, wakes) = app_in(home, &SessionId::generate());
    let flags = if trust {
        vec![(TRUST_FLAG.to_owned(), None)]
    } else {
        Vec::new()
    };
    let args = Invocation {
        flags,
        ..Invocation::default()
    };
    app.insert_resource(Invoked {
        args,
        terminal: false,
    })
    .insert_resource(ProjectRoot(dir.to_path_buf()))
    .init_resource::<Ended>()
    .add_observer(log_ended)
    .init_resource::<ToolOutputs>()
    .add_observer(log_tool_outputs)
    .add_plugins((
        AgentPlugin,
        SessionLogPlugin,
        rig_basics::coding_tools::ShellTool,
        rig_basics::coding_tools::EditTool,
        HookScriptsPlugin,
    ));
    let asks = answers.is_some();
    if let Some(answers) = answers {
        app.insert_resource(answers).add_plugins(AnswerAsks);
    }
    collect::<Notice>(&mut app);
    app.finish();
    app.update();
    let world = app.world_mut();
    let agent = first_agent(world).unwrap_or(Entity::PLACEHOLDER);
    if let Some(connection) = connect(&model) {
        world.entity_mut(agent).insert(connection);
    }
    world.trigger(Deliver::new(
        agent,
        "Check the site, then fix a.txt.",
        DeliveryMode::Steer,
    ));
    let started = Instant::now();
    run_until(&mut app, &wakes, |world| {
        !world.resource::<Ended>().0.is_empty()
    });
    let took = started.elapsed();
    // A last few frames: nothing more may happen.
    for _ in 0..5 {
        app.update();
    }
    let world = app.world_mut();
    Run {
        outputs: world.resource::<ToolOutputs>().0.clone(),
        notices: world.resource::<Collected<Notice>>().0.clone(),
        requests: model.requests(),
        asked: if asks {
            world.resource::<Answers>().asked.clone()
        } else {
            Vec::new()
        },
        took,
    }
}

/// The model's first reply: the three calls of the spec's test.
fn three_calls() -> Vec<MockStreamEvent> {
    let edit =
        serde_json::json!({ "path": "a.txt", "edits": [{ "old_text": "a", "new_text": "b" }] });
    vec![
        MockStreamEvent::tool_call(
            "c1",
            "shell",
            serde_json::json!({ "command": "curl example.com" }),
        ),
        MockStreamEvent::tool_call("c2", "shell", serde_json::json!({ "command": "echo hi" })),
        MockStreamEvent::tool_call("c3", "edit", edit),
        MockStreamEvent::final_response_with_default_usage(),
    ]
}

/// The spec's test: a trusted project's hooks block `curl`, rewrite the
/// other command, lint the edit and carry the turn on once; its session
/// and prompt hooks add context.
#[test]
fn project_hooks_block_rewrite_lint_and_carry_the_turn_on() {
    let home = scratch_home("corpus-sn-hook-scripts-trusted");
    let dir = project(&home);
    std::fs::write(dir.join("NEEDS_TESTS"), "").ok();
    let turns = vec![three_calls(), reply("Fixed."), reply("The tests pass.")];
    let run = run(&home, &dir, true, None, turns);

    // Request 1 carries the session's and the prompt's context.
    let first = run.request(0);
    assert!(first.contains("Team rule: no network in tests."), "{first}");
    assert!(first.contains("Ticket: ABC-1"), "{first}");
    // Request 2: curl refused with the script's message, the other
    // command rewritten, the edit linted.
    assert!(
        run.output("c1").contains("network not allowed"),
        "{}",
        run.output("c1")
    );
    assert!(
        run.output("c2").contains("hi\nhooked"),
        "{}",
        run.output("c2")
    );
    assert!(
        run.output("c3")
            .trim_end()
            .ends_with("lint: 1 warning in a.txt"),
        "{}",
        run.output("c3")
    );
    let second = run.request(1);
    for wanted in ["network not allowed", "hooked", "lint: 1 warning in a.txt"] {
        assert!(second.contains(wanted), "{wanted}: {second}");
    }
    // The turn went on once, with the hook's reason, then ended.
    assert!(
        run.request(2).contains("run the tests"),
        "{}",
        run.request(2)
    );
    assert_eq!(run.requests.len(), 3);
    assert!(!dir.join("NEEDS_TESTS").exists());
    std::fs::remove_dir_all(home.root()).ok();
}

/// Headless, an untrusted project's hooks are skipped with a warning; once
/// the user trusts them they run, and the trust is remembered until the
/// file changes.
#[test]
fn project_hooks_run_only_once_trusted() {
    let home = scratch_home("corpus-sn-hook-scripts-untrusted");
    let dir = project(&home);
    let echo = || {
        vec![
            calls("c1", "shell", serde_json::json!({ "command": "echo hi" })),
            reply("Done."),
        ]
    };

    let skipped = run(&home, &dir, false, None, echo());
    assert!(
        !skipped.output("c1").contains("hooked"),
        "{}",
        skipped.output("c1")
    );
    assert!(skipped.warned("are not trusted and were skipped"));

    let yes = Answers::new([Answer::choice("yes")]);
    let asked = run(&home, &dir, false, Some(yes), echo());
    assert_eq!(asked.asked, [TRUST_QUESTION]);
    assert!(
        asked.output("c1").contains("hi\nhooked"),
        "{}",
        asked.output("c1")
    );

    let remembered = run(&home, &dir, false, None, echo());
    assert!(
        remembered.output("c1").contains("hooked"),
        "{}",
        remembered.output("c1")
    );

    std::fs::write(
        dir.join(".rig/hooks.toml"),
        format!("{PROJECT_HOOKS}\n# changed\n"),
    )
    .ok();
    let changed = run(&home, &dir, false, None, echo());
    assert!(
        !changed.output("c1").contains("hooked"),
        "{}",
        changed.output("c1")
    );
    std::fs::remove_dir_all(home.root()).ok();
}

/// A hook that sleeps past its 1 s timeout lets the call run, with a hook
/// error shown, or refuses it when it fails closed.
#[test]
fn a_slow_hook_times_out_open_or_closed() {
    for fail_closed in [false, true] {
        let home = scratch_home(&format!("corpus-sn-hook-scripts-slow-{fail_closed}"));
        let dir = home.root().join("project");
        std::fs::create_dir_all(&dir).ok();
        let hooks = format!(
            "[[hook]]\nevent = \"before_tool\"\nmatcher = \"shell\"\ncommand = \"sleep 30\"\ntimeout = 1\nfail_closed = {fail_closed}\n"
        );
        std::fs::write(home.root().join("hooks.toml"), hooks).ok();
        let turns = vec![
            calls("c1", "shell", serde_json::json!({ "command": "echo ran" })),
            reply("Done."),
        ];
        let run = run(&home, &dir, false, None, turns);

        assert!(
            run.took >= Duration::from_secs(1) && run.took < Duration::from_secs(8),
            "{:?}",
            run.took
        );
        assert!(run.warned("timed out after 1 s"));
        let output = run.output("c1");
        if fail_closed {
            assert!(
                output.contains("A hook failed") && !output.contains("ran\n"),
                "{output}"
            );
        } else {
            assert!(output.contains("ran"), "{output}");
        }
        std::fs::remove_dir_all(home.root()).ok();
    }
}

#[test]
fn stages_are_ordered_with_the_defaults() {
    rig_defaults_test_support::assert_stages_ordered(HookScriptsPlugin);
}
