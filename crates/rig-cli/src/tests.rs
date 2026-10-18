use std::cell::Cell;

use rig_client::{ClientError, Target};
use rig_harness_protocol::guide;
use rig_protocol::methods::{WaitResult, WaitStatus};
use rig_protocol::{CommandSeq, Exit, RpcError, Rule, Seq, SessionId, codes};

use crate::args::{Args, parse_duration};
use crate::out::Failure;
use crate::verbs::{self, Ctx};

fn words(words: &[&str]) -> Vec<String> {
    words.iter().map(|word| (*word).to_owned()).collect()
}

/// M-1: a prompt given on the command line never reads stdin, so a driver
/// whose stdin is an open pipe that never closes does not hang; `-`, or no
/// prompt where the verb reads one, does.
#[test]
fn stdin_is_read_only_for_dash_or_a_missing_prompt() {
    let read = Cell::new(0);
    let stdin = || {
        read.set(read.get() + 1);
        Ok("from stdin".to_owned())
    };
    let given = verbs::prompt(&words(&["fix", "it"]), true, stdin)
        .ok()
        .flatten();
    assert_eq!(given.as_deref(), Some("fix it"));
    assert_eq!(read.get(), 0, "a given prompt must not touch stdin");
    let none = verbs::prompt(&[], false, || {
        read.set(read.get() + 1);
        Ok(String::new())
    });
    assert!(matches!(none, Ok(None)));
    assert_eq!(
        read.get(),
        0,
        "rig start without a prompt must not touch stdin"
    );
    let dash = verbs::prompt(&words(&["-"]), false, || {
        read.set(read.get() + 1);
        Ok("piped".to_owned())
    });
    assert_eq!(dash.ok().flatten().as_deref(), Some("piped"));
    assert_eq!(read.get(), 1);
}

#[test]
fn options_are_checked_against_the_verb() {
    let parsed = Args::parse("wait", &words(&["s-1", "--timeout=30s", "--json"])).ok();
    let parsed = parsed.unwrap_or_default();
    assert_eq!(parsed.words, ["s-1"]);
    assert_eq!(parsed.value("timeout"), Some("30s"));
    assert!(parsed.has("json"));

    let refused = Args::parse("start", &words(&["--follow"])).err();
    let refused = refused.map(|failure| (failure.exit, failure.next));
    assert_eq!(refused, Some((Exit::Usage, "rig help start".to_owned())));

    // rig answer's --allow is a switch, rig start's takes a rule.
    let answer = Args::parse("answer", &words(&["s-1", "--allow"])).unwrap_or_default();
    assert!(answer.has("allow"));
    assert_eq!(answer.words, ["s-1"]);
    let start =
        Args::parse("start", &words(&["--allow", "shell:cargo test", "go"])).unwrap_or_default();
    assert_eq!(start.value("allow"), Some("shell:cargo test"));
    assert_eq!(start.words, ["go"]);

    let missing = Args::parse("start", &words(&["--name"]))
        .err()
        .map(|f| f.exit);
    assert_eq!(missing, Some(Exit::Usage));
}

#[test]
fn allow_rules() {
    assert_eq!(
        verbs::parse_rule("edit"),
        Some(Rule::Tool {
            name: "edit".to_owned()
        })
    );
    assert_eq!(
        verbs::parse_rule("shell:cargo test"),
        Some(Rule::ShellPrefix {
            prefix: "cargo test".to_owned()
        })
    );
    assert!(matches!(
        verbs::parse_rule("edits:/srv/app"),
        Some(Rule::EditsUnder { path }) if path == std::path::Path::new("/srv/app")
    ));
    assert_eq!(verbs::parse_rule("shell:"), None);
    assert_eq!(verbs::parse_rule("two words"), None);
}

#[test]
fn durations() {
    use std::time::Duration;
    assert_eq!(parse_duration("90"), Some(Duration::from_secs(90)));
    assert_eq!(parse_duration("10m"), Some(Duration::from_secs(600)));
    assert_eq!(parse_duration("1h"), Some(Duration::from_secs(3600)));
    assert_eq!(parse_duration("250ms"), Some(Duration::from_millis(250)));
    assert_eq!(parse_duration("soon"), None);
}

fn local() -> Ctx {
    Ctx {
        target: Target::Local {
            rig_home: "/nonexistent".into(),
        },
        json: false,
    }
}

/// Each wait status has its documented exit code, and each but an answer
/// says what to run next.
#[test]
fn wait_statuses_map_to_exit_codes_and_next() {
    let id = SessionId("s-1".to_owned());
    let cases = [
        (WaitStatus::Answered, 0, None),
        (WaitStatus::Failed, 1, Some("rig log s-1")),
        (WaitStatus::NeedsApproval, 3, Some("rig answer s-1 --allow")),
        (
            WaitStatus::NeedsAnswer,
            4,
            Some("rig answer s-1 --text \"...\""),
        ),
        (WaitStatus::Stopped, 5, Some("rig send s-1 \"...\"")),
        (WaitStatus::TimedOut, 124, Some("rig wait s-1")),
    ];
    for (status, code, next) in cases {
        let result = WaitResult {
            status,
            answer: None,
            ask: None,
            error: None,
            seq: Seq(3),
            command_seq: CommandSeq(1),
        };
        assert_eq!(status.exit().code(), code, "{status:?}");
        assert_eq!(
            verbs::wait_next(&local(), &id, &result).as_deref(),
            next,
            "{status:?}"
        );
    }
}

/// Every client error becomes a failure with a `next:` command.
#[test]
fn every_error_has_a_next_command() {
    let errors = [
        ClientError::Unavailable {
            next: String::new(),
        },
        ClientError::Io(std::io::Error::other("gone")),
        ClientError::Version {
            update: rig_protocol::Side::Server,
            supported: vec![1],
        },
        ClientError::Rpc(RpcError {
            code: codes::NOT_FOUND,
            message: "no session s-9".to_owned(),
            data: None,
        }),
    ];
    for error in errors {
        let failure = Failure::client(error, "rig agents --all");
        assert!(!failure.next.is_empty(), "{failure:?}");
        assert!(
            failure.next.starts_with("rig ") || failure.next.contains(" rig "),
            "{failure:?}"
        );
    }
    let unknown = crate::dispatch(&words(&["frobnicate"])).err();
    assert!(unknown.is_some_and(|failure| failure.exit == Exit::Usage && !failure.next.is_empty()));
}

/// M-6 (F-005): `rig help` resolves options as well as verbs.
#[test]
fn help_resolves_verbs_and_options() {
    for name in [
        "--allow",
        "--json",
        "-p",
        "start",
        "wait",
        "--timeout",
        "-r",
    ] {
        assert!(guide::help_agents(name).is_ok(), "{name}");
    }
    assert!(guide::help_agents("--nope").is_err());
    for verb in guide::VERBS {
        assert!(guide::verb_help(verb).contains(verb.usage));
    }
    for (name, _) in guide::TOPICS {
        assert!(guide::FIRST_SCREEN.contains(name), "{name} is listed");
    }
    let lines = guide::FIRST_SCREEN.lines().count();
    assert!(lines <= 30, "the first screen has {lines} lines");
    for option in guide::OPTIONS {
        for verb in option.verbs {
            assert!(
                guide::verb(verb).is_some(),
                "--{} names verb {verb}",
                option.name
            );
        }
    }
}

/// `rig connect --stdio` gives a remote client's tokenless hello this
/// host's token, and passes every other frame untouched.
#[test]
fn the_bridge_puts_the_local_token_in_a_tokenless_hello() {
    let hello = br#"{"jsonrpc":"2.0","rig":1,"id":1,"method":"rig/hello","params":{"token":"","client":{"name":"x","version":"1"},"rig":{"min":1,"max":1}}}"#;
    let rewritten = crate::serve::with_token(hello, "secret").unwrap_or_default();
    let frame: serde_json::Value = serde_json::from_slice(&rewritten).unwrap_or_default();
    assert_eq!(frame["params"]["token"], "secret");
    assert!(rewritten.ends_with(b"\n"));
    let with_own =
        br#"{"jsonrpc":"2.0","rig":1,"id":1,"method":"rig/hello","params":{"token":"mine"}}"#;
    assert_eq!(crate::serve::with_token(with_own, "secret"), None);
    let other = br#"{"jsonrpc":"2.0","rig":1,"id":2,"method":"rig/session/list","params":{}}"#;
    assert_eq!(crate::serve::with_token(other, "secret"), None);
}

#[test]
fn an_acp_prompt_becomes_the_session_message() {
    let blocks = serde_json::json!([
        {"type": "text", "text": "fix this"},
        {"type": "resource", "resource": {"uri": "file:///a.rs", "text": "fn a() {}"}},
        {"type": "resource_link", "uri": "file:///b.rs", "name": "b.rs"},
        {"type": "image", "data": "...", "mimeType": "image/png"},
    ]);
    let text = crate::acp::prompt_text(blocks.as_array().map(Vec::as_slice).unwrap_or_default());
    assert_eq!(text, "fix this\n\nfile:///a.rs:\nfn a() {}\n\n@/b.rs");
}

/// F-017: every verb takes --help and -h, printing its help.
#[test]
fn every_verb_takes_help() {
    for verb in guide::VERBS.iter().filter(|verb| verb.name != "-p") {
        for flag in ["--help", "-h"] {
            let code = crate::dispatch(&words(&[verb.name, flag])).ok();
            assert_eq!(code, Some(0), "rig {} {flag}", verb.name);
        }
    }
    assert_eq!(crate::dispatch(&words(&["-p", "--help"])).ok(), Some(0));
}

/// F-020: `rig help-agents VERB` (or an option) shows that help.
#[test]
fn help_agents_takes_a_verb_or_an_option() {
    for name in ["start", "stop", "--allow", "approvals"] {
        assert!(guide::help_agents(name).is_ok(), "{name}");
    }
    assert!(guide::help_agents("nope").is_err());
}

/// API F-08: stdout carries only the answer; an open ask goes to stderr
/// with the command that answers it.
#[test]
fn an_ask_goes_to_stderr() {
    let id = SessionId("s-1".to_owned());
    let result = |status| WaitResult {
        status,
        answer: (status == WaitStatus::Answered).then(|| "done".to_owned()),
        ask: None,
        error: None,
        seq: Seq(3),
        command_seq: CommandSeq(1),
    };
    let next = Some("rig answer s-1 --allow");
    let (out, err) = verbs::wait_text(&id, &result(WaitStatus::NeedsApproval), None, next);
    assert_eq!(out, "");
    assert!(err.contains("next: rig answer s-1 --allow"), "{err}");
    let (out, err) = verbs::wait_text(&id, &result(WaitStatus::Answered), None, None);
    assert_eq!((out.as_str(), err.as_str()), ("done", ""));
}

/// API F-06: a question's choices go to the editor as ACP permission
/// options, whose kind must be one ACP v1 knows.
#[test]
fn acp_permission_options_have_acp_kinds() {
    let options = serde_json::json!([
        {"optionId": "red", "label": "red", "kind": "other"},
        {"optionId": "allow", "label": "Allow", "kind": "allow_once"},
        {"optionId": "deny", "label": "Deny", "kind": "reject_once"},
    ]);
    let mapped =
        crate::acp::permission_options(options.as_array().map(Vec::as_slice).unwrap_or_default());
    let kinds: Vec<&str> = mapped.iter().filter_map(|o| o["kind"].as_str()).collect();
    assert_eq!(kinds, ["allow_once", "allow_once", "reject_once"]);
}

/// Code audit: only the first -p (or --print) is rig -p's; the same word
/// later, after `--` say, is the prompt's.
#[test]
fn only_the_first_print_flag_is_taken() {
    assert_eq!(
        crate::print_args(&words(&["-p", "--json", "--", "explain", "-p"])),
        words(&["--json", "--", "explain", "-p"])
    );
    assert_eq!(
        crate::print_args(&words(&["-m", "a/b", "--print", "hi"])),
        words(&["-m", "a/b", "hi"])
    );
}

/// API audit: an unknown --host is a usage error whose next: is a
/// settings.toml entry that parses.
#[test]
fn an_unknown_host_says_what_to_add() {
    let home = std::env::temp_dir().join(format!("rig-cli-host-{}", std::process::id()));
    std::fs::create_dir_all(&home).ok();
    let failure = verbs::remote_target(&home, "box").err();
    std::fs::remove_dir_all(&home).ok();
    let failure = failure.unwrap_or_else(|| Failure::usage("", ""));
    assert_eq!(failure.exit, Exit::Usage);
    assert!(
        failure
            .next
            .contains("[remote.hosts]\nbox = [\"ssh\", \"box\"]"),
        "{}",
        failure.next
    );
}

/// F-022: --ask makes a session ask before those calls; --allow rules
/// answer them up front.
#[test]
fn ask_sets_the_grant() {
    let args = Args::parse(
        "start",
        &words(&["--ask", "shell,edits", "--allow", "shell:cargo test", "go"]),
    )
    .unwrap_or_default();
    let grant = verbs::grant(&args, "start")
        .ok()
        .flatten()
        .unwrap_or_default();
    assert_eq!(grant.ask, ["shell", "edits"]);
    assert_eq!(grant.approve.len(), 1);
    assert!(grant.net && grant.dirs.is_none() && grant.tools.is_none());
    let bad = Args::parse("start", &words(&["--ask", "shell,,"])).unwrap_or_default();
    assert!(verbs::grant(&bad, "start").is_err());
}

/// F-012 for the CLI: with --host, --cwd is the remote host's: `~` and
/// absolute paths go as written, a relative one is refused, none is `~`.
#[test]
fn a_remote_cwd_is_the_hosts() {
    let cwd = |extra: &[&str]| {
        let mut all = vec!["--host", "box"];
        all.extend_from_slice(extra);
        verbs::cwd(
            &Args::parse("start", &words(&all)).unwrap_or_default(),
            "start",
        )
    };
    assert_eq!(cwd(&["--cwd", "~/app"]).ok(), Some("~/app".into()));
    assert_eq!(cwd(&["--cwd", "/srv/app"]).ok(), Some("/srv/app".into()));
    assert_eq!(cwd(&[]).ok(), Some("~".into()));
    let refused = cwd(&["--cwd", "app"])
        .err()
        .map(|f| f.message)
        .unwrap_or_default();
    assert!(refused.contains("remote"), "{refused}");
}

/// Analysis §3.5: a wait on a session whose server is gone answers from
/// its journal when the latest message's turn ended, and only otherwise
/// needs a server (to resume it).
#[test]
fn a_finished_wait_is_read_from_the_journal() {
    let record = |seq: u64, kind: &str, data: serde_json::Value| rig_protocol::EventRecord {
        session: SessionId("s-1".to_owned()),
        seq: Seq(seq),
        at_ms: 0,
        kind: kind.to_owned(),
        data,
        source: None,
    };
    let sent = |seq, command: u64| {
        record(
            seq,
            "command.accepted",
            serde_json::json!({"commandSeq": command, "method": "rig/session/send"}),
        )
    };
    let ended = |seq, through: u64| {
        record(
            seq,
            "turn.ended",
            serde_json::json!({"status": "answered", "answer": "done", "through": through}),
        )
    };
    let done = verbs::journal_outcome(&[sent(1, 1), ended(2, 1)]);
    assert_eq!(
        done.as_ref().map(|r| (r.status, r.answer.clone())),
        Some((WaitStatus::Answered, Some("done".to_owned())))
    );
    // A later message with no outcome yet: only a server can say.
    assert!(verbs::journal_outcome(&[sent(1, 1), ended(2, 1), sent(3, 2)]).is_none());
    // A stop is no message.
    let stop = record(
        3,
        "command.accepted",
        serde_json::json!({"commandSeq": 2, "method": "rig/session/stop"}),
    );
    assert!(verbs::journal_outcome(&[sent(1, 1), ended(2, 1), stop]).is_some());
}
