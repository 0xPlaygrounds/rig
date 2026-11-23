use super::*;
use rig_core::completion::Usage;
use rig_core::error::ErrorKind;
use rig_core::message::{Reasoning, ToolCall, ToolFunction, ToolName};
use rig_core::tool::{ToolContext, ToolErrorKind, ToolExecutionError, ToolOutput, ToolResult};
use rig_harness::links::SessionLink;

fn action(name: &str, parameter: &str, result: Option<&str>) -> Activity {
    Activity::action(
        name.to_owned(),
        parameter.to_owned(),
        result.map(str::to_owned),
    )
}

fn response(choice: Vec<AssistantContent>) -> CompletionResponse {
    CompletionResponse::new(
        choice,
        Usage::default(),
        rig_core::message::Origin::new("messages", "anthropic", "claude"),
        Value::Null,
    )
}

fn tool_call() -> Vec<AssistantContent> {
    ToolName::new("shell")
        .map(|name| {
            AssistantContent::ToolCall(ToolCall::from_wire(
                "call-1",
                ToolFunction::new(name, json!({ "command": "ls" })),
            ))
        })
        .into_iter()
        .collect()
}

fn ok(text: &str) -> Result<Answered, ErrorReport> {
    Ok(Answered::ToolResult {
        result: ToolResult::success(ToolOutput::text(text)),
    })
}

fn timed_out(text: &str) -> Result<Answered, ErrorReport> {
    Ok(Answered::ToolResult {
        result: ToolResult::failed(ToolExecutionError::new(ToolErrorKind::Timeout, text)),
    })
}

fn exited(code: i32) -> ToolResultContext {
    let mut context = ToolContext::default();
    context.insert_result(ExitStatus::Exited(code)).ok();
    context.result_context()
}

fn call<'a>(name: &'a str, args: &'a str, outcome: &'a Result<Answered, ErrorReport>) -> Call<'a> {
    Call {
        name,
        args,
        outcome,
        published: None,
    }
}

#[test]
fn each_content_variant_is_the_json_linear_takes() {
    assert_eq!(
        Content::Thought("hm".into()).json(),
        json!({ "type": "thought", "body": "hm" })
    );
    assert_eq!(
        Content::Elicitation("which?".into()).json(),
        json!({ "type": "elicitation", "body": "which?" })
    );
    assert_eq!(
        Content::Response("done".into()).json(),
        json!({ "type": "response", "body": "done" })
    );
    assert_eq!(
        Content::Error("bad".into()).json(),
        json!({ "type": "error", "body": "bad" })
    );
}

#[test]
fn an_action_carries_its_result_only_when_it_has_one() {
    let without = Content::Action {
        action: "Ran".into(),
        parameter: "ls".into(),
        result: None,
    };
    let with = Content::Action {
        action: "Ran".into(),
        parameter: "ls".into(),
        result: Some("exit code 2".into()),
    };
    assert_eq!(
        without.json(),
        json!({ "type": "action", "action": "Ran", "parameter": "ls" })
    );
    assert_eq!(
        with.json(),
        json!({
            "type": "action",
            "action": "Ran",
            "parameter": "ls",
            "result": "exit code 2",
        })
    );
}

#[test]
fn only_a_response_an_error_or_a_question_ends_the_run() {
    assert!(Content::Response(String::new()).ends());
    assert!(Content::Error(String::new()).ends());
    assert!(Content::Elicitation(String::new()).ends());
    assert!(!Content::Thought(String::new()).ends());
    assert!(
        !Content::Action {
            action: String::new(),
            parameter: String::new(),
            result: None,
        }
        .ends()
    );
}

#[test]
fn the_constructors_keep_only_thoughts_and_actions_ephemeral() {
    assert!(Activity::passing("x".into()).ephemeral);
    assert!(!Activity::note("x".into()).ephemeral);
    assert!(action("Ran", "ls", None).ephemeral);
    assert!(!Activity::end(Content::Response("x".into())).ephemeral);
}

#[test]
fn head_leaves_short_text_and_trims_it() {
    assert_eq!(head("  short  ", 10), "short");
    assert_eq!(head("exact", 5), "exact");
}

#[test]
fn head_cuts_long_text_with_an_ellipsis() {
    assert_eq!(head("abcdefghij", 4), "abcd…");
    assert_eq!(head("ab   cdef", 5), "ab…");
}

#[test]
fn head_counts_characters_not_bytes() {
    assert_eq!(head("héllo wörld", 5), "héllo…");
    assert_eq!(head("日本語のテキスト", 3), "日本語…");
    assert_eq!(head("日本語", 3), "日本語");
}

#[test]
fn tail_leaves_short_text_and_trims_it() {
    assert_eq!(tail("  short\n", 10), "short");
    assert_eq!(tail("exact", 5), "exact");
}

#[test]
fn tail_keeps_the_end_of_long_text_after_an_ellipsis() {
    assert_eq!(tail("abcdefghij", 4), "…ghij");
    assert_eq!(tail("abcd   efgh", 5), "…efgh");
}

#[test]
fn tail_counts_characters_not_bytes() {
    assert_eq!(tail("héllo wörld", 5), "…wörld");
    assert_eq!(tail("日本語のテキスト", 3), "…キスト");
}

#[test]
fn a_reply_reads_its_text_and_trims_it() {
    let reply = Reply::of(&response(vec![
        AssistantContent::text("  Hello "),
        AssistantContent::text("world\n"),
    ]));
    assert_eq!(
        reply,
        Reply {
            text: "Hello world".into(),
            reasoning: String::new(),
            calls_tools: false,
        }
    );
}

#[test]
fn a_reply_shows_reasoning_the_provider_did_not_withhold() {
    let reply = Reply::of(&response(vec![
        AssistantContent::Reasoning(Reasoning::new("first")),
        AssistantContent::Reasoning(Reasoning {
            text: "hidden".into(),
            redacted: true,
            ..Reasoning::default()
        }),
        AssistantContent::Reasoning(Reasoning::new("second")),
        AssistantContent::text("answer"),
    ]));
    assert_eq!(
        reply,
        Reply {
            text: "answer".into(),
            reasoning: "first\n\nsecond".into(),
            calls_tools: false,
        }
    );
}

#[test]
fn a_reply_with_a_tool_call_goes_on() {
    let reply = Reply::of(&response(tool_call()));
    assert_eq!(
        reply,
        Reply {
            text: String::new(),
            reasoning: String::new(),
            calls_tools: true,
        }
    );
}

#[test]
fn a_reply_becomes_a_passing_thought_for_reasoning_then_a_note_for_text() {
    let reply = Reply {
        text: "answer".into(),
        reasoning: "thinking".into(),
        calls_tools: false,
    };
    assert_eq!(
        reply.activities(),
        vec![
            Activity::passing("thinking".into()),
            Activity::note("answer".into()),
        ]
    );
}

#[test]
fn an_empty_reply_becomes_no_activity() {
    assert_eq!(Reply::default().activities(), Vec::<Activity>::new());
}

#[test]
fn a_long_reply_text_is_cut_to_two_thousand_characters() {
    let reply = Reply {
        text: "a".repeat(2500),
        ..Reply::default()
    };
    let expected = format!("{}…", "a".repeat(2000));
    assert_eq!(reply.activities(), vec![Activity::note(expected)]);
}

#[test]
fn a_shell_call_is_ran_with_its_command() {
    let outcome = ok("");
    assert_eq!(
        call("shell", r#"{"command":"cargo test"}"#, &outcome).activity(),
        action("Ran", "cargo test", None)
    );
}

#[test]
fn a_read_call_is_read_with_its_path() {
    let outcome = ok("");
    assert_eq!(
        call("read", r#"{"path":"/a/b.rs"}"#, &outcome).activity(),
        action("Read", "/a/b.rs", None)
    );
}

#[test]
fn a_write_or_edit_call_is_edited_with_its_path() {
    let outcome = ok("");
    for name in ["write", "edit"] {
        assert_eq!(
            call(name, r#"{"path":"/a/b.rs","content":"x"}"#, &outcome).activity(),
            action("Edited", "/a/b.rs", None)
        );
    }
}

#[test]
fn a_search_call_is_searched_with_its_pattern_and_path() {
    let outcome = ok("");
    assert_eq!(
        call("search", r#"{"pattern":"fn main","path":"src"}"#, &outcome).activity(),
        action("Searched", "fn main in src", None)
    );
    assert_eq!(
        call("search", r#"{"pattern":"fn main"}"#, &outcome).activity(),
        action("Searched", "fn main", None)
    );
}

#[test]
fn a_task_call_is_delegated_with_its_compact_arguments() {
    let outcome = ok("");
    assert_eq!(
        call("task", "{ \"prompt\": \"do it\" }", &outcome).activity(),
        action("Delegated", r#"{"prompt":"do it"}"#, None)
    );
}

#[test]
fn an_mcp_tool_call_is_called_with_the_tool_name() {
    let outcome = ok("");
    assert_eq!(
        call("github__create_issue", r#"{ "title": "t" }"#, &outcome).activity(),
        action("Called create_issue", r#"{"title":"t"}"#, None)
    );
}

#[test]
fn another_tool_call_is_its_name_with_compact_arguments() {
    let outcome = ok("");
    assert_eq!(
        call("custom", r#"{ "a": 1 }"#, &outcome).activity(),
        action("custom", r#"{"a":1}"#, None)
    );
    assert_eq!(
        call("custom", "{}", &outcome).activity(),
        action("custom", "", None)
    );
    assert_eq!(
        call("custom", "not json", &outcome).activity(),
        action("custom", "not json", None)
    );
}

#[test]
fn a_long_parameter_is_cut_to_three_hundred_characters() {
    let outcome = ok("");
    let args = json!({ "command": "x".repeat(400) }).to_string();
    let expected = format!("{}…", "x".repeat(300));
    assert_eq!(
        call("shell", &args, &outcome).activity(),
        action("Ran", &expected, None)
    );
}

#[test]
fn a_tool_error_is_its_kind_then_the_end_of_its_output() {
    let outcome = timed_out("took too long");
    assert_eq!(
        call("shell", r#"{"command":"sleep 99"}"#, &outcome).activity(),
        action("Ran", "sleep 99", Some("timeout\ntook too long"))
    );
}

#[test]
fn a_tool_error_keeps_only_the_last_three_hundred_characters() {
    let outcome = timed_out(&format!("{}{}", "a".repeat(500), "b".repeat(300)));
    let expected = format!("timeout\n…{}", "b".repeat(300));
    assert_eq!(
        call("shell", r#"{"command":"x"}"#, &outcome).activity(),
        action("Ran", "x", Some(&expected))
    );
}

#[test]
fn a_shell_call_that_exited_non_zero_fails_with_its_code_and_output() {
    let outcome = ok(&format!("{}{}", "a".repeat(500), "tail end"));
    let published = exited(2);
    let activity = Call {
        name: "shell",
        args: r#"{"command":"make"}"#,
        outcome: &outcome,
        published: Some(&published),
    }
    .activity();
    let expected = format!("exit code 2\n…{}tail end", "a".repeat(292));
    assert_eq!(activity, action("Ran", "make", Some(&expected)));
}

#[test]
fn a_shell_call_that_exited_zero_has_no_result() {
    let outcome = ok("fine");
    let published = exited(0);
    let activity = Call {
        name: "shell",
        args: r#"{"command":"make"}"#,
        outcome: &outcome,
        published: Some(&published),
    }
    .activity();
    assert_eq!(activity, action("Ran", "make", None));
}

#[test]
fn a_call_that_never_answered_fails_with_its_error_message() {
    let outcome = Err(ErrorReport::new(ErrorKind::Internal, "boom"));
    assert_eq!(
        call("read", r#"{"path":"/a"}"#, &outcome).activity(),
        action("Read", "/a", Some("failed\nboom"))
    );
}

#[test]
fn a_pull_request_url_is_found_after_gh_pr_create() {
    let output = "Creating pull request\nhttps://github.com/acme/widgets/pull/42\n";
    assert_eq!(
        pull_request("gh pr create --fill", output),
        Some(SessionLink::new(
            "Pull request #42",
            "https://github.com/acme/widgets/pull/42"
        ))
    );
}

#[test]
fn a_pull_request_url_is_ignored_unless_the_command_created_one() {
    let output = "https://github.com/acme/widgets/pull/42";
    assert_eq!(pull_request("gh pr view 42", output), None);
    assert_eq!(pull_request("echo hi", output), None);
}

#[test]
fn the_last_pull_request_url_wins() {
    let output = "https://github.com/acme/widgets/pull/1 and https://github.com/acme/other/pull/7.";
    assert_eq!(
        pull_request("gh pr create", output),
        Some(SessionLink::new(
            "Pull request #7",
            "https://github.com/acme/other/pull/7"
        ))
    );
}

#[test]
fn issue_urls_and_non_numeric_pull_requests_are_ignored() {
    assert_eq!(
        pull_request("gh pr create", "https://github.com/acme/widgets/issues/42"),
        None
    );
    assert_eq!(
        pull_request("gh pr create", "https://github.com/acme/widgets/pull/new"),
        None
    );
    assert_eq!(pull_request("gh pr create", "no url here"), None);
}

#[test]
fn a_given_secret_is_replaced() {
    let redactor = Redactor::new(["hunter2-secret".to_owned()]);
    assert_eq!(
        redactor.text("password is hunter2-secret, twice: hunter2-secret"),
        "password is [REDACTED], twice: [REDACTED]"
    );
}

#[test]
fn an_empty_secret_redacts_nothing() {
    let mut redactor = Redactor::new([String::new()]);
    redactor.add(String::new());
    assert_eq!(redactor.text("plain text"), "plain text");
}

#[test]
fn a_secret_added_later_is_replaced() {
    let mut redactor = Redactor::default();
    redactor.add("later-secret".to_owned());
    assert_eq!(redactor.text("a later-secret b"), "a [REDACTED] b");
}

#[test]
fn a_secret_rig_core_recognises_is_redacted() {
    let key = "sk-ant-api03-abcdefghijklmnopqrstuvwxyz";
    let token = format!("ghp_{}", "a1B2c3D4e5".repeat(4));
    let text = Redactor::default().text(&format!("key {key} token {token}"));
    assert!(!text.contains(key));
    assert!(!text.contains(&token));
    assert!(text.starts_with("key "));
}

#[test]
fn redacting_an_action_redacts_every_string_of_it() {
    let redactor = Redactor::new(["s3cret".to_owned()]);
    let activity = action("Ran s3cret", "echo s3cret", Some("saw s3cret"));
    assert_eq!(
        redactor.activity(activity),
        action("Ran [REDACTED]", "echo [REDACTED]", Some("saw [REDACTED]"))
    );
}

#[test]
fn redacting_a_note_keeps_it_persistent_and_redacts_its_body() {
    let redactor = Redactor::new(["s3cret".to_owned()]);
    assert_eq!(
        redactor.activity(Activity::note("a s3cret".into())),
        Activity::note("a [REDACTED]".into())
    );
    assert_eq!(
        redactor.activity(Activity::end(Content::Error("s3cret".into()))),
        Activity::end(Content::Error("[REDACTED]".into()))
    );
}
