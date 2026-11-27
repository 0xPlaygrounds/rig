use super::*;
use crate::SessionLink;
use rig_core::completion::Usage;
use rig_core::message::Reasoning;
use rig_core::tool::{ToolErrorKind, ToolExecutionError, ToolOutput, ToolResult};

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

fn call<'a>(name: &'a str, args: &'a str, outcome: &'a Result<Answered, ErrorReport>) -> Call<'a> {
    Call {
        name,
        args,
        outcome,
        published: None,
    }
}

#[test]
fn redacted_reasoning_is_omitted_and_truncation_respects_char_boundaries() {
    let response = CompletionResponse::new(
        vec![
            AssistantContent::Reasoning(Reasoning::new("first")),
            AssistantContent::Reasoning(Reasoning {
                text: "hidden".into(),
                redacted: true,
                ..Reasoning::default()
            }),
            AssistantContent::Reasoning(Reasoning::new("second")),
            AssistantContent::text("  answer\n"),
        ],
        Usage::default(),
        rig_core::message::Origin::new("messages", "anthropic", "claude"),
        Value::Null,
    );
    assert_eq!(
        Reply::of(&response),
        Reply {
            text: "answer".into(),
            reasoning: "first\n\nsecond".into(),
            calls_tools: false,
        }
    );
    assert_eq!(head("日本語のテキスト", 3), "日本語…");
    assert_eq!(tail("日本語のテキスト", 3), "…キスト");
}

#[test]
fn pull_request_only_matches_numeric_pr_urls_from_gh_pr_create() {
    let multiple =
        "https://github.com/acme/widgets/pull/1 and https://github.com/acme/other/pull/7.";
    assert_eq!(
        pull_request("gh pr create", multiple),
        Some(SessionLink::new(
            "Pull request #7",
            "https://github.com/acme/other/pull/7"
        ))
    );
    assert_eq!(
        pull_request("gh pr view 42", "https://github.com/acme/widgets/pull/42"),
        None
    );
    assert_eq!(
        pull_request("gh pr create", "https://github.com/acme/widgets/issues/42"),
        None
    );
    assert_eq!(
        pull_request("gh pr create", "https://github.com/acme/widgets/pull/new"),
        None
    );
}

#[test]
fn secrets_straddling_truncation_cutoffs_are_scrubbed_before_cutting() {
    let secret = "lin_api_SUPER_SECRET_TOKEN_VALUE";
    let redactor = Redactor::new([secret.to_owned()]);

    // Action parameter crossing 300 chars.
    let cmd = format!("{}{secret}", "a".repeat(290));
    let outcome = ok("");
    let args = json!({ "command": cmd }).to_string();
    let act = call("shell", &args, &outcome).activity(&redactor);
    assert!(
        matches!(
            &act.content,
            Content::Action { parameter, .. }
                if !parameter.contains("lin_api_") && parameter.contains("[REDACTED]")
        ),
        "{act:?}"
    );

    // Failure tail crossing 300 chars.
    let err_out = format!("{secret}{}", "b".repeat(290));
    let fail = timed_out(&err_out);
    let act = call("shell", r#"{"command":"x"}"#, &fail).activity(&redactor);
    assert!(
        matches!(
            &act.content,
            Content::Action { result: Some(res), .. }
                if !res.contains("TOKEN_VALUE") && res.contains("[REDACTED]")
        ),
        "{act:?}"
    );

    // Reply text and reasoning crossing 2,000 chars.
    let reply = Reply {
        text: format!("{}{secret}", "c".repeat(1990)),
        reasoning: format!("{}{secret}", "d".repeat(1990)),
        calls_tools: false,
    };
    let activities = reply.activities(&redactor);
    assert_eq!(activities.len(), 2);
    for activity in activities {
        assert!(
            matches!(
                &activity.content,
                Content::Thought(body)
                    if !body.contains("lin_api_") && body.contains("[REDACTED]")
            ),
            "{activity:?}"
        );
    }
}
