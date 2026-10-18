use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use rig_core::message::{DocumentSourceKind, ImageMediaType, ToolResultContent};
use rig_core::tool::{ToolContext, ToolErrorKind};
use serde_json::{Value, json};

use crate::tools::{User, registry};
use crate::{Outcome, full_error, result_json, run_call, run_lines};

const TIMEOUT: Duration = Duration::from_millis(50);

#[tokio::test(start_paused = true)]
async fn one_call_runs_outside_any_agent_loop() {
    let add_calls = Arc::new(AtomicUsize::new(0));
    let tools = Arc::new(registry(Arc::clone(&add_calls)));
    let mut context = ToolContext::new();
    assert!(context.insert(User("u1".to_owned())).is_ok());
    let run = |line: &str| {
        let (tools, context, line) = (Arc::clone(&tools), context.clone(), line.to_owned());
        async move { run_call(&tools, &line, &context, TIMEOUT).await }
    };
    let ok = |line: &'static str| {
        let run = run(line);
        async move { result_json(&run.await) }
    };

    // R1: object or JSON-encoded string arguments run identically; R4 keeps the call id.
    let object = ok(r#"{"id":"c1","name":"add","arguments":{"a":2,"b":3}}"#).await;
    let string = ok(r#"{"id":"c1","name":"add","arguments":"{\"a\":2,\"b\":3}"}"#).await;
    assert_eq!(object, json!({"id": "c1", "ok": 5}));
    assert_eq!(string, object);
    let demo = r#"{"id":"c1","name":"convert_currency","arguments":"{\"amount\":10,\"from\":\"EUR\",\"to\":\"USD\"}"}"#;
    assert_eq!(ok(demo).await, json!({"id": "c1", "ok": "10.80 USD"}));
    assert_eq!(add_calls.load(Ordering::SeqCst), 2);

    // R2: a schema mismatch names where and why, and the body never runs.
    let missing = run(r#"{"name":"add","arguments":{"a":2}}"#).await;
    assert_eq!(
        problems(&missing.outcome),
        Some(vec!["/b: missing".to_owned()])
    );
    let feedback = missing.outcome.model_feedback().unwrap_or_default();
    assert!(feedback.contains("/b: missing"), "{feedback}");
    let wrong_type =
        run(r#"{"name":"convert_currency","arguments":{"amount":"ten","from":1,"to":"USD"}}"#)
            .await;
    let named = problems(&wrong_type.outcome).unwrap_or_default();
    assert!(
        named
            .first()
            .is_some_and(|it| it.starts_with("/amount: expected a number"))
    );
    assert!(
        named.get(1).is_some_and(|it| it.starts_with("/from:")),
        "{named:?}"
    );
    assert_eq!(add_calls.load(Ordering::SeqCst), 2);

    // R3: unknown tool, and a failure with model-safe text apart from the full error.
    let unknown = run(r#"{"name":"nope","arguments":{}}"#).await;
    assert!(matches!(&unknown.outcome, Outcome::UnknownTool { name } if name == "nope"));
    let failed = run(r#"{"name":"fail_always","arguments":{}}"#).await;
    let Outcome::Failed { error } = &failed.outcome else {
        return assert!(matches!(failed.outcome, Outcome::Failed { .. }));
    };
    assert_eq!(error.kind(), ToolErrorKind::Other);
    assert_eq!(error.model_output().as_text(), Some("the tool failed"));
    assert!(full_error(error).contains("password authentication failed"));
    let printed = result_json(&failed).to_string();
    assert!(!printed.contains("10.0.0.7"), "{printed}");

    // R6: a never-finishing tool times out under paused time.
    let slow = run(r#"{"name":"slow"}"#).await;
    assert!(matches!(slow.outcome, Outcome::TimedOut));

    // R5: context reaches the tool and nothing else.
    let whoami = ok(r#"{"id":"w","name":"whoami","arguments":{}}"#).await;
    assert_eq!(whoami, json!({"id": "w", "ok": "you are u1"}));
    let whoami = run(r#"{"id":"w","name":"whoami","arguments":{}}"#).await;
    let shown = success(&whoami.outcome)
        .and_then(|answer| serde_json::to_string(answer).ok())
        .unwrap_or_default();
    assert_eq!(shown.matches("u1").count(), 1, "{shown}");
    assert!(!shown.contains("\"user\""), "{shown}");

    // R4: multimodal output stays an image with its MIME type.
    let chart = run(r#"{"id":"img","name":"chart"}"#).await;
    let answer = success(&chart.outcome);
    assert!(answer.is_some_and(|answer| answer.call.wire() == "img" && !answer.is_error));
    let image = answer
        .into_iter()
        .flat_map(|answer| &answer.content)
        .find_map(|part| match part {
            ToolResultContent::Image(image) => Some(image),
            _ => None,
        });
    assert!(
        image.is_some_and(|image| image.media_type == Some(ImageMediaType::PNG)
            && matches!(image.data, DocumentSourceKind::Base64(_)))
    );

    // R6: 50 concurrent calls on one registry, each on its own task.
    let tasks: Vec<_> = (0..50)
        .map(|n| {
            let call = json!({"id": format!("c{n}"), "name": "add", "arguments": {"a": n, "b": 1}});
            tokio::spawn(run(&call.to_string()))
        })
        .collect();
    let results = futures::future::join_all(tasks).await;
    for (n, result) in (0..50).zip(results) {
        let json = result
            .map(|result| result_json(&result))
            .unwrap_or_default();
        assert_eq!(json, json!({"id": format!("c{n}"), "ok": n + 1}));
    }

    // R7: malformed input and random bytes give error lines, never a panic.
    let missing_name = ok(r#"{"id":"m","arguments":{}}"#).await;
    assert_eq!(
        missing_name.pointer("/error/kind"),
        Some(&json!("malformed"))
    );
    assert_eq!(missing_name.get("id"), Some(&json!("m")));
    let mut seed = 0x2545_f491_u32;
    let noise: Vec<u8> = (0..4096)
        .map(|_| {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            seed.to_le_bytes().first().copied().unwrap_or_default()
        })
        .collect();
    let (mut out, mut log) = (Vec::new(), Vec::new());
    let lines = run_lines(
        &tools,
        &context,
        TIMEOUT,
        noise.as_slice(),
        &mut out,
        &mut log,
    )
    .await;
    assert!(lines.is_ok());
    let printed = String::from_utf8_lossy(&out);
    assert!(printed.lines().count() > 0);
    for line in printed.lines() {
        let parsed: Value = serde_json::from_str(line).unwrap_or_default();
        assert_eq!(
            parsed.pointer("/error/kind"),
            Some(&json!("malformed")),
            "{line}"
        );
    }
}

/// The problems an invalid-arguments outcome names.
fn problems(outcome: &Outcome) -> Option<Vec<String>> {
    match outcome {
        Outcome::InvalidArguments { problems, .. } => {
            Some(problems.iter().map(ToString::to_string).collect())
        }
        _ => None,
    }
}

/// The tool result of a successful outcome.
fn success(outcome: &Outcome) -> Option<&rig_core::message::ToolResult> {
    match outcome {
        Outcome::Success(answer) => Some(answer),
        _ => None,
    }
}
