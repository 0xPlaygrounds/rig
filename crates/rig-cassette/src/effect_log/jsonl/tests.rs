use std::sync::Arc;

use rig_core::effect::{EffectKind, HandlerKey, Outcome};

use super::*;

fn record(id: u64, key: &str) -> EffectRecord {
    EffectRecord {
        stream_origin: None,
        tool_output: None,
        parent: None,
        scope: Some(Arc::from("agent")),
        id: EffectId::from_raw(id),
        key: HandlerKey::from(key),
        kind: EffectKind::Custom {
            kind: Arc::from("test"),
            payload: serde_json::json!({ "ask": id }),
        },
        outcome: Ok(Outcome::Custom {
            payload: serde_json::json!({ "n": id }),
        }),
        events: None,
    }
}

#[test]
fn appended_logs_read_back_as_one_with_merged_headers() {
    let dir = assert_fs::TempDir::new().expect("scratch directory");
    let path = dir.path().join("effects.jsonl");
    let mut writer = Writer::new(&path);
    writer
        .append(&EffectLog::from_records(vec![record(1, "model")]))
        .expect("first append");
    let second = EffectLog::from_records(vec![record(2, "model"), record(3, "tool")]);
    writer.append(&second).expect("second append");
    // Same header, no records: nothing to write.
    writer
        .append(&EffectLog {
            header: second.header.clone(),
            records: Vec::new(),
        })
        .expect("empty append");

    let text = std::fs::read_to_string(&path).expect("log text");
    assert_eq!(
        text.lines()
            .filter(|line| line.starts_with("{\"header\""))
            .count(),
        2
    );
    assert_eq!(text.lines().count(), 5);
    // The second header states only what the first does not: the new key.
    let headers: Vec<&str> = text
        .lines()
        .filter(|line| line.starts_with("{\"header\""))
        .collect();
    assert!(
        headers
            .last()
            .is_some_and(|line| line.contains("\"tool\"") && !line.contains("\"model\""))
    );

    let log = read(&path).expect("log");
    let ids: Vec<u64> = log
        .records
        .iter()
        .map(|record| record.id.as_u64())
        .collect();
    assert_eq!(ids, [1, 2, 3]);
    assert!(
        log.header
            .signature
            .contains_key(&HandlerKey::from("model"))
    );
    assert!(log.header.signature.contains_key(&HandlerKey::from("tool")));
    assert_eq!(last_id(&path).expect("tail"), Some(EffectId::from_raw(3)));

    // A new writer, such as after a restart, continues the file's header.
    Writer::new(&path)
        .append(&EffectLog {
            header: second.header.clone(),
            records: vec![record(4, "tool")],
        })
        .expect("append after a restart");
    let text = std::fs::read_to_string(&path).expect("log text");
    assert_eq!(text.lines().count(), 6, "no header line again: {text}");
}

#[test]
fn a_log_without_records_writes_no_file() {
    let dir = assert_fs::TempDir::new().expect("scratch directory");
    let path = dir.path().join("effects.jsonl");
    let mut writer = Writer::new(&path);
    let described = EffectLog::from_records(vec![record(1, "model")]);
    writer
        .append(&EffectLog {
            header: described.header.clone(),
            records: Vec::new(),
        })
        .expect("header only");
    assert!(!path.exists(), "a header alone is not written");
    // The first record brings the header it waited with.
    writer.append(&described).expect("first record");
    let text = std::fs::read_to_string(&path).expect("log text");
    assert_eq!(text.lines().count(), 2, "header and record: {text}");
    assert!(text.starts_with("{\"header\""));
}

#[test]
fn a_log_without_records_has_no_last_id() {
    let dir = assert_fs::TempDir::new().expect("scratch directory");
    let path = dir.path().join("effects.jsonl");
    std::fs::write(&path, "{\"header\":{}}\n").expect("header only");
    assert_eq!(last_id(&path).expect("tail"), None);
}

/// A completion record on `scope` whose request is `history`, as user
/// messages, answered with a raw document that echoes the request.
fn completion(id: u64, scope: &str, history: &[&str]) -> EffectRecord {
    completion_of(
        id,
        scope,
        history
            .iter()
            .map(|text| rig_core::completion::Message::user(*text))
            .collect(),
    )
}

/// The assistant message that repeats a [`completion_of`]'s reply in the
/// next request.
fn reply() -> rig_core::completion::Message {
    use rig_core::completion::{AssistantContent, AssistantMessage, Message};
    use rig_core::message::{Origin, StopReason};
    Message::Assistant(
        AssistantMessage::new(vec![AssistantContent::text("ok")])
            .with_origin(Origin::new("test.api", "model", ""))
            .with_stop(StopReason::ToolUse),
    )
}

/// A completion record on `scope` whose request is `history`, answered
/// "ok" with a raw document that echoes the request.
fn completion_of(
    id: u64,
    scope: &str,
    history: Vec<rig_core::completion::Message>,
) -> EffectRecord {
    use rig_core::completion::{AssistantContent, CompletionRequest, CompletionResponse, Usage};
    EffectRecord {
        stream_origin: None,
        tool_output: None,
        parent: None,
        scope: Some(Arc::from(scope)),
        id: EffectId::from_raw(id),
        key: HandlerKey::from("model"),
        kind: EffectKind::Completion {
            request: CompletionRequest::from(history),
            stream: true,
        },
        outcome: Ok(Outcome::Completion(CompletionResponse::new(
            vec![AssistantContent::text("ok")],
            Usage::default(),
            rig_core::message::Origin::new("test.api", "model", ""),
            serde_json::json!({
                "id": "resp_1",
                "instructions": "a long system prompt",
                "tools": [{"name": "read"}],
                "usage": {"input_tokens": 3, "attribution": {"items": [1, 2, 3]}},
            }),
        ))),
        events: None,
    }
}

/// The chat history of a completion record, as text.
fn history_of(record: &EffectRecord) -> Vec<String> {
    match &record.kind {
        EffectKind::Completion { request, .. } => request
            .chat_history
            .iter()
            .map(|message| serde_json::to_string(message).expect("message"))
            .collect(),
        _ => Vec::new(),
    }
}

#[test]
fn completion_requests_are_written_as_continuations_and_read_back_whole() {
    let dir = assert_fs::TempDir::new().expect("scratch directory");
    let path = dir.path().join("effects.jsonl");
    let written = vec![
        completion(1, "parent", &["a"]),
        completion(2, "child", &["x"]),
        completion(3, "parent", &["a", "b", "c"]),
        record(4, "tool"),
        completion(5, "parent", &["a", "b", "c", "d"]),
        // Compacted: shares nothing with the request before it.
        completion(6, "parent", &["summary"]),
        completion(7, "child", &["x", "y"]),
    ];
    let mut writer = Writer::new(&path);
    let (first, second) = written.split_at(3);
    writer
        .append(&EffectLog::from_records(first.to_vec()))
        .expect("first append");
    writer
        .append(&EffectLog::from_records(second.to_vec()))
        .expect("second append");

    let text = std::fs::read_to_string(&path).expect("log text");
    let continued: Vec<&str> = text
        .lines()
        .filter(|line| line.contains("\"after\":"))
        .collect();
    assert_eq!(continued.len(), 3, "records 3, 5 and 7 continue: {text}");
    assert!(
        continued
            .iter()
            .all(|line| line.starts_with("{\"id\":") && line.contains("\"same_tools\":true"))
    );
    assert!(!text.contains("a long system prompt"));
    assert!(!text.contains("attribution"));
    assert!(text.contains("\"input_tokens\":3"));

    let log = read(&path).expect("log");
    assert_eq!(log.records.len(), written.len());
    for (restored, original) in log.records.iter().zip(&written) {
        assert_eq!(restored.id, original.id);
        assert_eq!(history_of(restored), history_of(original));
        if let (
            EffectKind::Completion {
                request: restored, ..
            },
            EffectKind::Completion {
                request: original, ..
            },
        ) = (&restored.kind, &original.kind)
        {
            assert_eq!(restored.tools, original.tools);
        }
    }
    assert_eq!(last_id(&path).expect("tail"), Some(EffectId::from_raw(7)));

    // A new writer, such as after a restart, continues each chain.
    Writer::new(&path)
        .append(&EffectLog::from_records(vec![completion(
            8,
            "parent",
            &["summary", "e"],
        )]))
        .expect("append after a restart");
    let text = std::fs::read_to_string(&path).expect("log text");
    assert!(
        text.lines()
            .last()
            .is_some_and(|line| line.starts_with("{\"id\":8,\"after\":6,\"keep\":1,")),
        "{text}"
    );
    let log = read(&path).expect("log after a restart");
    assert_eq!(
        log.records.last().map(history_of),
        Some(history_of(&completion(8, "parent", &["summary", "e"])))
    );
}

#[test]
fn replies_and_tool_results_are_stored_once_and_a_compacted_request_keeps_its_tail() {
    use rig_core::completion::Message;
    use rig_core::message::{CallId, ProviderCallId, ToolName};
    use rig_core::tool::{ToolOutput, ToolResult};
    let dir = assert_fs::TempDir::new().expect("scratch directory");
    let path = dir.path().join("effects.jsonl");
    let system = Message::system("a long system prompt for the agent");
    let first = Message::user("first task, with enough words to be worth referring to");
    let listing = "Cargo.toml src target, the listing the shell tool returned";
    let second = Message::tool_result(
        CallId::from(ProviderCallId::new("call_1").expect("call id")),
        ToolName::new("shell").expect("tool name"),
        listing,
    );
    let mut shell = record(10, "tool:shell");
    shell.kind = EffectKind::ToolCall {
        name: "shell".into(),
        args: "{}".into(),
    };
    shell.outcome = Ok(Outcome::ToolResult {
        result: ToolResult::success(ToolOutput::text(listing)),
    });
    let written = vec![
        completion_of(1, "parent", vec![system.clone(), first.clone()]),
        shell,
        completion_of(
            2,
            "parent",
            vec![system.clone(), first.clone(), reply(), second.clone()],
        ),
        // The compaction's summarizing request, on the same chain.
        completion_of(
            3,
            "parent",
            vec![
                Message::system("summarize"),
                Message::user("the conversation so far"),
            ],
        ),
        // Compacted: the summary in place of the first task, the tail kept.
        completion_of(
            4,
            "parent",
            vec![
                system.clone(),
                Message::user("summary"),
                reply(),
                second.clone(),
            ],
        ),
    ];
    let mut writer = Writer::new(&path);
    writer
        .append(&EffectLog::from_records(written.clone()))
        .expect("append");

    let text = std::fs::read_to_string(&path).expect("log text");
    let lines: Vec<&str> = text.lines().skip(1).collect();
    assert_eq!(lines.len(), 5, "{text}");
    assert!(
        lines[2].contains("\"after\":1,\"keep\":2,\"then\":[{\"reply\":1},{\"results\":[10]}]"),
        "{text}"
    );
    assert!(!lines[3].contains("\"after\""), "{text}");
    assert!(
        lines[4].contains("\"after\":2,\"keep\":1,\"then\":[{\"new\":1},{\"base\":[2,4]}]"),
        "{text}"
    );
    // Each reply once: in the four outcomes, never in a request; the tool
    // result in its call's outcome only.
    assert_eq!(text.matches("\"text\":\"ok\"").count(), 4, "{text}");
    assert_eq!(text.matches(listing).count(), 1, "{text}");

    let log = read(&path).expect("log");
    assert_eq!(log.records.len(), written.len());
    for (restored, original) in log.records.iter().zip(&written) {
        assert_eq!(history_of(restored), history_of(original));
    }
}

#[test]
fn a_tool_result_whose_write_failed_is_written_whole_when_repeated() {
    use rig_core::completion::Message;
    use rig_core::message::{CallId, ProviderCallId, ToolName};
    use rig_core::tool::{ToolOutput, ToolResult};
    let dir = assert_fs::TempDir::new().expect("scratch directory");
    let path = dir.path().join("effects.jsonl");
    let listing = "Cargo.toml src target, the listing the shell tool returned";
    let mut shell = record(10, "tool:shell");
    shell.kind = EffectKind::ToolCall {
        name: "shell".into(),
        args: "{}".into(),
    };
    shell.outcome = Ok(Outcome::ToolResult {
        result: ToolResult::success(ToolOutput::text(listing)),
    });
    let mut writer = Writer::new(&path);
    // A directory in the file's place: the tool call never reaches the file.
    std::fs::create_dir(&path).expect("directory in the way");
    writer
        .append(&EffectLog::from_records(vec![shell]))
        .expect_err("the write fails");
    std::fs::remove_dir(&path).expect("directory removed");

    let task = Message::user("first task, with enough words to be worth referring to");
    let result = Message::tool_result(
        CallId::from(ProviderCallId::new("call_1").expect("call id")),
        ToolName::new("shell").expect("tool name"),
        listing,
    );
    writer
        .append(&EffectLog::from_records(vec![
            completion_of(1, "parent", vec![task.clone()]),
            completion_of(2, "parent", vec![task, reply(), result]),
        ]))
        .expect("append");
    let log = read(&path).expect("the log reads back");
    assert_eq!(log.records.len(), 2);
}

#[test]
fn a_continuation_of_an_unknown_request_is_an_error() {
    let dir = assert_fs::TempDir::new().expect("scratch directory");
    let path = dir.path().join("effects.jsonl");
    let mut writer = Writer::new(&path);
    writer
        .append(&EffectLog::from_records(vec![
            completion(1, "parent", &["a"]),
            completion(2, "parent", &["a", "b"]),
        ]))
        .expect("append");
    let text = std::fs::read_to_string(&path).expect("log text");
    let without_base: String = text
        .lines()
        .filter(|line| !line.starts_with("{\"tool_output\""))
        .map(|line| format!("{line}\n"))
        .collect();
    std::fs::write(&path, without_base).expect("rewrite");
    let failure = read(&path).expect_err("the base is missing");
    assert_eq!(failure.kind(), std::io::ErrorKind::InvalidData);
}

#[test]
fn last_id_reads_ids_off_long_lines_and_finds_the_highest_of_the_tail() {
    let dir = assert_fs::TempDir::new().expect("scratch directory");
    let path = dir.path().join("effects.jsonl");
    let long = "m".repeat(300 * 1024);
    let mut writer = Writer::new(&path);
    // Resolved out of order: the highest id is not on the last line.
    writer
        .append(&EffectLog::from_records(vec![
            completion(9, "parent", &[long.as_str()]),
            record(4, "tool"),
            completion(3, "child", &[long.as_str()]),
        ]))
        .expect("append");
    assert_eq!(last_id(&path).expect("tail"), Some(EffectId::from_raw(9)));
    assert_eq!(line_id(b"{\"header\":{}}"), None);
    assert_eq!(
        line_id(b"{\"tool_output\":{\"a\":1},\"id\":12,\"key\":\"k\"}"),
        Some(EffectId::from_raw(12))
    );
}
