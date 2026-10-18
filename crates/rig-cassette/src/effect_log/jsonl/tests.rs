use std::collections::HashMap;
use std::sync::Arc;

use rig_core::effect::{Answered, EffectKind, HandlerKey};

use super::*;

/// A writer appending to the file at `path`.
fn writer_at(path: &std::path::Path) -> Writer {
    let dir = path.parent().expect("a folder");
    let name = path.file_name().expect("a name").to_string_lossy();
    Writer::on(Arc::new(rig_memory::DirFiles::new(dir)), name)
}

/// The log in the file at `path`.
fn read(path: &std::path::Path) -> io::Result<EffectLog> {
    read_from(std::fs::read(path)?.as_slice())
}

/// `Writer::append` of a whole log.
trait AppendLog {
    fn append_log(&mut self, log: &EffectLog) -> io::Result<()>;
}

impl AppendLog for Writer {
    fn append_log(&mut self, log: &EffectLog) -> io::Result<()> {
        self.append(&log.header, &log.records)
    }
}

fn record(id: u64, key: &str) -> EffectRecord {
    EffectRecord {
        purpose: None,
        turn: None,
        started_ms: None,
        finished_ms: None,
        reviewed: None,
        disposition: Default::default(),
        decided_by: None,
        stream_origin: None,
        published: None,
        parent: None,
        scope: Some(Arc::from("agent")),
        id: EffectId::from_raw(id),
        key: HandlerKey::from(key),
        kind: EffectKind::Custom {
            kind: Arc::from("test"),
            payload: serde_json::json!({ "ask": id }),
        },
        outcome: Ok(Answered::Custom {
            payload: serde_json::json!({ "n": id }),
        }),
        events: None,
    }
}

#[test]
fn appended_logs_read_back_as_one_with_merged_headers() {
    let dir = assert_fs::TempDir::new().expect("scratch directory");
    let path = dir.path().join("effects.jsonl");
    let mut writer = writer_at(&path);
    let first = EffectLog::from_records(vec![record(1, "model")]);
    // A header without records: nothing to write.
    writer
        .append_log(&EffectLog {
            records: Vec::new(),
            ..first.clone()
        })
        .expect("header only");
    assert!(!path.exists(), "a header alone is not written");
    writer.append_log(&first).expect("first append");
    let second = EffectLog::from_records(vec![record(2, "model"), record(3, "tool")]);
    writer.append_log(&second).expect("second append");

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

    // A new writer, such as after a restart, continues the file's header
    // and ids.
    let mut writer = writer_at(&path);
    assert_eq!(writer.last_id().expect("ids"), Some(EffectId::from_raw(3)));
    writer
        .append_log(&EffectLog {
            header: second.header.clone(),
            records: vec![record(4, "tool")],
        })
        .expect("append after a restart");
    let text = std::fs::read_to_string(&path).expect("log text");
    assert_eq!(text.lines().count(), 6, "no header line again: {text}");
}

#[test]
fn program_identities_are_stated_once_and_read_back_merged() {
    use crate::effect_log::ProgramIdentity;
    let dir = assert_fs::TempDir::new().expect("scratch directory");
    let path = dir.path().join("effects.jsonl");
    let mut writer = writer_at(&path);
    let program = |policy| ProgramIdentity {
        required: rig_core::effect::EffectRow::new(),
        policy,
    };
    let mut first = EffectLog::from_records(vec![record(1, "model")]);
    first.header.programs.insert("one".into(), program(1));
    writer.append_log(&first).expect("first append");
    let mut second = EffectLog::from_records(vec![record(2, "model")]);
    second.header.programs = first.header.programs.clone();
    second.header.programs.insert("two".into(), program(2));
    writer.append_log(&second).expect("second append");

    let text = std::fs::read_to_string(&path).expect("log text");
    let last = text
        .lines()
        .rfind(|line| line.starts_with("{\"header\""))
        .expect("a second header");
    assert!(
        last.contains("\"two\"") && !last.contains("\"one\""),
        "{last}"
    );
    assert_eq!(
        read(&path).expect("log").header.programs,
        second.header.programs
    );
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
        purpose: None,
        turn: None,
        started_ms: None,
        finished_ms: None,
        reviewed: None,
        disposition: Default::default(),
        decided_by: None,
        stream_origin: None,
        published: None,
        parent: None,
        scope: Some(Arc::from(scope)),
        id: EffectId::from_raw(id),
        key: HandlerKey::from("model"),
        kind: EffectKind::Completion {
            request: CompletionRequest::from(history).into(),
            stream: true,
        },
        outcome: Ok(Answered::Completion(
            CompletionResponse::new(
                vec![AssistantContent::text("ok")],
                Usage::default(),
                rig_core::message::Origin::new("test.api", "model", ""),
                serde_json::json!({
                    "id": "resp_1",
                    "instructions": "a long system prompt",
                    "tools": [{"name": "read"}],
                    "usage": {"input_tokens": 3, "attribution": {"items": [1, 2, 3]}},
                }),
            )
            .into(),
        )),
        events: None,
    }
}

/// The listing the shell tool returned, long enough to be worth referring to.
const LISTING: &str = "Cargo.toml src target, the listing the shell tool returned";

/// A shell call that returned [`LISTING`], and the tool-result message that
/// carries it.
fn shell_call() -> (EffectRecord, rig_core::completion::Message) {
    use rig_core::message::{CallId, ProviderCallId, ToolName};
    use rig_core::tool::{ToolOutput, ToolResult};
    let mut shell = record(10, "tool:shell");
    shell.kind = EffectKind::ToolCall {
        name: "shell".into(),
        args: "{}".into(),
    };
    shell.outcome = Ok(Answered::ToolResult {
        result: ToolResult::success(ToolOutput::text(LISTING)),
    });
    let result = rig_core::completion::Message::tool_result(
        CallId::from(ProviderCallId::new("call_1").expect("call id")),
        ToolName::new("shell").expect("tool name"),
        LISTING,
    );
    (shell, result)
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
    let mut writer = writer_at(&path);
    let (first, second) = written.split_at(3);
    writer
        .append_log(&EffectLog::from_records(first.to_vec()))
        .expect("first append");
    writer
        .append_log(&EffectLog::from_records(second.to_vec()))
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

    // A new writer, such as after a restart, continues each chain.
    let mut writer = writer_at(&path);
    assert_eq!(writer.last_id().expect("ids"), Some(EffectId::from_raw(7)));
    writer
        .append_log(&EffectLog::from_records(vec![completion(
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
    let dir = assert_fs::TempDir::new().expect("scratch directory");
    let path = dir.path().join("effects.jsonl");
    let system = Message::system("a long system prompt for the agent");
    let first = Message::user("first task, with enough words to be worth referring to");
    let (shell, second) = shell_call();
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
    let mut writer = writer_at(&path);
    writer
        .append_log(&EffectLog::from_records(written.clone()))
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
    assert_eq!(text.matches(LISTING).count(), 1, "{text}");

    let log = read(&path).expect("log");
    assert_eq!(log.records.len(), written.len());
    for (restored, original) in log.records.iter().zip(&written) {
        assert_eq!(history_of(restored), history_of(original));
    }

    // Read for what each effect was and how it ended: every record, with
    // no request rebuilt.
    let bytes = std::fs::read(&path).expect("log bytes");
    let outcomes = read_outcomes_from(bytes.as_slice()).expect("outcomes");
    assert_eq!(outcomes.len(), log.records.len());
    let json = |record: &EffectRecord| serde_json::to_value(&record.outcome).expect("outcome");
    for (read, whole) in outcomes.iter().zip(&log.records) {
        assert_eq!((read.id, json(read)), (whole.id, json(whole)));
        assert!(history_of(read).is_empty());
    }
}

#[test]
fn a_tool_result_whose_write_failed_is_written_whole_when_repeated() {
    use rig_core::completion::Message;
    let dir = assert_fs::TempDir::new().expect("scratch directory");
    let path = dir.path().join("effects.jsonl");
    let (shell, result) = shell_call();
    let mut writer = writer_at(&path);
    // A directory in the file's place: the tool call never reaches the file.
    std::fs::create_dir(&path).expect("directory in the way");
    writer
        .append_log(&EffectLog::from_records(vec![shell]))
        .expect_err("the write fails");
    std::fs::remove_dir(&path).expect("directory removed");

    let task = Message::user("first task, with enough words to be worth referring to");
    writer
        .append_log(&EffectLog::from_records(vec![
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
    let mut writer = writer_at(&path);
    writer
        .append_log(&EffectLog::from_records(vec![
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
fn a_log_that_cannot_be_read_back_is_continued_after_every_id_it_states() {
    let dir = assert_fs::TempDir::new().expect("scratch directory");
    let path = dir.path().join("effects.jsonl");
    // Resolved out of order: the highest id is not on the last line.
    writer_at(&path)
        .append_log(&EffectLog::from_records(vec![
            completion(9, "parent", &["a"]),
            record(4, "tool"),
        ]))
        .expect("append");
    let mut text = std::fs::read_to_string(&path).expect("log text");
    // An odd line, a whole record after it, and a line torn by a crash.
    text.push_str("not a record\n");
    text.push_str(&serde_json::to_string(&record(12, "tool")).expect("record"));
    text.push_str("\n{\"id\":15,\"key\":\"to");
    std::fs::write(&path, text).expect("rewrite");

    let mut writer = writer_at(&path);
    assert_eq!(writer.last_id().expect("ids"), Some(EffectId::from_raw(12)));
    writer
        .append_log(&EffectLog::from_records(vec![completion(
            16,
            "parent",
            &["a", "b"],
        )]))
        .expect("append after the torn line");
    let text = std::fs::read_to_string(&path).expect("log text");
    let last = text.lines().last().unwrap_or_default();
    assert!(
        last.contains("\"id\":16") && !last.contains("\"after\""),
        "a whole record: {text}"
    );
}

/// The golden effect logs, each with every completion request's messages
/// shared with the earlier requests that hold an equal message, as a
/// conversation's requests share them.
fn golden_logs_sharing_messages() -> Vec<(String, EffectLog)> {
    use rig_core::completion::Message;
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/effects");
    let mut files: Vec<_> = std::fs::read_dir(&root)
        .expect("the golden logs")
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.to_string_lossy().ends_with(".effects.json"))
        .collect();
    files.sort();
    assert!(!files.is_empty(), "no golden logs under {}", root.display());
    files
        .into_iter()
        .map(|file| {
            let text = std::fs::read_to_string(&file).expect("a golden log");
            let mut log: EffectLog = serde_json::from_str(&text).expect("a golden log parses");
            let mut seen: HashMap<String, Arc<Message>> = HashMap::new();
            for record in &mut log.records {
                if let EffectKind::Completion { request, .. } = &mut record.kind {
                    for message in &mut Arc::make_mut(request).chat_history {
                        let text = serde_json::to_string(&**message).expect("a message");
                        *message = Arc::clone(seen.entry(text).or_insert_with(|| message.clone()));
                    }
                }
            }
            (file.display().to_string(), log)
        })
        .collect()
}

/// A request's messages the writer already holds take the text it noted
/// for them, so that text must be each message's own serialization: then
/// sharing messages leaves every byte written as it is without sharing.
#[test]
fn shared_messages_are_written_as_unshared_ones() {
    for (name, log) in golden_logs_sharing_messages() {
        let shared = rig_memory::MemoryStore::default();
        let mut writer = Writer::on(Arc::new(shared.clone()), SESSION_EFFECT_LOG);
        for record in &log.records {
            writer.append(&log.header, [record]).expect("append");
            let EffectKind::Completion { request, .. } = &record.kind else {
                continue;
            };
            let head = writer
                .known
                .as_ref()
                .and_then(|known| known.head(record.id));
            let noted: Vec<&str> = head
                .map(|head| head.history.iter().map(|text| &*text.text).collect())
                .unwrap_or_default();
            let fresh: Vec<String> = request
                .chat_history
                .iter()
                .map(|message| {
                    serde_json::to_value(&**message)
                        .expect("a message")
                        .to_string()
                })
                .collect();
            assert_eq!(noted, fresh, "{name}: record {}", record.id);
        }

        let unshared: EffectLog =
            serde_json::from_value(serde_json::to_value(&log).expect("the log")).expect("a copy");
        let alone = rig_memory::MemoryStore::default();
        let mut writer = Writer::on(Arc::new(alone.clone()), SESSION_EFFECT_LOG);
        for record in &unshared.records {
            writer.append(&unshared.header, [record]).expect("append");
        }
        assert_eq!(
            shared.read_file(SESSION_EFFECT_LOG).ok(),
            alone.read_file(SESSION_EFFECT_LOG).ok(),
            "{name}"
        );
    }
}

/// A key in a request never reaches the file, and a continuation of that
/// request still reads back whole, the key redacted in both.
#[test]
fn a_key_in_a_request_never_reaches_the_file() {
    let dir = assert_fs::TempDir::new().expect("scratch directory");
    let path = dir.path().join("effects.jsonl");
    let said = "run env:\nOPENAI_API_KEY=sk-test-probe-1234567890abcdef";
    let mut writer = writer_at(&path);
    let written = [
        completion(1, "agent", &[said]),
        completion(2, "agent", &[said, "and then?"]),
    ];
    writer
        .append(&LogHeader::default(), &written)
        .expect("appended");
    let text = std::fs::read_to_string(&path).expect("the file");
    assert!(!text.contains("sk-test-probe"), "{text}");
    let log = read(&path).expect("read back");
    let histories: Vec<Vec<String>> = log.records.iter().map(history_of).collect();
    assert_eq!(histories.len(), 2);
    for history in &histories {
        let first = history.first().expect("a message");
        assert!(first.contains("OPENAI_API_KEY=[REDACTED]"), "{first}");
    }
    assert_eq!(histories.get(1).map(Vec::len), Some(2));
}
