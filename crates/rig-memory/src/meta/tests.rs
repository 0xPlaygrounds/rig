use std::collections::BTreeMap;

use rig_core::completion::Message;
use rig_core::conversation::Conversation;
use rig_core::message::{AssistantContent, AssistantMessage, CallId, ToolName};
use serde_json::json;

use crate::{Files, MemoryStore, SessionMeta, excerpt};

/// A listing written by an older rig, with a field this one does not know
/// (a name), reads with it skipped.
#[test]
fn a_listing_reads_back_and_unknown_fields_are_skipped() {
    let store = MemoryStore::default();
    assert_eq!(SessionMeta::read(&store), None);
    let old = br#"{"name":"release","title":"fix the parser","labels":{"spent":"$1"},"updated":7}"#;
    assert!(store.write_file("meta.json", old).is_ok());
    let meta = SessionMeta {
        title: "fix the parser".to_owned(),
        labels: BTreeMap::from([("spent".to_owned(), "$1".to_owned())]),
        updated: 7,
        ..SessionMeta::default()
    };
    assert_eq!(SessionMeta::read(&store).as_ref(), Some(&meta));
    assert!(SessionMeta::default().write(&store).is_ok());
    assert_eq!(SessionMeta::read(&store), Some(SessionMeta::default()));
}

/// A picker's line for each kind of message says who wrote it and the
/// start of its text: the tools an assistant called when it said nothing,
/// and nothing for the user's tool results.
#[test]
fn an_excerpt_says_who_wrote_a_message_and_how_it_starts() {
    let shell = ToolName::new("shell").expect("a tool name");
    let read = ToolName::new("read").expect("a tool name");
    let calls = vec![
        AssistantContent::tool_call("c1", shell.clone(), json!({})),
        AssistantContent::tool_call("c2", read, json!({})),
    ];
    let conversation = Conversation::from_messages([
        Message::system("be brief"),
        Message::user(format!("fix\n  the parser {}", "x".repeat(80))),
        Message::Assistant(AssistantMessage::new(calls)),
        Message::tool_result(CallId::from_wire("c1"), shell, "a.rs"),
        Message::assistant("done"),
    ]);
    let lines: Vec<String> = conversation
        .entries()
        .map(|entry| excerpt(&entry))
        .collect();
    let cut = format!("user: fix the parser {}…", "x".repeat(44));
    assert_eq!(
        lines,
        [
            "system: be brief",
            cut.as_str(),
            "calls: shell, read",
            "tool results:",
            "assistant: done",
        ]
    );
}
