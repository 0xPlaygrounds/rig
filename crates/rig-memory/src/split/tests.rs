use std::collections::BTreeMap;

use rig_core::completion::Message;
use rig_core::conversation::{
    Aside, AsideId, Author, Conversation, MessageId, MessageKey, pairing::pending_calls,
};
use rig_core::message::{AssistantContent, AssistantMessage, CallId, ToolName};
use serde_json::json;

use crate::{
    Files, Line, LineId, MemoryStore, SPLIT, SavedAgent, Session, SessionMeta, SplitOf,
    SplitPointer, Store, lines, split,
};

fn keys(conversation: &Conversation) -> Vec<MessageKey> {
    conversation.entries().map(|entry| entry.key()).collect()
}

/// A session of one agent, `main`, whose line parts into a kept branch
/// `side` after its second entry, with a rating on its first, a saved
/// value, a session value, a listing with a listed value and a plugin
/// file.
fn saved() -> (MemoryStore, Conversation, AsideId) {
    let by = Author::plugin("tree");
    let mut conversation = Conversation::from_messages([
        Message::user("one"),
        Message::assistant("two"),
        Message::user("three"),
        Message::assistant("four"),
    ]);
    let first = conversation.id(0);
    conversation.add_aside(Aside::new("rating", by.clone(), json!(5)).after(first));
    let main = conversation.branch_off(2, "side", by.clone());
    conversation.push(Message::user("three, again"));
    conversation.set_line_value("k", Some(json!("side")), by.clone());
    let side = conversation.branches().first().map(|kept| kept.id);
    assert_eq!(side, main.ok());
    let Some(side) = side else {
        return (MemoryStore::default(), conversation, AsideId::default());
    };
    // Main is active again; `side` is the kept branch.
    assert!(conversation.switch_branch(side, by).is_ok());
    let side = conversation.branches().first().map(|kept| kept.id);
    let store = MemoryStore::default();
    let mut session = Session {
        agents: vec![SavedAgent {
            id: "main".to_owned(),
            conversation: conversation.clone(),
            saved: BTreeMap::from([("model".to_owned(), json!("m"))]),
            ..SavedAgent::default()
        }],
        saved: BTreeMap::from([("theme".to_owned(), json!("dark"))]),
        repairs: Vec::new(),
    };
    assert!(session.save(&store).is_ok());
    assert!(store.write_file("plugins/notes.md", b"kept").is_ok());
    let listing = SessionMeta {
        title: "one".to_owned(),
        messages: 4,
        listed: BTreeMap::from([("goals.goal".to_owned(), json!("ship"))]),
        ..SessionMeta::default()
    };
    assert!(listing.write(&store).is_ok());
    let read = Session::read(&store).ok();
    let read = read.and_then(|session| session.agents.into_iter().next());
    let conversation = read.map(|agent| agent.conversation).unwrap_or_default();
    (store, conversation, side.unwrap_or_default())
}

/// A split of a kept branch through its last entry is a session of the
/// one agent holding that line: its entries' keys, the asides that follow
/// them but not the branch on the way to it, its title and line values,
/// the agent's and the session's saved values, the plugin files, and the
/// listing with its listed values, so a picker finds the split before it
/// ever runs. One through an earlier entry stops there, without the line
/// values.
#[test]
fn a_split_holds_the_line_with_its_keys_asides_and_values() {
    let (store, source, side) = saved();
    let line = source.line(Some(side)).unwrap_or_default();
    let copy = MemoryStore::default();
    let of = SplitOf {
        agent: "main",
        line: Some(side),
        through: None,
    };
    let written = split(&store, &copy, &of);
    assert!(written.is_ok(), "{written:?}");
    let read = Session::read(&copy);
    assert!(read.is_ok(), "{read:?}");
    let Ok(read) = read else { return };
    let ids: Vec<_> = read.agents.iter().map(|agent| agent.id.as_str()).collect();
    assert_eq!(ids, ["main"]);
    let Some(agent) = read.agents.first() else {
        return;
    };
    let conversation = &agent.conversation;
    let expected: Vec<MessageKey> = line
        .iter()
        .filter_map(|entry| {
            Some(MessageKey {
                id: entry.id?,
                made: entry.made?,
            })
        })
        .collect();
    assert_eq!(keys(conversation), expected);
    assert_eq!(conversation.line_title(), "side");
    let kinds: Vec<_> = conversation
        .asides()
        .map(|(_, aside)| aside.kind.as_str())
        .filter(|kind| !kind.starts_with("rig."))
        .collect();
    assert_eq!(kinds, ["rating"]);
    assert!(conversation.branches().is_empty(), "no branch on the way");
    let value = conversation
        .line_values()
        .and_then(|values| values.get("k"));
    assert_eq!(value, Some(&json!("side")));
    assert_eq!(agent.saved.get("model"), Some(&json!("m")));
    assert_eq!(read.saved.get("theme"), Some(&json!("dark")));
    assert_eq!(
        copy.read_file("plugins/notes.md").ok(),
        Some(b"kept".to_vec())
    );
    let listing = SessionMeta::read(&copy).unwrap_or_default();
    assert_eq!(listing.title, "one");
    assert_eq!(listing.messages, line.len());
    assert_eq!(listing.listed.get("goals.goal"), Some(&json!("ship")));

    let second = line.get(1).and_then(|entry| entry.id);
    let short = MemoryStore::default();
    let of = SplitOf {
        through: second,
        ..of
    };
    let written = split(&store, &short, &of);
    assert_eq!(
        written.as_ref().map(|copy| copy.messages().len()).ok(),
        Some(2)
    );
    let listing = SessionMeta::read(&short).unwrap_or_default();
    assert_eq!(listing.messages, 2);
    assert_eq!(
        written.ok().and_then(|copy| copy.line_values().cloned()),
        None
    );
}

/// Nothing is written for a message that is not on the line, nor for a
/// point that parts a call from its result.
#[test]
fn a_split_off_the_line_or_inside_a_call_writes_nothing() {
    let (store, _, side) = saved();
    let copy = MemoryStore::default();
    let off = SplitOf {
        agent: "main",
        line: Some(side),
        through: "m99".parse::<MessageId>().ok(),
    };
    let none = SplitOf {
        agent: "nobody",
        line: None,
        through: None,
    };
    assert!(split(&store, &copy, &none).is_err());
    let shell = ToolName::new("shell").expect("a tool name");
    let call = AssistantContent::tool_call("c1", shell.clone(), json!({}));
    let mut conversation = Conversation::from_messages([
        Message::user("list"),
        Message::Assistant(AssistantMessage::new(vec![call])),
        Message::tool_result(CallId::from_wire("c1"), shell, "a.rs"),
    ]);
    assert!(!pending_calls(conversation.messages().get(..2).unwrap_or_default()).is_empty());
    let calling = MemoryStore::default();
    assert!(crate::save(&calling, "main", &mut conversation).is_ok());
    let inside = SplitOf {
        agent: "main",
        line: None,
        through: conversation.id(1),
    };
    assert!(split(&calling, &copy, &inside).is_err());
    assert!(split(&store, &copy, &off).is_err());
    assert_eq!(copy.logs().ok(), Some(Vec::new()), "nothing written");
}

/// The lines of a conversation are a tree in display order: the active
/// one, then each line kept in it by where it parts, each followed by the
/// lines kept in it, splits kept in other sessions among them.
#[test]
fn lines_list_branches_and_splits_as_a_tree() {
    let by = Author::plugin("tree");
    let (_, mut conversation, side) = saved();
    let pointer = |title: &str| SplitPointer {
        session: "1-2".to_owned(),
        agent: "main".to_owned(),
        title: title.to_owned(),
        moved: 2,
        merged: false,
    };
    let value = |title| serde_json::to_value(pointer(title)).unwrap_or_default();
    let first = conversation.id(0);
    conversation.add_aside(Aside::new(SPLIT, by.clone(), value("out")).after(first));
    let side_third = conversation
        .line(Some(side))
        .and_then(|line| line.get(2).and_then(|entry| entry.id));
    let inner = Aside::new(SPLIT, by, value("deeper")).after(side_third);
    assert!(conversation.add_aside_in(Some(side), inner).is_ok());
    let side = conversation.branches().first().map(|kept| kept.id);
    let listed = lines(&conversation);
    let shape: Vec<_> = listed
        .iter()
        .map(|line: &Line| (line.title.as_str(), line.inside, line.at, line.entries))
        .collect();
    assert_eq!(
        shape,
        [
            ("main", None, 0, 4),
            ("out", Some(0), 1, 2),
            ("side", Some(0), 2, 1),
            ("deeper", Some(2), 3, 2),
        ]
    );
    let ids: Vec<_> = listed.iter().map(|line| line.id.clone()).collect();
    let session = LineId::Split {
        session: "1-2".to_owned(),
        agent: "main".to_owned(),
    };
    let side = side.map(LineId::Branch).unwrap_or(LineId::Active);
    assert_eq!(ids, [LineId::Active, session.clone(), side, session]);
}
