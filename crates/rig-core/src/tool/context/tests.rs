use super::*;

#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct Counter(u32);
impl ContextValue for Counter {
    const KEY: &'static str = "test.counter";
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct RequestId(String);
impl ContextValue for RequestId {
    const KEY: &'static str = "test.request_id";
}

// Two shapes that deliberately share a key: the fixture for "the slot holds
// something that is not a `T`".
#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct A(u32);
impl ContextValue for A {
    const KEY: &'static str = "k";
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct B {
    name: String,
}
impl ContextValue for B {
    const KEY: &'static str = "k";
}

#[test]
fn context_separates_inbound_and_result_values() {
    let mut context = ToolContext::new();
    context.insert(Counter(42)).unwrap();
    context
        .insert_result(RequestId("request-1".to_string()))
        .unwrap();
    assert_eq!(context.get::<Counter>().unwrap(), Some(Counter(42)));
    assert_eq!(
        context.result::<RequestId>().unwrap(),
        Some(RequestId("request-1".to_string()))
    );

    let next = context.for_dispatch();
    assert_eq!(next.get::<Counter>().unwrap(), Some(Counter(42)));
    assert_eq!(next.result::<RequestId>().unwrap(), None);
}

#[test]
fn context_round_trips_through_serde() {
    #[derive(Serialize, Deserialize, Debug, PartialEq)]
    struct Session {
        id: String,
    }
    impl ContextValue for Session {
        const KEY: &'static str = "test.session";
    }
    #[derive(Serialize, Deserialize, Debug, PartialEq)]
    struct Seq(u64);
    impl ContextValue for Seq {
        const KEY: &'static str = "test.seq";
    }
    let mut context = ToolContext::new();
    context
        .insert(Session {
            id: "abc".to_string(),
        })
        .unwrap();
    context.insert_result(Seq(7)).unwrap();

    let json = serde_json::to_string(&context).unwrap();
    let back: ToolContext = serde_json::from_str(&json).unwrap();
    assert_eq!(back, context);
    assert_eq!(
        back.get::<Session>().unwrap(),
        Some(Session {
            id: "abc".to_string()
        })
    );
    assert_eq!(back.result::<Seq>().unwrap(), Some(Seq(7)));

    let empty: ToolContext = serde_json::from_str("{}").unwrap();
    assert!(empty.is_empty());
    assert_eq!(serde_json::to_string(&ToolContext::new()).unwrap(), "{}");
}

#[test]
fn decode_failure_keeps_the_serde_error_as_its_source() {
    let mut context = ToolContext::new();
    context
        .inbound
        .insert(Counter::KEY.to_string(), serde_json::json!("not a number"));
    let error = context.require::<Counter>().unwrap_err();
    let source = std::error::Error::source(&error)
        .and_then(|source| source.downcast_ref::<serde_json::Error>())
        .expect("decode failure should expose the serde error");
    assert!(source.is_data());
}

#[test]
fn insert_replaces_an_undecodable_displaced_value_and_returns_none() {
    let mut context = ToolContext::new();
    context.insert(A(1)).unwrap();
    assert_eq!(
        context
            .insert(B {
                name: "b".to_string()
            })
            .unwrap(),
        None
    );
    assert_eq!(
        context.get::<B>().unwrap(),
        Some(B {
            name: "b".to_string()
        })
    );
}

/// What a conversation keeps of a call's facts: secrets redacted, no old
/// file text, and past the budget the largest values left out while a
/// small one stays.
#[test]
fn kept_facts_are_redacted_without_old_text_and_bounded() {
    use crate::tool::{
        ExitStatus, FileChange, FileRead, FilesChanged, FilesRead, Hunk, HunkLine, Lines,
        content_hash,
    };

    let mut context = ToolContext::new();
    let secret = "sk-ant-api03-abcdefghijklmnopqrstuvwxyz";
    let read = FileRead::new(format!("/work/key {secret}"), content_hash(b""));
    let hunk = Hunk {
        old: Lines { start: 1, count: 1 },
        new: Lines { start: 1, count: 1 },
        lines: vec![
            HunkLine::Removed(format!("{secret}\n")),
            HunkLine::Added("removed\n".to_owned()),
        ],
    };
    let change = FileChange::modified("a.rs", 1, 1)
        .with_hunks(vec![hunk])
        .with_before_text("x".repeat(1024 * 1024));
    let inserted = [
        context.insert_result(ExitStatus::Exited(3)).is_ok(),
        context.insert_result(FilesRead(vec![read])).is_ok(),
        context.insert_result(FilesChanged(vec![change])).is_ok(),
    ];
    assert_eq!(inserted, [true; 3]);
    let kept = context.result_context().kept();
    let paths = kept.get::<FilesRead>().ok().flatten().unwrap_or_default();
    assert_eq!(paths.paths().collect::<Vec<_>>(), ["/work/key [REDACTED]"]);
    let changed = kept
        .get::<FilesChanged>()
        .ok()
        .flatten()
        .unwrap_or_default();
    let old = changed.0.first().map(|change| change.before_text.is_none());
    assert_eq!(old, Some(true));
    let removed = changed
        .0
        .first()
        .map(FileChange::unified)
        .unwrap_or_default();
    assert!(
        removed.contains("-[REDACTED]\n") && !removed.contains(secret),
        "{removed}"
    );

    // A key block is in no one line, whichever its line ends: the change
    // keeps no hunks.
    for end in ["\n", "\r\n", "\r"] {
        let block = [
            "-----BEGIN RSA PRIVATE KEY-----",
            "MIIEowIBAAKCAQEAabc",
            "-----END RSA PRIVATE KEY-----",
        ];
        let hunk = Hunk {
            old: Lines { start: 1, count: 0 },
            new: Lines { start: 1, count: 3 },
            lines: block
                .map(|line| HunkLine::Added(format!("{line}{end}")))
                .to_vec(),
        };
        let mut keyed = ToolContext::new();
        let change = FileChange::created("id_rsa", 3).with_hunks(vec![hunk]);
        assert!(keyed.insert_result(FilesChanged(vec![change])).is_ok());
        let kept = keyed.result_context().kept();
        let changed = kept
            .get::<FilesChanged>()
            .ok()
            .flatten()
            .unwrap_or_default();
        let hunks = changed.0.first().map(|change| change.hunks.len());
        assert_eq!(hunks, Some(0), "{end:?}");
    }

    let huge = FileRead::new("x".repeat(MAX_KEPT_FACTS_BYTES), content_hash(b""));
    assert!(context.insert_result(FilesRead(vec![huge])).is_ok());
    let kept = context.result_context().kept();
    assert_eq!(kept.get::<FilesRead>().ok().flatten(), None);
    assert_eq!(
        kept.get::<ExitStatus>().ok().flatten(),
        Some(ExitStatus::Exited(3))
    );

    // Two values that each fit but not together: one goes, and what stays
    // is within the budget as JSON.
    #[derive(Serialize, Deserialize)]
    struct Note(String);
    impl ContextValue for Note {
        const KEY: &'static str = "test.note";
    }
    let mut two = ToolContext::new();
    let most = MAX_KEPT_FACTS_BYTES * 3 / 5;
    assert!(two.insert_result(RequestId("x".repeat(most))).is_ok());
    assert!(two.insert_result(Note("y".repeat(most))).is_ok());
    let kept = two.result_context().kept();
    let request = kept.get::<RequestId>().ok().flatten().is_some();
    let note = kept.get::<Note>().ok().flatten().is_some();
    assert!(request != note, "one of the two stays");
    let size = serde_json::to_vec(&kept).map_or(usize::MAX, |json| json.len());
    assert!(size <= MAX_KEPT_FACTS_BYTES, "{size}");
    // At the edge, the key's quotes and the map's braces count too.
    let mut edge = ToolContext::new();
    let fill = MAX_KEPT_FACTS_BYTES - RequestId::KEY.len() - 2;
    assert!(edge.insert_result(RequestId("x".repeat(fill))).is_ok());
    let kept = edge.result_context().kept();
    let size = serde_json::to_vec(&kept).map_or(usize::MAX, |json| json.len());
    assert!(size <= MAX_KEPT_FACTS_BYTES, "{size}");
}
