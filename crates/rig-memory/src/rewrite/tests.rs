use std::collections::BTreeMap;
use std::sync::Arc;

use rig_core::completion::Message;
use rig_core::conversation::{Aside, Author, Change, Conversation};
use rig_core::message::{ImageMediaType, UserContent};
use serde_json::json;

use super::{Messages, Place, Rewrite, Strings, rewrite_file, rewrite_value};
use crate::format::stamp_of;
use crate::{
    Files, LogRewrite, MemoryStore, SESSION, Session, SessionError, Store, Writer, load, rewrite,
    save,
};

/// Replaces `sk-1` with `[R]` everywhere.
fn redact() -> Strings<impl Fn(&mut String) -> bool + Send + Sync> {
    Strings(|text: &mut String| {
        let hit = text.contains("sk-1");
        if hit {
            *text = text.replace("sk-1", "[R]");
        }
        hit
    })
}

fn image(data: &str, media_type: ImageMediaType) -> UserContent {
    UserContent::image_base64(data, Some(media_type), None)
}

/// Every log of `store` with its bytes.
fn logs(store: &dyn Store) -> BTreeMap<String, Vec<u8>> {
    let names = store.logs().expect("listed");
    let read = |name: String| {
        let bytes = store.read_log(&name).expect("read");
        (name, bytes)
    };
    names.into_iter().map(read).collect()
}

/// Everything `store` holds, to tell that nothing was written.
fn everything(store: &MemoryStore) -> (BTreeMap<String, Vec<u8>>, Vec<String>, Vec<String>) {
    let mut blobs = store.blobs().expect("listed");
    blobs.sort();
    let mut files = store.files().expect("listed");
    files.sort();
    (logs(store), blobs, files)
}

fn lines(bytes: &[u8]) -> Vec<&[u8]> {
    bytes
        .split_inclusive(|byte| *byte == b'\n')
        .collect::<Vec<_>>()
}

/// A rewrite maps every value of every record in place: each record keeps
/// its place and stamp, an unchanged one its bytes; a changed log gets a
/// fresh lineage and says who rewrote it; every file goes through it; a
/// value read before it no longer saves; a blob no record names any more
/// is removed.
#[test]
fn a_rewrite_maps_every_value_in_place_keeping_each_records_stamp() {
    let store = Arc::new(MemoryStore::default());
    let (writer, _) = Writer::open(store.clone()).expect("opened");
    writer.start();
    let mut main = Conversation::new();
    let look = main.push(Message::User {
        content: vec![
            UserContent::text("look at sk-1"),
            image("cG5n", ImageMediaType::PNG),
        ],
    });
    main.push(Message::User {
        content: vec![image("Z2lm", ImageMediaType::GIF)],
    });
    main.push(Message::assistant("the key sk-1 is set"));
    let note = json!({ "sk-1": "about sk-1" });
    main.add_aside(Aside::new("note", Author::plugin("btw"), note).after(Some(look)));
    writer.save("main", &mut main);
    writer.save_value("main", "rig::title", Some(json!("about sk-1")), true);
    writer.save_value(SESSION, "theme", Some(json!("dark")), false);
    let mut other = Conversation::from_messages([Message::user("plain")]);
    writer.save("other", &mut other);
    let mut replaced = Conversation::from_messages([Message::user("a reset with sk-1")]);
    let reset = writer.save("other", &mut replaced);
    assert!(matches!(reset.as_slice(), [only] if matches!(only.change, Change::Reset { .. })));
    let mut clean = Conversation::from_messages([Message::user("nothing to hide")]);
    writer.save("clean", &mut clean);
    writer.flush().expect("flushed");
    let files: [(&str, &[u8]); 4] = [
        ("meta.json", b"{\n  \"title\": \"sk-1\"\n}\n"),
        (
            "effects.jsonl",
            b"{\"a\":\"sk-1\"}\n{\"b\":1}\n{\"torn sk-1",
        ),
        ("notes.md", b"a line with sk-1\n"),
        ("image.bin", &[0xff, 0xfe, b's', b'k', b'-', b'1']),
    ];
    for (name, bytes) in files {
        store.write_file(name, bytes).expect("written");
    }
    let before = logs(&*store);
    let stale = load(&*store, "main").expect("loaded");
    let blobs = store.blobs().expect("listed");
    assert_eq!(blobs.len(), 2);

    let by = Author::plugin("redact");
    let rewritten = rewrite(&*store, by.clone(), &redact()).expect("rewritten");
    let names: Vec<&str> = rewritten.logs.iter().map(|log| log.log.as_str()).collect();
    assert_eq!(names, ["main", "other"]);
    let main_log = rewritten.logs.first().expect("main");
    // Two pushes, the aside and the title; the push of only an image is
    // left.
    assert_eq!(main_log.records, 4);
    assert_eq!(
        rewritten.files,
        ["effects.jsonl", "meta.json", "notes.md"],
        "a binary file is left"
    );
    assert!(rewritten.blobs_removed.is_empty());

    let after = logs(&*store);
    for (name, old) in &before {
        let new = after.get(name).expect("still there");
        let (old, new) = (lines(old), lines(new));
        assert_eq!(old.len(), new.len(), "{name}");
        for (at, (old, new)) in old.iter().zip(&new).enumerate() {
            assert_eq!(
                stamp_of(old).expect("a stamp"),
                stamp_of(new).expect("a stamp")
            );
            let secret = String::from_utf8_lossy(old).contains("sk-1");
            if at > 0 && !secret {
                assert_eq!(old, new, "an unchanged record keeps its bytes");
            }
            assert!(!String::from_utf8_lossy(new).contains("sk-1"), "{name}");
        }
    }
    assert_eq!(after.get("clean"), before.get("clean"));
    assert_eq!(after.get(SESSION), before.get(SESSION));
    let read = |name| store.read_file(name).expect("a file");
    assert_eq!(read("meta.json"), b"{\n  \"title\": \"[R]\"\n}\n");
    assert_eq!(
        read("effects.jsonl"),
        b"{\"a\":\"[R]\"}\n{\"b\":1}\n{\"torn [R]"
    );
    assert_eq!(read("notes.md"), b"a line with [R]\n");

    let session = Session::read(&*store).expect("read");
    let saved = session.agent("main").expect("main");
    assert_eq!(saved.saved.get("rig::title"), Some(&json!("about [R]")));
    assert_eq!(saved.conversation.revision(), main.revision());
    assert_ne!(saved.conversation.lineage(), main.lineage());
    let rewrote = &saved.rewrites;
    assert!(matches!(rewrote.as_slice(), [LogRewrite { by: who, .. }] if *who == by));
    let texts: Vec<String> = saved
        .conversation
        .messages()
        .iter()
        .map(|message| text_of(message))
        .collect();
    assert_eq!(texts, ["look at [R]", "", "the key [R] is set"]);
    assert!(
        saved
            .conversation
            .messages()
            .first()
            .is_some_and(|message| message.images().count() == 1),
        "the image is read back from its blob"
    );
    let (_, aside) = saved.conversation.asides().next().expect("the aside");
    assert_eq!(aside.value, json!({ "[R]": "about [R]" }));
    let other = session.agent("other").expect("other");
    assert_eq!(
        other.conversation.messages(),
        [Message::user("a reset with [R]")].map(std::sync::Arc::new)
    );
    let mut stale = stale;
    stale.push(Message::user("more sk-1"));
    let refused = save(&*store, "main", &mut stale);
    assert!(
        matches!(&refused, Err(SessionError::Changed { log }) if log == "main"),
        "{refused:?}"
    );

    // A map that takes an image out of every record leaves its blob unnamed.
    let images_only = |content: &[UserContent]| {
        content
            .iter()
            .all(|item| matches!(item, UserContent::Image(_)))
    };
    let drop_images = Messages(move |message: &mut Message| match message {
        Message::User { content } if images_only(content.as_slice()) => {
            *message = Message::user("[an image]");
            true
        }
        _ => false,
    });
    let rewritten = rewrite(&*store, Author::plugin("tidy"), &drop_images).expect("rewritten");
    assert_eq!(rewritten.blobs_removed.len(), 1);
    assert!(rewritten.files.is_empty(), "messages only");
    assert_eq!(store.blobs().expect("listed").len(), 1);
    assert!(
        blobs
            .iter()
            .any(|blob| rewritten.blobs_removed.contains(blob))
    );
    let main = Session::read(&*store).expect("read");
    assert_eq!(
        main.agent("main").map(|agent| agent.rewrites.len()),
        Some(2)
    );
}

fn text_of(message: &Message) -> String {
    match message {
        Message::User { content } => content
            .iter()
            .filter_map(|item| match item {
                UserContent::Text(text) => Some(text.text.clone()),
                _ => None,
            })
            .collect(),
        Message::Assistant(_) | Message::System { .. } => serde_json::to_value(message)
            .ok()
            .and_then(|value| {
                let text = value.pointer("/content/0/text")?.as_str()?.to_owned();
                Some(text)
            })
            .unwrap_or_default(),
    }
}

/// A session of another format, or broken, is refused before anything is
/// written.
#[test]
fn a_session_it_cannot_read_is_refused_before_anything_is_written() {
    let store = MemoryStore::default();
    let mut conversation = Conversation::from_messages([Message::user("sk-1")]);
    save(&store, "main", &mut conversation).expect("saved");
    store.write_file("notes.md", b"sk-1").expect("written");
    let old = store.clone();
    old.append_log(
        "old",
        b"{\"seq\":0,\"n\":9,\"t\":0,\"type\":\"header\",\"format\":\"rig-session/1\",\"log\":\"old\",\"lineage\":7}\n",
    )
    .expect("written");
    let before = everything(&store);
    let refused = rewrite(&store, Author::plugin("redact"), &redact());
    assert!(
        matches!(&refused, Err(SessionError::Format { found: Some(found), .. }) if found == "rig-session/1"),
        "{refused:?}"
    );
    assert_eq!(everything(&store), before);

    store.remove_log("old").expect("removed");
    store
        .append_log(
            "main",
            b"{\"seq\":2,\"n\":2,\"t\":0,\"type\":\"mystery\"}\n",
        )
        .expect("written");
    let before = everything(&store);
    let refused = rewrite(&store, Author::plugin("redact"), &redact());
    assert!(
        matches!(&refused, Err(SessionError::Record { line: 3, .. })),
        "{refused:?}"
    );
    assert_eq!(everything(&store), before);
}

/// A map that also changes a field name or a tag of a message keeps the
/// message readable and still maps its data.
#[test]
fn a_map_that_hits_a_field_name_keeps_the_message_readable() {
    let text = Strings(|text: &mut String| {
        let hit = text.contains("text");
        if hit {
            *text = text.replace("text", "[R]");
        }
        hit
    });
    let mut message = Message::user("my text");
    assert!(text.message(&Place::File { name: "x" }, &mut message));
    assert_eq!(message, Message::user("my [R]"));
}

/// A branch kept as an aside is saved like the rest of the conversation:
/// its image is a blob, read back whole, and a rewrite reaches its
/// messages and title once each, so the active branch switched back to
/// carries the rewrite.
#[test]
fn a_rewrite_reaches_a_stashed_branch_and_its_image_is_a_blob() {
    let store = MemoryStore::default();
    let by = Author::plugin("tree");
    let mut conversation = load(&store, "main").expect("loaded");
    let shown = Message::User {
        content: vec![
            UserContent::text("my key is sk-1"),
            image("cG5n", ImageMediaType::PNG),
        ],
    };
    conversation.extend([Message::user("Hi."), shown.clone()]);
    conversation.push(Message::assistant("Seen sk-1."));
    let branch = conversation.branch_off(1, "about sk-1", by.clone());
    assert!(branch.is_ok(), "{branch:?}");
    conversation.push(Message::assistant("Hello."));
    save(&store, "main", &mut conversation).expect("saved");

    let saved = logs(&store).remove("main").unwrap_or_default();
    let text = String::from_utf8_lossy(&saved);
    assert!(text.contains("blob:") && !text.contains("cG5n"), "{text}");
    assert_eq!(store.blobs().map(|blobs| blobs.len()).ok(), Some(1));
    let read = load(&store, "main").expect("read");
    let stashed = read.branches().into_iter().next().map(|kept| kept.branch);
    let first = stashed.as_ref().and_then(|branch| branch.entries.first());
    assert_eq!(first.map(|entry| &entry.message), Some(&shown.into()));

    let rewritten = rewrite(&store, Author::plugin("redact"), &redact()).expect("rewritten");
    assert_eq!(rewritten.logs.len(), 1);
    assert_eq!(rewritten.blobs_removed, Vec::<String>::new());
    let mut read = load(&store, "main").expect("read");
    let kept = read.branches().into_iter().next().expect("a branch");
    let (id, stashed) = (kept.id, kept.branch);
    assert_eq!(stashed.title, "main");
    assert_eq!(read.line_title(), "about [R]", "the title is rewritten too");
    let texts: Vec<String> = stashed
        .entries
        .iter()
        .map(|entry| serde_json::to_string(&entry.message).unwrap_or_default())
        .collect();
    assert!(texts.iter().all(|text| !text.contains("sk-1")), "{texts:?}");
    assert!(
        texts.iter().any(|text| text.contains("my key is [R]")),
        "{texts:?}"
    );
    assert!(
        texts.iter().any(|text| text.contains("cG5n")),
        "the image is back"
    );

    read.switch_branch(id, by).expect("switched");
    let back = read.messages().get(2);
    assert_eq!(back, Some(&Message::assistant("Seen [R].").into()));
    save(&store, "main", &mut read).expect("saved");
    assert_eq!(load(&store, "main").expect("read"), read);
}

/// A rewrite reaches the saved asides a stashed branch keeps, which come
/// back rewritten when it is switched to; a branch it finds nothing in
/// keeps its record's bytes.
#[test]
fn a_rewrite_reaches_the_asides_a_stashed_branch_keeps() {
    let store = MemoryStore::default();
    let by = Author::plugin("tree");
    let mut conversation = load(&store, "main").expect("loaded");
    conversation.extend([Message::user("Hi."), Message::user("Deploy it.")]);
    let deploy = conversation.id(1);
    let note = Aside::new("note", by.clone(), json!("used sk-1")).after(deploy);
    conversation.add_aside(note);
    let secret = conversation.branch_off(1, "retry", by.clone());
    let secret = secret.expect("branched");
    conversation.push(Message::user("Wait."));
    assert!(conversation.branch_off(1, "later", by.clone()).is_ok());
    save(&store, "main", &mut conversation).expect("saved");
    let before = logs(&store).remove("main").unwrap_or_default();

    rewrite(&store, Author::plugin("redact"), &redact()).expect("rewritten");
    let after = logs(&store).remove("main").unwrap_or_default();
    let text = String::from_utf8_lossy(&after);
    assert!(!text.contains("sk-1"), "{text}");
    let waited: Vec<&[u8]> = lines(&before)
        .into_iter()
        .filter(|line| String::from_utf8_lossy(line).contains("Wait."))
        .collect();
    assert!(!waited.is_empty());
    let kept = lines(&after);
    assert!(waited.iter().all(|line| kept.contains(line)), "{text}");

    let mut read = load(&store, "main").expect("read");
    read.switch_branch(secret, by).expect("switched");
    let notes: Vec<_> = read
        .asides()
        .filter(|(_, aside)| aside.kind == "note")
        .map(|(_, aside)| (aside.after, aside.value.clone()))
        .collect();
    assert_eq!(notes, [(deploy, json!("used [R]"))]);
}

#[test]
fn what_a_plugin_keeps_outside_the_store_maps_through_the_same_rewrite() {
    let kept = MemoryStore::default();
    kept.write_file("tui/history.jsonl", b"\"my key is sk-1\"\n\"hello\"\n")
        .expect("written");
    assert!(rewrite_file(&kept, "tui/history.jsonl", &redact()).expect("mapped"));
    let mapped = kept.read_file("tui/history.jsonl").expect("read");
    assert_eq!(mapped, b"\"my key is [R]\"\n\"hello\"\n");
    assert!(!rewrite_file(&kept, "tui/missing.jsonl", &redact()).expect("left"));

    let at = Place::File { name: "copy" };
    let mut call = Message::user("run `echo sk-1`");
    assert!(rewrite_value(&redact(), &at, "copy", &mut call).expect("reads back"));
    assert_eq!(call, Message::user("run `echo [R]`"));
    assert!(!rewrite_value(&redact(), &at, "copy", &mut call).expect("reads back"));
    // A map that leaves the value unreadable leaves it as it was.
    let everything = Strings(|text: &mut String| {
        *text = "x".to_owned();
        true
    });
    assert!(rewrite_value(&everything, &at, "copy", &mut call).is_err());
    assert_eq!(call, Message::user("run `echo [R]`"));
}
