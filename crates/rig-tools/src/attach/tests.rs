use std::path::PathBuf;

use rig_core::completion::Message;
use rig_core::conversation::{Author, Conversation, Entry, attached_file};
use rig_core::message::UserContent;

use super::{Reference, attachments, first_typed, typed};
use crate::workspace::Local;

/// The crate's own directory, where its manifest is.
fn base() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

#[test]
fn a_text_file_is_attached_as_its_lines_and_typed_text_is_not_one() {
    let path = base().join("Cargo.toml").to_string_lossy().into_owned();
    let attached = attachments(&format!("look at @{path}, please"), &base(), &Local);
    assert!(attached.notes.is_empty(), "{:?}", attached.notes);
    assert_eq!(attached.files.len(), 1);
    let text = match attached.files.first().map(|(_, content)| content) {
        Some(UserContent::Text(text)) => text.text.as_str(),
        _ => "",
    };
    assert!(text.contains("\n[package]\n"), "{text}");
    let lines = std::fs::read_to_string(&path)
        .ok()
        .map(|manifest| manifest.lines().count());
    assert_eq!(
        attached_file(text).map(|(label, body)| (label.to_owned(), Some(body.lines().count()))),
        Some((path, lines))
    );
    assert_eq!(attached_file("<file path=\"x\"> is how it starts"), None);
}

/// A reference resolves from the base, once however often it is named; a
/// path with no file there is missing, and a word that only looks like a
/// name stays text.
#[test]
fn references_resolve_from_the_base_and_missing_paths_are_reported() {
    let attached = attachments(
        "@Cargo.toml and @Cargo.toml, @missing/x.rs, @gone.rs. Ask @alice.smith or @bob",
        &base(),
        &Local,
    );
    let labels: Vec<&str> = attached
        .files
        .iter()
        .map(|(label, _)| label.as_str())
        .collect();
    assert_eq!(labels, ["Cargo.toml"]);
    let pasted = attachments(r#"var s = @"hello world";"#, &base(), &Local);
    assert!(pasted.warnings.is_empty(), "{:?}", pasted.warnings);
    assert_eq!(
        attached.warnings,
        ["@missing/x.rs: no such file", "@gone.rs: no such file"]
    );
    assert!(attached.notes.is_empty(), "{:?}", attached.notes);
}

/// A file with no extension, such as a `Makefile`, is attached when it is
/// there; the same word with no file there stays text.
#[test]
fn an_extensionless_file_that_is_there_is_attached() {
    let dir = std::env::temp_dir().join(format!("rig-tools-attach-{}", std::process::id()));
    std::fs::create_dir_all(&dir).ok();
    std::fs::write(dir.join("Makefile"), "all:\n").ok();
    let attached = attachments("run @Makefile, ask @reviewer", &dir, &Local);
    let labels: Vec<&str> = attached
        .files
        .iter()
        .map(|(label, _)| label.as_str())
        .collect();
    assert_eq!(labels, ["Makefile"]);
    assert!(attached.warnings.is_empty(), "{:?}", attached.warnings);
    std::fs::remove_dir_all(dir).ok();
}

/// A path is written so it reads back whole, quoted only when a space, a
/// leading quote or trailing punctuation would cut it; every other
/// reference reads as before.
#[test]
fn a_formatted_path_reads_back_whole() {
    for path in [
        "notes.md",
        "src/lib.rs",
        "~/x",
        "bench:~/a b/c.txt",
        "attach notes.txt",
        "say \"hi\".txt",
        "\"quoted",
        "ends.",
        "notes.md:2",
        r"C:\Users\me\my file.png",
    ] {
        let text = format!("look at {}, then", Reference::format(path));
        let paths: Vec<_> = Reference::parse(&text).map(|found| found.path).collect();
        assert_eq!(paths, [path], "{text}");
    }
    assert_eq!(Reference::format("notes.md"), "@notes.md");
    assert_eq!(Reference::format("my file.txt"), "@\"my file.txt\"");
    assert_eq!(Reference::format("notes.md:2"), "@\"notes.md:2\"");
    assert_eq!(Reference::format("bench:~/x"), "@bench:~/x");
    let text = "@notes.md, @~/x and @alice; mail a@b.c, @\"\" or @\"unclosed @a.rs";
    let paths: Vec<_> = Reference::parse(text).map(|found| found.path).collect();
    assert_eq!(paths, ["notes.md", "~/x", "alice", "unclosed @a.rs"]);
    let typing = |text| Reference::typing(text).map(|found| found.path.into_owned());
    assert_eq!(typing("see @\"my fi").as_deref(), Some("my fi"));
    assert_eq!(typing("see @src/li").as_deref(), Some("src/li"));
    assert_eq!(typing("see @\"my file.txt\""), None);
    assert_eq!(typing("see @src/lib.rs "), None);
}

/// A temporary directory of its own for `name`, empty.
fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("rig-tools-{name}-{}", std::process::id()));
    std::fs::remove_dir_all(&dir).ok();
    std::fs::create_dir_all(&dir).ok();
    dir
}

/// The text of each file attached.
fn texts(attached: &super::Attached) -> Vec<&str> {
    attached
        .files
        .iter()
        .filter_map(|(_, content)| match content {
            UserContent::Text(text) => Some(text.text.as_str()),
            _ => None,
        })
        .collect()
}

/// A picked name with a space attaches that file, not a file named by its
/// first word, and is warned about by its whole name when it is gone.
#[test]
fn a_name_with_a_space_attaches_that_file() {
    let dir = scratch("attach-space");
    std::fs::write(dir.join("attach notes.txt"), "INTENDED").ok();
    std::fs::write(dir.join("attach"), "WRONG_PREFIX_FILE").ok();
    let text = format!("quote {}", Reference::format("attach notes.txt"));
    let attached = attachments(&text, &dir, &Local);
    let quoted = texts(&attached);
    assert!(
        quoted.len() == 1 && quoted.iter().all(|text| text.contains("INTENDED")),
        "{quoted:?}"
    );
    std::fs::remove_file(dir.join("attach notes.txt")).ok();
    let attached = attachments(&text, &dir, &Local);
    assert!(attached.files.is_empty(), "{:?}", attached.files);
    assert_eq!(attached.warnings, ["@\"attach notes.txt\": no such file"]);
    std::fs::remove_dir_all(dir).ok();
}

/// A path that cannot be looked at is reported with why, not as missing.
#[cfg(unix)]
#[test]
fn an_unreadable_path_says_why() {
    use std::os::unix::fs::PermissionsExt;
    let dir = scratch("attach-locked");
    let locked = dir.join("locked");
    std::fs::create_dir_all(&locked).ok();
    std::fs::write(locked.join("x.txt"), "secret").ok();
    std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o000)).ok();
    // Root reads it anyway: nothing to check there.
    let readable = std::fs::metadata(locked.join("x.txt")).is_ok();
    let attached = attachments("read @locked/x.txt", &dir, &Local);
    std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o755)).ok();
    std::fs::remove_dir_all(&dir).ok();
    if readable {
        return;
    }
    assert!(attached.files.is_empty(), "{:?}", attached.files);
    let said = attached.warnings.join("\n");
    assert!(
        said.starts_with("could not read locked/x.txt: ") && said.contains("ermission denied"),
        "{said}"
    );
}

#[test]
fn what_the_user_typed_leaves_out_attached_files_and_delivered_notes() {
    let attached = "<file path=\"a.txt\">\nfile text\n</file>";
    let mut conversation = Conversation::new();
    conversation.push(Entry::new(Message::user("build failed")).by(Author::plugin("build")));
    conversation.push(Message::assistant("ok"));
    conversation.push(Message::user(attached));
    conversation.push(Message::user("  fix it  "));
    conversation.push(Message::user("and test"));
    assert_eq!(typed(&conversation, 0), None);
    assert_eq!(typed(&conversation, 1), None);
    assert_eq!(typed(&conversation, 2), None);
    assert_eq!(typed(&conversation, 4).as_deref(), Some("and test"));
    assert_eq!(first_typed(&conversation).as_deref(), Some("fix it"));
}
