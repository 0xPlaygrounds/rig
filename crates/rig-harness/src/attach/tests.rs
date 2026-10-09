use rig_core::message::UserContent;

use super::{attached_file, attachments};

fn manifest() -> String {
    format!("{}/Cargo.toml", env!("CARGO_MANIFEST_DIR"))
}

#[test]
fn a_text_file_is_attached_as_numbered_lines() {
    let path = manifest();
    let (attached, notes) = attachments(&format!("look at @{path}, please"));
    assert!(notes.is_empty(), "{notes:?}");
    assert_eq!(attached.len(), 1);
    let UserContent::Text(text) = &attached[0].content else {
        panic!("not text: {:?}", attached[0].content);
    };
    assert!(text.text.contains("     1\t[package]"), "{}", text.text);
    let (label, lines) = attached_file(&text.text).expect("an attached file");
    assert_eq!(label, path);
    assert_eq!(
        lines,
        std::fs::read_to_string(&path).unwrap().lines().count()
    );
}

#[test]
fn a_file_named_twice_is_attached_once_and_a_missing_one_not_at_all() {
    let path = manifest();
    let (attached, notes) = attachments(&format!("@{path} @{path} @no/such/file.rs"));
    assert_eq!(attached.len(), 1);
    assert!(notes.is_empty(), "{notes:?}");
}

#[test]
fn typed_text_is_not_an_attached_file() {
    assert_eq!(attached_file("<file path=\"x\"> is how it starts"), None);
}
