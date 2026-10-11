use rig_core::message::UserContent;

use super::{attached_file, attachments};

fn manifest() -> String {
    format!("{}/Cargo.toml", env!("CARGO_MANIFEST_DIR"))
}

#[test]
fn a_text_file_is_attached_as_numbered_lines_and_typed_text_is_not_one() {
    let path = manifest();
    let (attached, notes) = attachments(&format!("look at @{path}, please"));
    assert!(notes.is_empty(), "{notes:?}");
    assert_eq!(attached.len(), 1);
    let text = match attached.first().map(|(_, content)| content) {
        Some(UserContent::Text(text)) => text.text.as_str(),
        _ => "",
    };
    assert!(text.contains("     1\t[package]"), "{text}");
    let lines = std::fs::read_to_string(&path)
        .ok()
        .map(|manifest| manifest.lines().count());
    assert_eq!(
        attached_file(text).map(|(label, lines)| (label.to_owned(), Some(lines))),
        Some((path, lines))
    );
    assert_eq!(attached_file("<file path=\"x\"> is how it starts"), None);
}

#[test]
fn a_file_named_twice_is_attached_once_and_a_missing_one_not_at_all() {
    let path = manifest();
    let (attached, notes) = attachments(&format!("@{path} @{path} @no/such/file.rs"));
    assert_eq!(attached.len(), 1);
    assert!(notes.is_empty(), "{notes:?}");
}
