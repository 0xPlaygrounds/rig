//! Both gates, stated as the source each one accepts and rejects.

use super::*;

#[test]
fn a_public_function_is_found_by_its_declaration() {
    let source = "\
pub fn kept(&self) -> u8 { 1 }
pub const fn also_kept() -> u8 { 2 }
pub async fn awaited() {}
pub trait Contract {}
fn private() {}
pub struct NotAFunction;
";
    let found = declarations(source);
    assert_eq!(
        found,
        vec![
            ("fn", "kept".to_owned()),
            ("fn", "also_kept".to_owned()),
            ("fn", "awaited".to_owned()),
            ("trait", "Contract".to_owned()),
        ],
        "a private fn and a struct are not the surface this gate is about"
    );
}

/// The second gate needs the *wire* key, not the field name: a renamed
/// field is recorded under the rename, and looking for the Rust name would
/// report it as unrecorded.
#[test]
fn a_field_carries_the_key_the_wire_uses() {
    let source = "\
pub struct Reply {
    pub plain: Option<String>,
    #[serde(default, rename = \"inferenceQueueTime\")]
    pub inference_queue_time: Option<f64>,
    #[serde(default)]
    #[serde(rename = \"camelCased\")]
    pub snake_cased: Option<u8>,
    private: u8,
}
";
    let found = fields(source);
    assert_eq!(
        found,
        vec![
            ("plain".to_owned(), "plain".to_owned()),
            (
                "inference_queue_time".to_owned(),
                "inferenceQueueTime".to_owned()
            ),
            ("snake_cased".to_owned(), "camelCased".to_owned()),
        ]
    );
}

/// A container's `rename_all` is the key for every field that has no
/// `rename` of its own — looking the snake_case name up in the recordings
/// would report a carried field as unrecorded.
#[test]
fn a_container_rename_all_decides_the_key() {
    let source = "\
#[derive(Deserialize)]
#[serde(rename_all = \"camelCase\")]
pub struct Cached {
    pub update_time: Option<String>,
    #[serde(rename = \"explicit\")]
    pub overridden: Option<String>,
}
";
    let found = fields(source);
    assert_eq!(
        found,
        vec![
            ("update_time".to_owned(), "updateTime".to_owned()),
            ("overridden".to_owned(), "explicit".to_owned()),
        ]
    );
    assert_eq!(camel_case("inference_queue_time"), "inferenceQueueTime");
    assert_eq!(camel_case("plain"), "plain");
}

#[test]
fn an_identifier_is_counted_by_word_not_by_substring() {
    let counts = words("let model = model_ref; models(model)");
    assert_eq!(counts.get("model").copied(), Some(2), "{counts:?}");
    assert_eq!(counts.get("model_ref").copied(), Some(1));
    assert_eq!(counts.get("models").copied(), Some(1));
}

/// A key only counts as recorded when the recording uses it *as a key*: a
/// provider's reply text that happens to contain the word is not evidence
/// that the field is carried.
#[test]
fn a_recorded_key_is_a_key_and_not_a_value() {
    let dir = tempdir();
    let cassettes = dir.join("tests/cassettes");
    std::fs::create_dir_all(&cassettes).expect("the fixture root");
    std::fs::write(
        cassettes.join("one.yaml"),
        "body: '{\"updateTime\":\"now\",\"text\":\"the word inferenceQueueTime appears here\"}'",
    )
    .expect("a cassette");
    std::fs::create_dir_all(dir.join("crates/rig-verify/fixtures")).expect("the golden root");

    let keys = recorded_keys(&dir).expect("the keys");
    assert!(keys.contains("updateTime"), "{keys:?}");
    assert!(keys.contains("text"));
    assert!(
        !keys.contains("inferenceQueueTime"),
        "a word inside a value is not a key: {keys:?}"
    );
    std::fs::remove_dir_all(&dir).ok();
}

/// Most cassettes store the body as a double-quoted YAML scalar, so the JSON
/// quotes arrive escaped. Reading those is the whole point of the gate: if
/// they are skipped, a field 180 recordings carry looks unrecorded and the
/// gate licenses deleting it.
#[test]
fn an_escaped_body_is_read_like_a_plain_one() {
    let dir = tempdir();
    let cassettes = dir.join("tests/cassettes");
    std::fs::create_dir_all(&cassettes).expect("the fixture root");
    std::fs::write(
        cassettes.join("one.yaml"),
        "    body: \"{\\\"updateTime\\\":\\\"now\\\",\\\"response_type\\\":\\\"message\\\"}\"",
    )
    .expect("a cassette");
    std::fs::create_dir_all(dir.join("crates/rig-verify/fixtures")).expect("the golden root");

    let keys = recorded_keys(&dir).expect("the keys");
    assert!(keys.contains("updateTime"), "{keys:?}");
    assert!(keys.contains("response_type"), "{keys:?}");
    std::fs::remove_dir_all(&dir).ok();
}

/// Test code is not a reader: a `pub fn` whose only callers are tests is
/// surface kept for the test that calls it.
#[test]
fn test_files_are_not_readers() {
    for path in [
        "crates/rig-core/src/wire/tests.rs",
        "crates/rig-core/src/providers/openai/wire/tests.rs",
        "tests/providers/openai/support.rs",
        "crates/rig-ecs/tests/run_binding.rs",
    ] {
        assert!(is_test(path), "{path}");
    }
    for path in [
        "crates/rig-core/src/wire.rs",
        "crates/rig-core/src/test_utils/http.rs",
    ] {
        assert!(
            !is_test(path),
            "{path} ships, so what it reads keeps a field alive"
        );
    }
}

/// A scratch directory of this test's own.
fn tempdir() -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "rig-check-dead-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|since| since.as_nanos())
            .unwrap_or_default()
    ));
    std::fs::create_dir_all(&dir).expect("a scratch directory");
    dir
}

/// The two gates answer different questions, and only one of them can speak
/// about a field a caller sets: `FileSearchTool`'s store names have no
/// in-tree constructor and appear in no recording, and deleting them left a
/// search of nowhere. A written field is a request option, not dead weight.
#[test]
fn a_field_a_caller_sets_is_never_reported_as_unread() {
    let source = r#"
    pub struct FileSearchTool {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub file_search_store_names: Option<Vec<String>>,
    }
    pub struct ListedModel {
        #[serde(default)]
        pub is_deprecated: Option<bool>,
    }
"#;
    let found = fields(source);
    assert!(
        !found
            .iter()
            .any(|(name, _)| name == "file_search_store_names"),
        "a written field is a request option: {found:?}"
    );
    assert!(
        found.iter().any(|(name, _)| name == "is_deprecated"),
        "a read-only field is still judged: {found:?}"
    );
}

/// A container's casing does not leak to the next container.
#[test]
fn a_neighbours_casing_does_not_key_this_container() {
    let source = "\
#[derive(Deserialize)]
#[serde(rename_all = \"camelCase\")]
pub struct Renamed {
    pub update_time: Option<String>,
}

#[derive(Deserialize)]
pub struct Plain {
    pub max_items: Option<u64>,
}
";
    let found = fields(source);
    assert_eq!(
        found,
        vec![
            ("update_time".to_owned(), "updateTime".to_owned()),
            ("max_items".to_owned(), "max_items".to_owned()),
        ],
        "the second container has no `rename_all` of its own"
    );
}

/// A word inside a value is not a key, whichever quote it sits behind.
/// Accepting any quote-delimited run before a colon minted `ent` and `lue`
/// out of the middles of values, and a junk key that collides with a short
/// field name reads as "a recording carries this".
#[test]
fn a_key_must_open_where_a_key_can_open() {
    let dir = tempdir();
    let cassettes = dir.join("tests/cassettes");
    std::fs::create_dir_all(&cassettes).expect("the fixture root");
    std::fs::write(
        cassettes.join("one.yaml"),
        "    body: \"{\\\"content\\\":\\\"a value: with a colon\\\",\\\"id\\\":\\\"x\\\"}\"",
    )
    .expect("a cassette");
    std::fs::create_dir_all(dir.join("crates/rig-verify/fixtures")).expect("the golden root");

    let keys = recorded_keys(&dir).expect("the keys");
    assert!(keys.contains("content"), "{keys:?}");
    assert!(keys.contains("id"), "{keys:?}");
    assert!(
        !keys.contains("a value"),
        "a colon inside a value does not make the text before it a key: {keys:?}"
    );
    std::fs::remove_dir_all(&dir).ok();
}

/// The gate reports and does not fail: its inputs cannot distinguish a
/// symbol with no caller from one whose caller is a string, a macro body or
/// a downstream crate, so a run over a workspace full of uncalled surface
/// still succeeds. What it must not do is fail a release on a heuristic.
#[test]
fn the_gate_reports_rather_than_failing() {
    let dir = tempdir();
    let core = dir.join("crates/rig-core/src");
    std::fs::create_dir_all(&core).expect("the crate root");
    std::fs::write(
        core.join("lib.rs"),
        "pub fn nobody_calls_this() {}\npub trait NobodyImplementsThis {}\n",
    )
    .expect("a source file");
    std::fs::create_dir_all(dir.join("tests/cassettes")).expect("the cassette root");
    std::fs::create_dir_all(dir.join("crates/rig-verify/fixtures")).expect("the golden root");

    assert!(
        check(&dir).is_ok(),
        "an uncalled function is reported, not a build failure"
    );
    std::fs::remove_dir_all(&dir).ok();
}
