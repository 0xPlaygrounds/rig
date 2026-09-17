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

/// The exemption list is the only way a caller-less function passes, and it
/// is named one at a time so a new one is argued for rather than matched by
/// a path pattern.
#[test]
fn the_exemption_list_is_named_one_at_a_time() {
    for name in QUERY_BUILDERS {
        assert!(
            !name.is_empty() && name.chars().all(|c| c.is_lowercase() || c == '_'),
            "`{name}` is not a function name"
        );
    }
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
