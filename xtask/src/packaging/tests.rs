use super::{Dependency, indent, names};

#[test]
fn a_crate_name_matches_only_as_a_whole_word() {
    assert!(names("use serde::Serialize;", "serde"));
    assert!(names("    serde_json::to_value(&x)", "serde_json"));
    assert!(names("#[derive(serde::Deserialize)]", "serde"));
    // The bug this guards: `serde` must not be found inside `serde_json`.
    assert!(!names("use serde_json::Value;", "serde"));
    assert!(!names("let rig_derived = 1;", "rig_derive"));
    assert!(!names("", "serde"));
}

#[test]
fn a_renamed_dependency_is_looked_up_by_its_rust_name() {
    let renamed = Dependency {
        name: "rig-core".to_owned(),
        rename: Some("core_alias".to_owned()),
    };
    assert_eq!(renamed.identifier(), "core_alias");
    let plain = Dependency {
        name: "rig-derive".to_owned(),
        rename: None,
    };
    assert_eq!(plain.identifier(), "rig_derive");
}

#[test]
fn reported_lines_are_indented_under_their_package() {
    assert_eq!(indent(&["a", "b"]), "      a\n      b");
}
