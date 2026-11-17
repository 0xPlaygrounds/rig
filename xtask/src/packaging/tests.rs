use std::collections::BTreeSet;

use serde_json::json;

use super::{
    Dependency, derives_into, docs_rs_gaps, indent, metadata, names,
    one_package_owns_the_rig_binary,
};

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
fn a_bevy_derive_uses_the_crate_it_expands_to() {
    assert!(derives_into("#[derive(Component, Default)]", "bevy_ecs"));
    assert!(derives_into(
        "#[derive(Clone, bevy_reflect::Reflect)]",
        "bevy_reflect"
    ));
    assert!(!derives_into("#[derive(Component)]", "bevy_reflect"));
    assert!(!derives_into(
        "#[derive(Debug)]\nstruct Component;",
        "bevy_ecs"
    ));
    assert!(!derives_into("#[derive(Serialize)]", "serde_json"));
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
    let library = Dependency {
        name: "steel-core".to_owned(),
        rename: None,
    };
    assert_eq!(library.identifier(), "steel");
}

#[test]
fn reported_lines_are_indented_under_their_package() {
    assert_eq!(indent(&["a", "b"]), "      a\n      b");
}

#[test]
fn docs_rs_must_list_every_feature_but_the_exclusions() {
    let features: BTreeSet<&str> = ["default", "agent", "surrealdb", "rmcp"].into();
    let excluded: BTreeSet<&str> = ["surrealdb"].into();
    let listed = |names: &[&str]| -> BTreeSet<String> {
        names.iter().map(|name| (*name).to_owned()).collect()
    };

    assert!(docs_rs_gaps(&features, &listed(&["agent", "rmcp"]), false, &excluded).is_empty());

    let missing = docs_rs_gaps(&features, &listed(&["agent"]), false, &excluded);
    assert_eq!(missing.len(), 1);
    assert!(missing.iter().any(|failure| failure.contains("rmcp")));

    let listed_excluded = docs_rs_gaps(
        &features,
        &listed(&["agent", "rmcp", "surrealdb"]),
        false,
        &excluded,
    );
    assert!(
        listed_excluded
            .iter()
            .any(|failure| failure.contains("surrealdb"))
    );

    let unknown = docs_rs_gaps(
        &features,
        &listed(&["agent", "rmcp", "typo"]),
        false,
        &excluded,
    );
    assert!(unknown.iter().any(|failure| failure.contains("typo")));

    let everything = docs_rs_gaps(&features, &listed(&[]), true, &excluded);
    assert_eq!(everything.len(), 1);
    assert!(everything[0].contains("all-features"));
}

#[test]
fn docs_rs_may_build_all_features_once_nothing_is_excluded() {
    let features: BTreeSet<&str> = ["default", "agent", "surrealdb"].into();
    assert!(docs_rs_gaps(&features, &BTreeSet::new(), true, &BTreeSet::new()).is_empty());
}

/// One package's metadata with the targets `targets`, each `(name, kind)`.
fn package(name: &str, targets: &[(&str, &str)]) -> serde_json::Value {
    let targets: Vec<_> = targets
        .iter()
        .map(|(name, kind)| json!({"name": name, "kind": [kind]}))
        .collect();
    json!({"name": name, "targets": targets})
}

#[test]
fn only_the_root_package_may_have_a_rig_binary() {
    let alone = json!({"packages": [
        package("rig", &[("rig", "lib"), ("rig", "bin")]),
        package("rig-cli", &[("rig_cli", "lib")]),
    ]});
    assert!(one_package_owns_the_rig_binary(&alone).is_empty());
    let two = json!({"packages": [
        package("rig", &[("rig", "bin")]),
        package("rig-cli", &[("rig", "bin")]),
    ]});
    let failures = one_package_owns_the_rig_binary(&two);
    assert!(
        failures
            .iter()
            .any(|failure| failure.contains("[rig, rig-cli]"))
    );
    let elsewhere = json!({"packages": [
        package("rig", &[("rig", "lib")]),
        package("rig-cli", &[("rig", "bin")]),
    ]});
    assert!(!one_package_owns_the_rig_binary(&elsewhere).is_empty());
}

#[test]
fn the_workspace_has_one_rig_binary_in_the_root_package() {
    let workspace = metadata(&crate::workspace_root(), &["--no-deps"]);
    assert!(workspace.is_ok(), "{workspace:?}");
    let Ok(workspace) = workspace else { return };
    assert_eq!(
        one_package_owns_the_rig_binary(&workspace),
        Vec::<String>::new()
    );
}
