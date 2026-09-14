use super::*;
use serde_json::json;

fn metadata(version: &str) -> Value {
    json!({"workspace_members":["rig-ecs"], "packages": [
        {"id":"rig-ecs", "name":"rig-ecs", "source":null, "dependencies":[{"name":"bevy_ecs", "source":CRATES_IO, "req":"^0.19.1"}]},
        {"name":"bevy_ecs", "version":version, "source":CRATES_IO, "dependencies":[]},
        {"name":"local-fixture", "source":null, "dependencies":[]}
    ]})
}

#[test]
fn compatible_lockfile_bumps_and_non_bevy_paths_are_allowed() {
    for version in ["0.19.1", "0.19.2"] {
        check(&metadata(version)).unwrap();
    }
}

#[test]
fn bevy_git_path_and_alternate_registry_sources_fail() {
    for source in [
        Value::Null,
        json!("git+https://example.invalid/bevy"),
        json!("registry+https://example.invalid/index"),
    ] {
        let mut resolved = metadata("0.19.1");
        resolved["packages"][1]["source"] = source.clone();
        assert!(check(&resolved).is_err());
        let mut declared = metadata("0.19.1");
        declared["packages"][0]["dependencies"][0]["source"] = source;
        assert!(check(&declared).is_err());
    }
}

#[test]
fn missing_inventory_is_not_a_pass() {
    assert!(check(&json!({"packages":[]})).is_err());
    assert!(check(&json!({})).is_err());
}

#[test]
fn workspace_requirements_preserve_the_bevy_release_floor() {
    for requirement in ["*", "^0.18", "^0.19.0", "=0.19.1", "^0.20"] {
        let mut declared = metadata("0.19.2");
        declared["packages"][0]["dependencies"][0]["req"] = json!(requirement);
        assert!(matches!(check(&declared), Err(Error::Requirement(..))));
    }
}

#[test]
fn transitive_requirements_need_not_match_the_workspace_floor() {
    let mut declared = metadata("0.19.2");
    declared["packages"][1]["dependencies"] = json!([
        {"name":"bevy_platform", "source":CRATES_IO, "req":"^0.19.0"}
    ]);
    check(&declared).unwrap();
}
