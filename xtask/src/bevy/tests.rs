use super::*;
use serde_json::json;

fn metadata(version: &str) -> Value {
    json!({"workspace_members":["agent_no_tokio"], "packages": [
        {"id":"agent_no_tokio", "name":"agent_no_tokio", "source":null, "dependencies":[{"name":"bevy_tasks", "source":CRATES_IO, "req":"=0.20.0-rc.2"}]},
        {"name":"bevy_tasks", "version":version, "source":CRATES_IO, "dependencies":[]},
        {"name":"local-fixture", "source":null, "dependencies":[]}
    ]})
}

#[test]
fn the_pinned_version_and_non_bevy_paths_are_allowed() {
    check(&metadata("0.20.0-rc.2")).unwrap();
}

#[test]
fn bevy_git_path_and_alternate_registry_sources_fail() {
    for source in [
        Value::Null,
        json!("git+https://example.invalid/bevy"),
        json!("registry+https://example.invalid/index"),
    ] {
        let mut resolved = metadata("0.20.0-rc.2");
        resolved["packages"][1]["source"] = source.clone();
        assert!(check(&resolved).is_err());
        let mut declared = metadata("0.20.0-rc.2");
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
fn workspace_requirements_preserve_the_bevy_pin() {
    for requirement in ["*", "^0.19.1", "^0.20.0-rc.2", "=0.20.0-rc.1", "^0.20"] {
        let mut declared = metadata("0.20.0-rc.2");
        declared["packages"][0]["dependencies"][0]["req"] = json!(requirement);
        assert!(matches!(check(&declared), Err(Error::Requirement(..))));
    }
}

#[test]
fn transitive_requirements_need_not_match_the_workspace_pin() {
    let mut declared = metadata("0.20.0-rc.2");
    declared["packages"][1]["dependencies"] = json!([
        {"name":"bevy_platform", "source":CRATES_IO, "req":"^0.20.0-rc.1"}
    ]);
    check(&declared).unwrap();
}

#[test]
fn the_bevy_umbrella_crate_is_checked_like_its_parts() {
    let mut declared = metadata("0.20.0-rc.2");
    declared["packages"][0]["dependencies"][0]["name"] = json!("bevy");
    check(&declared).unwrap();
    declared["packages"][0]["dependencies"][0]["req"] = json!("^0.19.1");
    assert!(matches!(check(&declared), Err(Error::Requirement(..))));
    declared["packages"][0]["dependencies"][0]["req"] = json!("=0.20.0-rc.2");
    declared["packages"][0]["dependencies"][0]["source"] =
        json!("git+https://example.invalid/bevy");
    assert!(matches!(check(&declared), Err(Error::Source(..))));
}
