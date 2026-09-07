use super::*;
use serde_json::json;
use std::fs;
fn catalog() -> Value {
    json!({"schema":1,"configurations":{"root-bedrock":{}},"scenarios":[{"id":"rig::anthropic::original","classification":"agent","source":"xtask/Cargo.toml","configuration":"root-bedrock","ecs":{"binary":"anthropic","test":"native","source":"xtask/Cargo.toml"}}]})
}
fn root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap()
}
fn listing() -> Value {
    json!({"rust-suites":{"rig::anthropic":{"package-name":"rig","binary-name":"anthropic","testcases":{"original":{},"native":{}}}}})
}
#[test]
fn accepts_current_correspondences_without_execution_claims() {
    assert_eq!(
        validate(root(), &catalog(), Some(&listing())).unwrap(),
        (1, 1, 0)
    );
}
#[test]
fn missing_original_and_native_registrations_fail() {
    for name in ["original", "native"] {
        let mut l = listing();
        l["rust-suites"]["rig::anthropic"]["testcases"]
            .as_object_mut()
            .unwrap()
            .remove(name);
        assert!(validate(root(), &catalog(), Some(&l)).is_err());
    }
}
#[test]
fn filtered_listing_fails() {
    let mut l = listing();
    l["rust-suites"]["rig::anthropic"]["testcases"]["original"] =
        json!({"filter-match":{"status":"mismatch"}});
    assert!(validate(root(), &catalog(), Some(&l)).is_err());
}
#[test]
fn duplicates_stale_files_and_historical_verdicts_fail() {
    let mut c = catalog();
    let row = c["scenarios"][0].clone();
    c["scenarios"].as_array_mut().unwrap().push(row);
    assert!(validate(root(), &c, None).is_err());
    let mut c = catalog();
    c["scenarios"][0]["source"] = json!("missing.rs");
    assert!(validate(root(), &c, None).is_err());
    let mut c = catalog();
    c["scenarios"][0]["scoped_parity"] = json!({"status":"verified"});
    assert!(validate(root(), &c, None).is_err());
}
#[test]
fn escaping_paths_fail() {
    assert!(file(root(), "../Cargo.toml").is_err());
    assert!(file(root(), "/etc/passwd").is_err());
}

#[test]
fn ignored_registration_is_not_treated_as_execution() {
    let mut list = listing();
    list["rust-suites"]["rig::anthropic"]["testcases"]["original"] =
        json!({"ignored":true,"filter-match":{"status":"mismatch","reason":"ignored"}});
    assert_eq!(
        validate(root(), &catalog(), Some(&list)).unwrap(),
        (1, 1, 0)
    );
}

#[test]
fn unmapped_source_only_scenarios_remain_explicit() {
    let mut catalog = catalog();
    catalog["scenarios"][0]["ecs"] = Value::Null;
    let mut list = listing();
    list["rust-suites"]["rig::anthropic"]["testcases"]
        .as_object_mut()
        .unwrap()
        .remove("original");
    assert_eq!(validate(root(), &catalog, Some(&list)).unwrap(), (1, 0, 1));
}

#[test]
fn nested_file_references_cannot_skip_validation() {
    for value in [
        json!("missing.md"),
        json!("../outside.md"),
        json!("/etc/passwd"),
        json!({"wrong":"type"}),
    ] {
        let mut catalog = catalog();
        catalog["scenarios"][0]["family_contract"] = value;
        assert!(validate(root(), &catalog, None).is_err());
    }
    let mut catalog = catalog();
    catalog["scenarios"][0]["external_helper_source"] = json!({"role":"missing path"});
    assert!(validate(root(), &catalog, None).is_err());
}

#[test]
fn shared_provider_correspondences_cannot_claim_agent_migration() {
    let mut c = catalog();
    c["scenarios"][0]["ecs"] = Value::Null;
    c["shared_provider_correspondences"] =
        json!([{"original":"rig::anthropic::original","counterpart":"rig::anthropic::original"}]);
    assert!(validate(root(), &c, Some(&listing())).is_err());
    c["scenarios"][0]["classification"] = json!("shared_provider");
    assert_eq!(validate(root(), &c, Some(&listing())).unwrap(), (1, 0, 0));
    c["shared_provider_correspondences"][0]["counterpart"] = json!("rig::anthropic::missing");
    assert!(validate(root(), &c, Some(&listing())).is_err());
}

fn temp_root(name: &str) -> std::path::PathBuf {
    let root =
        std::env::temp_dir().join(format!("rig-xtask-scenarios-{name}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(root.join(CATALOG_DIR)).unwrap();
    fs::create_dir_all(root.join("tests/common")).unwrap();
    fs::create_dir_all(root.join("tests/providers/anthropic")).unwrap();
    fs::write(root.join("tests/common/support.rs"), "").unwrap();
    fs::write(root.join("tests/providers/anthropic/agent.rs"), "").unwrap();
    root
}

fn write_part(root: &Path, name: &str, part: &Value) {
    fs::write(
        root.join(CATALOG_DIR).join(name),
        serde_json::to_vec(part).unwrap(),
    )
    .unwrap();
}

fn row(id: &str, source: &str) -> Value {
    json!({"id":id,"classification":"infrastructure","source":source,"configuration":"root-bedrock"})
}

#[test]
fn scenario_rows_are_filed_by_their_source_tree() {
    assert_eq!(
        expected_file("tests/providers/anthropic/cassette/agent.rs"),
        "anthropic.json"
    );
    assert_eq!(expected_file("tests/providers/xai/agent.rs"), "xai.json");
    assert_eq!(expected_file("tests/common/support.rs"), "common.json");
    assert_eq!(expected_file("tests/core.rs"), "common.json");
}

#[test]
fn catalog_directory_merges_and_rejects_misplaced_rows() {
    let root = temp_root("merge");
    write_part(
        &root,
        "shared.json",
        &json!({"schema":1,"purpose":"test","configurations":{"root-bedrock":{}}}),
    );
    write_part(
        &root,
        "anthropic.json",
        &json!({"schema":1,"scenarios":[row("rig::anthropic::a", "tests/providers/anthropic/agent.rs")]}),
    );
    write_part(
        &root,
        "common.json",
        &json!({"schema":1,"scenarios":[row("rig::core::b", "tests/common/support.rs")]}),
    );
    let catalog = load(&root).unwrap();
    assert_eq!(catalog["scenarios"].as_array().unwrap().len(), 2);
    assert_eq!(catalog["purpose"], "test");
    assert_eq!(validate(&root, &catalog, None).unwrap(), (2, 0, 0));

    write_part(
        &root,
        "common.json",
        &json!({"schema":1,"scenarios":[row("rig::anthropic::c", "tests/providers/anthropic/agent.rs")]}),
    );
    let error = load(&root).unwrap_err().to_string();
    assert!(error.contains("belongs in anthropic.json"), "{error}");
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn catalog_directory_rejects_duplicate_configurations_unknown_keys_and_schemas() {
    let root = temp_root("shape");
    write_part(
        &root,
        "shared.json",
        &json!({"schema":1,"configurations":{"root-bedrock":{}}}),
    );
    write_part(
        &root,
        "common.json",
        &json!({"schema":1,"configurations":{"root-bedrock":{}}}),
    );
    assert!(
        load(&root)
            .unwrap_err()
            .to_string()
            .contains("duplicate configuration")
    );
    write_part(&root, "common.json", &json!({"schema":1,"verdicts":[]}));
    assert!(
        load(&root)
            .unwrap_err()
            .to_string()
            .contains("unknown catalog key")
    );
    write_part(&root, "common.json", &json!({"schema":2}));
    assert!(load(&root).unwrap_err().to_string().contains("schema"));
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn maintained_catalog_directory_validates_its_file_references() {
    let catalog = load(root()).unwrap();
    let (originals, natives, unlisted) = validate(root(), &catalog, None).unwrap();
    assert!(
        originals > natives,
        "{originals} originals, {natives} natives"
    );
    assert_eq!(unlisted, 0);
}
