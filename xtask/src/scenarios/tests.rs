use super::*;
use serde_json::json;
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
