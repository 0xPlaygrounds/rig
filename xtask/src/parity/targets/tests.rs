use super::*;

fn metadata() -> Value {
    json!({
        "workspace_members": ["root-id", "companion-id"],
        "packages": [
            {"id":"root-id", "name":"rig", "targets":[
                {"name":"rig", "kind":["lib"], "test":true},
                {"name":"openai", "kind":["test"], "test":true},
                {"name":"bedrock", "kind":["test"], "test":true, "required-features":["bedrock"]},
                {"name":"demo", "kind":["example"], "test":false}
            ]},
            {"id":"companion-id", "name":"companion", "targets":[
                {"name":"companion", "kind":["lib"], "test":true}
            ]}
        ]
    })
}

fn listing() -> Value {
    let mut suites = serde_json::Map::new();
    for (name, kind) in [("rig", "lib"), ("openai", "test"), ("bedrock", "test")] {
        suites.insert(
            name.into(),
            json!({
                "package-name":"rig", "package-id":"root-id", "binary-name":name,
                "kind":kind, "status":"listed", "testcases": {}
            }),
        );
    }
    json!({"rust-suites":suites, "test-count":0})
}

#[test]
fn missing_target_is_visible_even_when_supplied_list_is_internally_consistent() {
    let mut list = listing();
    list["rust-suites"]
        .as_object_mut()
        .unwrap()
        .remove("bedrock");
    let result = reconcile(&metadata(), "rig", &list).unwrap();
    assert_eq!(result["complete_target_registration"], false);
    assert_eq!(result["missing_targets"], json!([["bedrock", "test"]]));
    assert_eq!(result["expected_targets"], 3);
}

#[test]
fn independent_companion_members_cannot_be_lost_from_metadata() {
    let mut input = metadata();
    input["packages"].as_array_mut().unwrap().pop();
    assert!(
        workspace_packages(&input)
            .unwrap_err()
            .to_string()
            .contains("every workspace member")
    );
}

#[test]
fn zero_case_targets_do_not_claim_executed_coverage() {
    let result = reconcile(&metadata(), "rig", &listing()).unwrap();
    assert_eq!(result["complete_target_registration"], true);
    assert_eq!(result["compiled_testcases"], 0);
    assert!(result["execution_result"].is_null());
    assert!(result["parity_verdict"].is_null());
}

#[test]
fn wrong_target_kind_and_duplicate_binary_are_rejected() {
    let mut list = listing();
    list["rust-suites"]["rig"]["kind"] = json!("bin");
    let result = reconcile(&metadata(), "rig", &list).unwrap();
    assert_eq!(result["complete_target_registration"], false);
    assert_eq!(result["unexpected_targets"], json!([["rig", "bin"]]));
    let mut list = listing();
    list["rust-suites"]["duplicate"] = list["rust-suites"]["openai"].clone();
    assert!(
        reconcile(&metadata(), "rig", &list)
            .unwrap_err()
            .to_string()
            .contains("duplicate")
    );
}

#[test]
fn mismatched_checkout_and_skipped_target_are_rejected() {
    let mut list = listing();
    list["rust-suites"]["rig"]["package-id"] = json!("another-checkout-id");
    assert!(
        reconcile(&metadata(), "rig", &list)
            .unwrap_err()
            .to_string()
            .contains("identity")
    );
    let mut list = listing();
    list["rust-suites"]["bedrock"]["status"] = json!("skipped");
    assert!(
        reconcile(&metadata(), "rig", &list)
            .unwrap_err()
            .to_string()
            .contains("skipped")
    );
}

#[test]
fn ignored_cases_are_retained_but_user_filtered_cases_are_rejected() {
    let mut list = listing();
    list["rust-suites"]["openai"]["testcases"]["live"] = json!({
        "ignored":true, "filter-match":{"status":"mismatch", "reason":"ignored"}
    });
    list["test-count"] = json!(1);
    let result = reconcile(&metadata(), "rig", &list).unwrap();
    assert_eq!(result["ignored_testcases"], 1);
    list["rust-suites"]["openai"]["testcases"]["live"]["filter-match"]["reason"] =
        json!("expression");
    assert!(
        reconcile(&metadata(), "rig", &list)
            .unwrap_err()
            .to_string()
            .contains("filtered")
    );
}

#[test]
fn incorrect_total_is_rejected() {
    let mut list = listing();
    list["test-count"] = json!(1);
    assert!(
        reconcile(&metadata(), "rig", &list)
            .unwrap_err()
            .to_string()
            .contains("test-count")
    );
}

#[test]
fn explicitly_test_enabled_examples_and_benches_cannot_be_omitted() {
    let mut input = metadata();
    let mut list = listing();
    for kind in ["example", "bench"] {
        input["packages"][0]["targets"]
            .as_array_mut()
            .unwrap()
            .push(json!({
                "name":kind, "kind":[kind], "test":true
            }));
        let partial = reconcile(&input, "rig", &list).unwrap();
        assert_eq!(partial["complete_target_registration"], false);
        assert_eq!(partial["missing_targets"], json!([[kind, kind]]));
        list["rust-suites"][kind] = json!({
            "package-name":"rig", "package-id":"root-id", "binary-name":kind,
            "kind":kind, "status":"listed", "testcases":{}
        });
        assert_eq!(
            reconcile(&input, "rig", &list).unwrap()["complete_target_registration"],
            true
        );
    }
}
