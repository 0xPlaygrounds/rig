use super::*;
use std::sync::atomic::{AtomicUsize, Ordering};

static NEXT: AtomicUsize = AtomicUsize::new(0);

struct Fixture {
    root: PathBuf,
    discovery: Value,
    candidate: Value,
    evidence: Value,
    manifest: Value,
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}
impl Fixture {
    fn new() -> Self {
        let root = std::env::temp_dir().join(format!(
            "rig-parity-unit-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&root).expect("new temporary directory");
        fs::write(root.join("original.rs"), "original test and assertions").expect("source");
        fs::write(root.join("ecs.rs"), "native test and assertions").expect("native");
        fs::write(root.join("fixture.yaml"), "recorded provider exchange").expect("fixture");
        let discovery = json!({"evidence_kind":"discovery_not_coverage", "source_revision":"baseline", "source_status":"", "listing_sha256":"listing", "unresolved_compiled_tests":[], "unlisted_provider_targets":[],
            "tests":[{"binary":"provider", "test":"original", "source":"original.rs", "source_sha256":hash(b"original test and assertions"), "line":1, "compiled":true, "attributes":[], "assertions":[], "classification":"unclassified"}]});
        let candidate = json!({"evidence_kind":"discovery_not_coverage", "tests":[{"binary":"provider", "test":"native", "source":"ecs.rs", "source_sha256":hash(b"native test and assertions"), "compiled":true}]});
        let evidence = json!({"schema":1, "baseline_listing_sha256":"listing", "configurations":{"unit-config":{"package":"rig", "features":[], "listing_command":"unit fixture"}}});
        let manifest = initialize(&discovery, &evidence).expect("initialize");
        Self {
            root,
            discovery,
            candidate,
            evidence,
            manifest,
        }
    }
    fn inspect(&self) -> Result<Value> {
        inspect(
            &self.manifest,
            &self.discovery,
            &self.root,
            &self.root,
            &self.candidate,
            &self.evidence,
        )
    }
    fn row(&mut self) -> &mut Value {
        &mut self.manifest["scenarios"][0]
    }
    fn mapped(&mut self) {
        let row = self.row();
        row["classification"] = json!("agent");
        row["classification_reason"] = json!("original agent runner");
        row["evidence_scope"] = json!("cassette_agent");
        row["ecs"] = json!({"binary":"provider", "test":"native", "source":"ecs.rs"});
        row["fixtures"] =
            json!([{"path":"fixture.yaml", "sha256":hash(b"recorded provider exchange")}]);
    }
    fn completed(&mut self) {
        let row = self.row();
        row["classification"] = json!("infrastructure");
        row["classification_reason"] = json!("synthetic utility test");
        row["inventory_complete"] = json!(true);
        row["fixture_inventory_complete"] = json!(true);
        row["assertion_sources"] =
            json!([{"path":"original.rs", "symbol":"original", "direct_assertions":[]}]);
    }
    fn helper(&mut self) {
        self.row()["assertion_sources"]
            .as_array_mut()
            .expect("anchors")
            .push(json!({"path":"original.rs", "symbol":"check_response", "direct_assertions":[]}));
        self.discovery["source_helpers"] = json!([{"source":"original.rs", "source_sha256":hash(b"original test and assertions"), "binary":"provider", "function":"check_response", "assertions":[]}]);
    }
    fn rejects(&self, expected: &str) {
        let error = self.inspect().expect_err("must reject").to_string();
        assert!(
            error.contains(expected),
            "{error} should contain {expected}"
        );
    }
}

#[test]
fn mapping_is_not_completed_inventory_or_execution() {
    let mut f = Fixture::new();
    assert_eq!(f.inspect().expect("report")["inventory_complete"], false);
    f.mapped();
    let report = f.inspect().expect("report");
    assert_eq!(report["counts"]["native_mappings"], 1);
    assert_eq!(
        report["counts"]["classified_agent_scopes"]["cassette_agent"],
        1
    );
    assert_eq!(report["inventory_complete"], false);
}

#[test]
fn missing_and_duplicate_scenarios_fail() {
    let mut f = Fixture::new();
    let row = f.row().clone();
    f.manifest["scenarios"] = json!([]);
    f.rejects("missing manifest scenarios");
    f.manifest["scenarios"] = json!([row, row]);
    f.rejects("duplicate manifest ID");
}

#[test]
fn stale_compilation_and_fixture_drift_fail() {
    let mut f = Fixture::new();
    f.mapped();
    f.candidate["tests"][0]["compiled"] = json!(false);
    f.rejects("empty candidate compiled");
    f.candidate["tests"][0]["compiled"] = json!(true);
    fs::write(f.root.join("ecs.rs"), "changed").expect("source");
    f.rejects("candidate source changed");
    f.candidate["tests"][0]["source_sha256"] = json!(hash(b"changed"));
    fs::write(f.root.join("fixture.yaml"), "weakened").expect("fixture");
    f.rejects("fixture drift");
}

#[test]
fn provider_only_rows_cannot_receive_agent_credit() {
    let mut f = Fixture::new();
    f.mapped();
    f.row()["classification"] = json!("shared_provider");
    f.rejects("non-agent scenario receives");
}

#[test]
fn completed_inventory_requires_unchanged_direct_assertions() {
    let mut f = Fixture::new();
    f.completed();
    let anchors = f.row()["assertion_sources"].take();
    f.rejects("invalid array assertion_sources");
    f.row()["assertion_sources"] = anchors;
    assert_eq!(f.inspect().expect("report")["inventory_complete"], true);
    f.discovery["tests"][0]["assertions"] =
        json!([{"line":3,"assertion":"assert_eq!(actual, expected)"}]);
    f.rejects("direct assertion inventory drift");
}

#[test]
fn complete_and_partial_inventories_validate_helper_assertions() {
    for complete in [true, false] {
        let mut f = Fixture::new();
        f.completed();
        f.helper();
        f.row()["inventory_complete"] = json!(complete);
        assert_eq!(f.inspect().expect("report")["inventory_complete"], complete);
        f.discovery["source_helpers"][0]["assertions"] =
            json!([{"line":4,"assertion":"assert!(response_valid)"}]);
        f.rejects("helper assertion inventory drift");
    }
}

#[test]
fn configuration_source_provenance_is_checked() {
    let mut f = Fixture::new();
    f.row()["configuration_sources"] = json!([{"path":"original.rs", "symbol":"defaults", "sha256":hash(b"original test and assertions")}]);
    f.inspect().expect("report");
    f.row()["configuration_sources"][0]["sha256"] = json!("stale");
    f.rejects("configuration source drift");
}

#[test]
fn feature_and_registration_evidence_cannot_drift() {
    let mut f = Fixture::new();
    f.manifest["configurations"]["unit-config"]["features"] = json!(["unrecorded"]);
    f.rejects("configuration evidence drift");
    f.discovery["unresolved_compiled_tests"] = json!([{"binary":"provider", "test":"generated"}]);
    assert!(initialize(&f.discovery, &f.evidence).is_err());
}

#[test]
fn baseline_revision_source_and_declared_gates_cannot_drift() {
    let mut f = Fixture::new();
    f.manifest["baseline_revision"] = json!("other");
    f.rejects("revision mismatch");
    f.manifest["baseline_revision"] = json!("baseline");
    f.row()["declared_gates"] = json!(["different feature"]);
    f.rejects("configuration drift");
    f.row()["declared_gates"] = json!([]);
    fs::write(f.root.join("original.rs"), "weakened").expect("source");
    f.rejects("baseline source drift");
}
