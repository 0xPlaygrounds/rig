use super::*;
use serde_json::json;
use std::sync::atomic::{AtomicUsize, Ordering};

static NEXT: AtomicUsize = AtomicUsize::new(0);
struct Fixture {
    root: PathBuf,
    manifest: Value,
    review: Value,
}
impl Fixture {
    fn new() -> Self {
        let root = std::env::temp_dir().join(format!(
            "rig-review-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&root).expect("new fixture");
        fs::write(root.join("original.rs"), "original").expect("source");
        fs::write(root.join("native.rs"), "native").expect("source");
        let sources = json!({"original.rs":hash(b"original"),"native.rs":hash(b"native")});
        let index = store(&root.join("provenance"), &sources).expect("index");
        let batch = store(&root.join("batches"),&json!({"cells":[{"original":"rig::provider$original","native":"rig::provider$native"}],"features":[]})).expect("batch");
        let mut runs = json!({});
        for surface in ["original", "native"] {
            let id = format!("rig::provider${surface}");
            runs[surface] = store(&root.join("runs"),&json!({"batch":batch,"surface":surface,"revision":"base","status":"executed","exit_code":0,"source_index":index,"source_index_after":index,"results":[{"id":id,"status":"ok","expected":"ok"}]})).expect("run").into();
        }
        let manifest = json!({"baseline_revision":"base","scenarios":[{"id":"rig::provider::original","classification":"agent","source":"original.rs"}]});
        let review = json!({"baseline_revision":"base","kind":"parity","status":"accepted","reviewer":"independent","configuration":"fixture","limits":"synthetic tool test","sources":sources,"runs":runs,"cells":[{"id":"rig::provider::original","original":"rig::provider$original","native":"rig::provider$native","fields":{"ecs":{"binary":"provider","test":"native","source":"native.rs"},"assertion_mappings":[{"comparison":"explicit reviewed fixture decision"}]}}]});
        Self {
            root,
            manifest,
            review,
        }
    }
    fn apply(&mut self) -> Result<()> {
        apply(
            &mut self.manifest,
            &self.review,
            &self.root,
            &self.root,
            "review.json",
        )
    }
    fn edit_run(&mut self, surface: &str, edit: impl FnOnce(&mut Value)) {
        let name = self.review["runs"][surface].as_str().expect("run");
        let mut run = read(&self.root.join("runs").join(name)).expect("run");
        edit(&mut run);
        self.review["runs"][surface] = store(&self.root.join("runs"), &run)
            .expect("changed run")
            .into();
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}

#[test]
fn accepted_source_bound_pair_applies_explicit_decision() {
    let mut fixture = Fixture::new();
    fixture.apply().expect("accepted review");
    assert_eq!(
        fixture.manifest["scenarios"][0]["scoped_parity"]["status"],
        "verified"
    );
}

#[test]
fn incompatible_pairs_and_invalid_decisions_never_partially_apply() {
    for case in 0..9 {
        let mut fixture = Fixture::new();
        match case {
            0 => {
                let row = fixture.manifest["scenarios"][0].clone();
                fixture.manifest["scenarios"]
                    .as_array_mut()
                    .expect("rows")
                    .push(row);
            }
            1 => fixture.edit_run("original", |run| run["surface"] = "native".into()),
            2 => {
                let batch=store(&fixture.root.join("batches"),&json!({"cells":[{"original":"rig::provider$original","native":"rig::provider$native"}],"features":["different"]})).expect("batch");
                fixture.edit_run("native", |run| run["batch"] = batch.into());
            }
            3 => fixture.edit_run("native", |run| {
                run["results"][0]["id"] = "rig::provider$unselected".into()
            }),
            4 => {
                fs::write(fixture.root.join("native.rs"), "changed").expect("drift");
            }
            5 => {
                let name = fixture.review["runs"]["native"].as_str().expect("run");
                fs::write(fixture.root.join("runs").join(name), "corrupt").expect("corrupt");
            }
            6 => fixture.review["cells"][0]["id"] = "unknown".into(),
            7 => {
                let cell = fixture.review["cells"][0].clone();
                fixture.review["cells"]
                    .as_array_mut()
                    .expect("cells")
                    .push(cell);
            }
            _ => fixture.edit_run("native", |run| {
                run["source_index_after"] = "different".into()
            }),
        }
        let before = fixture.manifest.clone();
        assert!(fixture.apply().is_err(), "case {case}");
        assert_eq!(fixture.manifest, before, "case {case} partially applied");
    }
}

#[test]
fn classification_decisions_cannot_assign_parity() {
    let mut fixture = Fixture::new();
    fixture.review["kind"] = "classification".into();
    fixture.review["cells"][0]["fields"] = json!({"classification":"shared_provider","classification_reason":"direct completion call"});
    fixture.apply().expect("classification accepted");
    assert!(fixture.manifest["scenarios"][0]["scoped_parity"].is_null());
    fixture.review["cells"][0]["fields"]["status"] = "executed".into();
    let before = fixture.manifest.clone();
    assert!(fixture.apply().is_err());
    assert_eq!(fixture.manifest, before);
}

#[test]
fn unaccepted_review_cannot_promote_a_mapping() {
    let mut manifest = json!({"baseline_revision":"base","scenarios":[]});
    let before = manifest.clone();
    assert!(
        apply(
            &mut manifest,
            &json!({"status":"pending"}),
            Path::new("."),
            Path::new("."),
            "review"
        )
        .is_err()
    );
    assert_eq!(manifest, before);
}

#[test]
fn classification_review_cannot_omit_source_evidence() {
    let mut manifest = json!({"baseline_revision":"base","scenarios":[]});
    let before = manifest.clone();
    assert!(apply(&mut manifest,&json!({"status":"accepted","reviewer":"independent","kind":"classification","baseline_revision":"base","sources":{},"cells":[]}),Path::new("."),Path::new("."),"review").is_err());
    assert_eq!(manifest, before);
}

#[test]
fn incomplete_assertion_and_fixture_inventories_do_not_apply() {
    for case in 0..4 {
        let mut fixture = Fixture::new();
        let fields = &mut fixture.review["cells"][0]["fields"];
        match case {
            0 => fields["inventory_complete"] = true.into(),
            1 => {
                fields["inventory_complete"] = true.into();
                fields["assertion_sources"] = json!([{"path":"unreviewed.rs"}]);
            }
            2 => fields["fixtures"] = json!([{"path":"unreviewed.yaml","sha256":"bogus"}]),
            _ => fields["fixtures"] = json!({"path":"malformed"}),
        }
        let before = fixture.manifest.clone();
        assert!(fixture.apply().is_err());
        assert_eq!(fixture.manifest, before);
    }
}

#[test]
fn reviewed_inventory_fields_are_preserved_for_discovery_validation() {
    let mut fixture = Fixture::new();
    let fields = &mut fixture.review["cells"][0]["fields"];
    fields["inventory_complete"] = true.into();
    fields["assertion_sources"] =
        json!([{"path":"original.rs","symbol":"original","direct_assertions":[]}]);
    fields["fixture_inventory_complete"] = true.into();
    fields["fixtures"] = json!([]);
    fixture.apply().expect("reviewed inventory");
    assert_eq!(fixture.manifest["scenarios"][0]["inventory_complete"], true);
}
