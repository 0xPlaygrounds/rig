use super::*;
use anyhow::{Context, ensure};
use std::{
    collections::BTreeSet,
    sync::atomic::{AtomicUsize, Ordering},
};

static BODY_CALLS: AtomicUsize = AtomicUsize::new(0);

fn test(name: &str, scenarios: Vec<Scenario>) -> Test {
    Test {
        name: name.into(),
        scenarios,
        ignored: false,
    }
}
fn inventory(tests: Vec<Test>) -> Inventory {
    Inventory {
        tests,
        families: vec![],
    }
}
fn fixture(root: &std::path::Path, id: &str) -> std::io::Result<()> {
    let path = root.join(format!("{id}.yaml"));
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, "fixture bytes")
}
fn check(inventory: &Inventory, root: &std::path::Path) -> Result<(), Error> {
    validate(inventory, root, &BTreeSet::from(["example".into()]))
}
fn scenario_mut(inventory: &mut Inventory, index: usize) -> anyhow::Result<&mut Scenario> {
    inventory
        .tests
        .get_mut(index)
        .and_then(|t| t.scenarios.first_mut())
        .context("test scenario")
}

cassette_test! {
    scenarios: [Scenario::live("example/live")];
    #[allow(dead_code)]
    async fn registered_without_execution() {
        BODY_CALLS.fetch_add(1, Ordering::SeqCst);
    }
}

#[test]
fn compiled_registration_has_exact_identity_and_scenarios() {
    let entries = tests();
    let entry = entries
        .iter()
        .find(|entry| entry.name == "tests::registered_without_execution");
    assert_eq!(
        entry.map(|entry| entry.scenarios.as_slice()),
        Some([Scenario::live("example/live")].as_slice())
    );
    assert!(entry.is_some_and(|entry| !entry.ignored));
    assert_eq!(BODY_CALLS.load(Ordering::SeqCst), 0);
}

#[test]
fn ignored_attribute_is_structural() {
    const {
        assert!(cassette_ignored!(
            [tokio::test][ignore = "needs first capture"]
        ));
        assert!(!cassette_ignored!([allow(dead_code)]));
    }
}

#[test]
fn fixture_ownership_is_checked_in_both_directions() -> anyhow::Result<()> {
    let root = assert_fs::TempDir::new()?;
    let declarations = inventory(vec![test("cell", vec![Scenario::live("example/cell")])]);
    ensure!(check(&declarations, root.path()).is_err());
    fixture(root.path(), "example/cell")?;
    check(&declarations, root.path())?;
    fixture(root.path(), "example/orphan")?;
    ensure!(check(&declarations, root.path()).is_err());
    Ok(())
}

#[test]
fn missing_capture_requires_explicit_allowance_not_ignore_status() -> anyhow::Result<()> {
    let root = assert_fs::TempDir::new()?;
    let mut cell = test("cell", vec![Scenario::live("example/cell")]);
    cell.ignored = true;
    let mut declarations = inventory(vec![cell]);
    ensure!(check(&declarations, root.path()).is_err());
    scenario_mut(&mut declarations, 0)?.missing = Some("No capture yet".into());
    check(&declarations, root.path())?;
    authorize(&declarations.tests, root.path(), "example/cell", None)?;
    fixture(root.path(), "example/cell")?;
    ensure!(
        check(&declarations, root.path()).is_err(),
        "retire first-capture allowance after capture"
    );
    Ok(())
}

#[test]
fn declaration_conflicts_and_duplicates_fail_closed() -> anyhow::Result<()> {
    let root = assert_fs::TempDir::new()?;
    fixture(root.path(), "example/cell")?;
    let cell = test("first", vec![Scenario::live("example/cell")]);
    ensure!(check(&inventory(vec![cell.clone(), cell.clone()]), root.path()).is_err());
    let duplicate = test(
        "first",
        vec![
            Scenario::live("example/cell"),
            Scenario::live("example/cell"),
        ],
    );
    ensure!(check(&inventory(vec![duplicate]), root.path()).is_err());
    let conflict = test(
        "second",
        vec![Scenario::synthetic("example/cell", "handmade")],
    );
    ensure!(check(&inventory(vec![cell.clone(), conflict]), root.path()).is_err());
    check(
        &inventory(vec![
            cell,
            test("second", vec![Scenario::live("example/cell")]),
        ]),
        root.path(),
    )?;
    Ok(())
}

#[test]
fn malformed_ids_and_empty_policy_metadata_are_rejected() -> anyhow::Result<()> {
    let root = assert_fs::TempDir::new()?;
    for id in [
        "../escape",
        "example/../escape",
        "example//cell",
        "example/cell.yaml",
        "example/",
    ] {
        ensure!(
            check(
                &inventory(vec![test(
                    "cell",
                    vec![Scenario::live(id).missing("first capture")]
                )]),
                root.path()
            )
            .is_err()
        );
    }
    for scenario in [
        Scenario::live("example/cell").missing(""),
        Scenario::synthetic("example/cell", ""),
    ] {
        ensure!(check(&inventory(vec![test("cell", vec![scenario])]), root.path()).is_err());
    }
    ensure!(check(&Inventory::default(), root.path()).is_err());
    Ok(())
}

#[test]
fn derived_dependencies_are_valid_acyclic_and_not_recordable() -> anyhow::Result<()> {
    let root = assert_fs::TempDir::new()?;
    fixture(root.path(), "example/source")?;
    fixture(root.path(), "example/derived")?;
    let derived = Scenario::derived(
        "example/derived",
        &["example/source"],
        "corrupted",
        "mutate source",
    );
    let mut declarations = inventory(vec![
        test("source", vec![Scenario::live("example/source")]),
        test("derived", vec![derived]),
    ]);
    check(&declarations, root.path())?;
    ensure!(authorize(&declarations.tests, root.path(), "example/derived", None).is_err());
    ensure!(plan(&declarations, root.path(), Some("example/derived"), false).is_err());
    *scenario_mut(&mut declarations, 0)? = Scenario::derived(
        "example/source",
        &["example/derived"],
        "cycle",
        "mutate derived",
    );
    ensure!(check(&declarations, root.path()).is_err());
    *scenario_mut(&mut declarations, 0)? = Scenario::live("example/source");
    for source in ["example/derived", "example/unknown"] {
        *scenario_mut(&mut declarations, 1)? =
            Scenario::derived("example/derived", &[source], "bad source", "mutate source");
        ensure!(check(&declarations, root.path()).is_err());
    }
    Ok(())
}

#[test]
fn scripted_dependencies_must_be_declared() -> anyhow::Result<()> {
    let root = assert_fs::TempDir::new()?;
    fixture(root.path(), "example/source")?;
    let mut declarations = inventory(vec![test("source", vec![Scenario::live("example/source")])]);
    declarations.families.push(Family {
        name: "example/faults".into(),
        sources: vec!["example/source".into()],
    });
    check(&declarations, root.path())?;
    declarations
        .families
        .first_mut()
        .context("family")?
        .sources
        .push("example/unknown".into());
    ensure!(check(&declarations, root.path()).is_err());
    Ok(())
}

#[test]
fn scopes_never_authorize_unknown_or_unselected_sessions() -> anyhow::Result<()> {
    let root = assert_fs::TempDir::new()?;
    fixture(root.path(), "example/first")?;
    fixture(root.path(), "example/second")?;
    let tests = vec![test(
        "cell",
        vec![
            Scenario::live("example/first"),
            Scenario::live("example/second"),
        ],
    )];
    let scope = BTreeSet::from(["example/first".into()]);
    authorize(&tests, root.path(), "example/first", Some(&scope))?;
    ensure!(authorize(&tests, root.path(), "example/second", Some(&scope)).is_err());
    ensure!(authorize(&tests, root.path(), "example/unknown", None).is_err());
    Ok(())
}

#[test]
fn multi_session_plans_never_silently_expand_a_partial_selection() -> anyhow::Result<()> {
    let root = assert_fs::TempDir::new()?;
    fixture(root.path(), "example/first")?;
    fixture(root.path(), "example/second")?;
    let declarations = inventory(vec![test(
        "cell",
        vec![
            Scenario::live("example/first"),
            Scenario::live("example/second"),
        ],
    )]);
    ensure!(plan(&declarations, root.path(), Some("example/first"), false).is_err());
    let planned = plan(&declarations, root.path(), None, false)?;
    ensure!(planned.len() == 1);
    ensure!(
        planned.first().context("recording")?.scenarios
            == BTreeSet::from(["example/first".into(), "example/second".into()])
    );
    Ok(())
}

#[test]
fn mixed_live_and_forbidden_tests_cannot_record_partially() -> anyhow::Result<()> {
    let root = assert_fs::TempDir::new()?;
    fixture(root.path(), "example/live")?;
    fixture(root.path(), "example/fault")?;
    let declarations = inventory(vec![test(
        "mixed",
        vec![
            Scenario::live("example/live"),
            Scenario::synthetic("example/fault", "handmade"),
        ],
    )]);
    ensure!(plan(&declarations, root.path(), None, false).is_err());
    ensure!(plan(&declarations, root.path(), Some("example/live"), false).is_err());
    Ok(())
}

#[test]
fn ignored_tests_need_explicit_selection_and_repeated_producers_are_deduplicated()
-> anyhow::Result<()> {
    let root = assert_fs::TempDir::new()?;
    let scenario = Scenario::live("example/first").missing("first capture");
    let mut first = test("first", vec![scenario.clone()]);
    first.ignored = true;
    let declarations = inventory(vec![first]);
    ensure!(plan(&declarations, root.path(), None, false).is_err());
    let planned = plan(&declarations, root.path(), Some("example/first"), false)?;
    ensure!(planned.first().context("recording")?.ignored);
    ensure!(plan(&declarations, root.path(), None, true)?.len() == 1);
    let declarations = inventory(vec![
        test("first", vec![scenario.clone()]),
        test("second", vec![scenario]),
    ]);
    ensure!(plan(&declarations, root.path(), None, false)?.len() == 1);
    ensure!(plan(&declarations, root.path(), Some("example/unknown"), false).is_err());
    Ok(())
}
