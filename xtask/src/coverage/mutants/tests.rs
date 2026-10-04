use super::*;

#[test]
fn an_identity_drops_the_position_and_numbers_repeats() {
    assert_eq!(
        strip_position("crates/a/src/x.rs:66:5: replace f -> bool with true"),
        "crates/a/src/x.rs: replace f -> bool with true"
    );
    assert_eq!(strip_position("no position"), "no position");
    let names: Vec<String> = [
        "crates/a/src/x.rs:1:2: delete ! in f",
        "crates/a/src/x.rs:9:4: delete ! in f",
        "crates/a/src/x.rs:12:1: replace g with ()",
    ]
    .into_iter()
    .map(str::to_owned)
    .collect();
    let ids: Vec<String> = identities(&names).into_iter().map(|(_, id)| id).collect();
    assert_eq!(
        ids,
        [
            "crates/a/src/x.rs: delete ! in f",
            "crates/a/src/x.rs: delete ! in f #2",
            "crates/a/src/x.rs: replace g with ()"
        ]
    );
}

#[test]
fn the_sample_is_a_fixed_hash_residue() {
    assert!(selected("anything", 1));
    let chosen: Vec<bool> = (0..64)
        .map(|n| selected(&format!("mutant {n}"), 4))
        .collect();
    let again: Vec<bool> = (0..64)
        .map(|n| selected(&format!("mutant {n}"), 4))
        .collect();
    assert_eq!(chosen, again);
    let count = chosen.iter().filter(|chosen| **chosen).count();
    assert!((4..=28).contains(&count), "{count}");
}

#[test]
fn killers_are_read_from_nextest_status_lines() {
    let log = "\
        PASS [   0.004s] (1/3) rig-core a::passes
        FAIL [   0.012s] (2/3) rig-core history::tests::drops_orphans
     TIMEOUT [  60.000s] rig-core::driver_adoption slow
     SIGSEGV [   0.100s] (3/3) rig-ecs systems::tests::crash
        FAIL [   0.012s] (2/3) rig-core history::tests::drops_orphans
   Summary [   1.000s] 3 tests run
";
    assert_eq!(
        killers(log).into_iter().collect::<Vec<_>>(),
        [
            "rig-core history::tests::drops_orphans",
            "rig-core::driver_adoption slow",
            "rig-ecs systems::tests::crash"
        ]
    );
}

fn tested(outcome: Outcome, killers: &[&str]) -> Tested {
    Tested::new(outcome, killers.iter().map(|k| (*k).to_owned()).collect())
}

fn set(entries: &[(&str, Tested)]) -> KillSet {
    KillSet {
        sample: 8,
        mutants: entries
            .iter()
            .map(|(id, tested)| ((*id).to_owned(), tested.clone()))
            .collect(),
    }
}

#[test]
fn the_baseline_round_trips() {
    let baseline = set(&[
        (
            "x.rs: a",
            tested(Outcome::Caught, &["rig-core t1", "rig-core t2"]),
        ),
        ("x.rs: b", tested(Outcome::Missed, &[])),
        ("x.rs: c", tested(Outcome::Timeout, &[])),
        ("x.rs: d", tested(Outcome::Unviable, &[])),
    ]);
    let text = render(&baseline);
    assert!(text.starts_with(&format!(
        "sample\t8\n{HEADER}\nx.rs: a\tcaught\t2\trig-core t1, rig-core t2\n"
    )));
    assert_eq!(parse(&text).unwrap(), baseline);
    assert!(parse("mutant\toutcome\n").is_err());
    assert!(parse(&format!("sample\t8\n{HEADER}\nx\tsurvived\t0\t\n")).is_err());
    assert!(parse(&format!("sample\t8\n{HEADER}\nx\tcaught\tmany\t\n")).is_err());
}

#[test]
fn a_row_names_the_first_few_failed_tests_and_counts_the_rest() {
    let many = tested(Outcome::Caught, &["e", "d", "c", "b", "a"]);
    assert_eq!(many.failed, 5);
    assert_eq!(many.killers.iter().collect::<Vec<_>>(), ["a", "b", "c"]);
}

#[test]
fn only_a_killed_mutant_that_now_survives_fails() {
    let baseline = set(&[
        ("caught then missed", tested(Outcome::Caught, &["t"])),
        ("timeout then caught", tested(Outcome::Timeout, &[])),
        ("missed then missed", tested(Outcome::Missed, &[])),
        ("caught then gone", tested(Outcome::Caught, &["t"])),
    ]);
    let current = set(&[
        ("caught then missed", tested(Outcome::Missed, &[])),
        ("timeout then caught", tested(Outcome::Caught, &["u"])),
        ("missed then missed", tested(Outcome::Missed, &[])),
    ]);
    assert_eq!(
        survivors(&baseline, &current),
        ["mutant survives: caught then missed (missed)"]
    );
}

#[test]
fn every_group_mutates_its_own_package() {
    for group in GROUPS {
        assert!(!group.files.is_empty());
        assert!(
            group
                .files
                .iter()
                .all(|file| file.starts_with(&format!("crates/{}/src/", group.package))),
            "{}",
            group.package
        );
    }
    // rig-core's other targets include a nested Cargo build.
    let core = &groups(None).unwrap()[0];
    assert_eq!(core.package, "rig-core");
    assert_eq!(core.test_args.first(), Some(&"--lib"));
}

#[test]
fn a_package_restriction_selects_its_groups_and_their_mutants() {
    let picked = groups(Some(&["rig-ecs".to_owned()])).unwrap();
    assert_eq!(picked.len(), 1);
    assert!(in_groups(
        "crates/rig-ecs/src/systems/mod.rs: delete ! in f",
        &picked
    ));
    assert!(!in_groups(
        "crates/rig-core/src/a.rs: delete ! in f",
        &picked
    ));
    assert!(groups(Some(&["rig-nope".to_owned()])).is_err());
    assert_eq!(groups(None).unwrap().len(), GROUPS.len());
}
