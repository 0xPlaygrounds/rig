use super::*;

#[test]
fn production_is_crate_source_without_tests_or_helpers() {
    for path in [
        "src/lib.rs",
        "crates/rig-core/src/completion/history.rs",
        "crates/rig-agent/src/run/mod.rs",
    ] {
        assert!(is_production(path), "{path}");
    }
    for path in [
        "crates/rig-core/src/completion/history/tests.rs",
        "crates/rig-cassette/src/http/try_session_tests.rs",
        "crates/rig-core/src/test_utils/history.rs",
        "crates/rig-http/src/test_utils.rs",
        "crates/rig-rmcp/src/tests/dispatch.rs",
        "crates/rig-core/src/loaders/test_fixtures.rs",
        "crates/rig-core/tests/driver_adoption.rs",
        "crates/rig-cassette/tests/minimal/src/lib.rs",
        "test-support/rig-test-support/src/lib.rs",
        "examples/agent/src/main.rs",
        "crates/rig-core/src/data.json",
    ] {
        assert!(!is_production(path), "{path}");
    }
}

const REPORT: &str = "\
SF:/repo/crates/rig-core/src/a.rs
DA:1,3
DA:2,0
DA:4,1
BRDA:2,0,0,1
BRDA:2,0,1,0
BRDA:4,0,0,-
end_of_record
SF:/repo/crates/rig-core/src/a/tests.rs
DA:1,9
end_of_record
SF:/elsewhere/src/b.rs
DA:1,9
end_of_record
";

fn report() -> BTreeMap<String, FileCoverage> {
    parse_lcov(REPORT, Path::new("/repo"))
}

#[test]
fn lcov_keeps_production_files_under_the_root() {
    let files = report();
    assert_eq!(
        files.keys().collect::<Vec<_>>(),
        ["crates/rig-core/src/a.rs"]
    );
    let a = &files["crates/rig-core/src/a.rs"];
    assert_eq!(
        a.counts(),
        Counts {
            lines: (2, 3),
            branches: (1, 3)
        }
    );
}

#[test]
fn a_wrapped_count_is_no_evidence_of_execution() {
    let report = "\
SF:/repo/crates/rig-core/src/a.rs
DA:1,18446744073709551615
DA:2,2
BRDA:1,0,0,4294967295
BRDA:1,0,1,7
end_of_record
";
    let files = parse_lcov(report, Path::new("/repo"));
    let a = &files["crates/rig-core/src/a.rs"];
    assert_eq!(a.covered_lines().collect::<Vec<_>>(), [2]);
    assert_eq!(a.counts().branches, (1, 2));
    assert!(!executed("0") && !executed("-") && executed("1"));
}

#[test]
fn ranges_round_trip() {
    let lines = [1, 2, 3, 7, 9, 10];
    assert_eq!(ranges(lines), "1-3,7,9-10");
    assert_eq!(
        parse_ranges("1-3,7,9-10").unwrap(),
        lines.into_iter().collect()
    );
    assert_eq!(ranges([]), "");
    assert_eq!(parse_ranges("").unwrap(), BTreeSet::new());
    assert_eq!(parse_ranges("x"), None);
}

#[test]
fn the_baseline_round_trips() {
    let text = render(&report(), |_| "abc".to_owned());
    assert_eq!(
        text,
        format!("{HEADER}\ncrates/rig-core/src/a.rs\tabc\t2/3\t1/3\t1,4\t2.0.0\n")
    );
    let rows = parse(&text).unwrap();
    let row = &rows["crates/rig-core/src/a.rs"];
    assert_eq!(row.covered_lines, BTreeSet::from([1, 4]));
    assert_eq!(row.covered_branches, BTreeSet::from([(2, 0, 0)]));
    assert!(parse(&format!("{HEADER}\na\tb\tc")).is_err());
}

#[test]
fn a_drop_is_a_smaller_share() {
    assert!(dropped((5, 10), (4, 10)));
    assert!(!dropped((5, 10), (5, 10)));
    assert!(!dropped((5, 10), (10, 20)));
    assert!(dropped((5, 10), (9, 20)));
    assert!(!dropped((5, 10), (5, 9)));
    assert!(dropped((1, 10), (0, 0)));
    assert!(!dropped((0, 10), (0, 0)));
}

fn baseline() -> BTreeMap<String, Row> {
    parse(&render(&report(), |_| "same".to_owned())).unwrap()
}

#[test]
fn an_unchanged_file_fails_on_any_lost_line_or_branch() {
    let mut current = report();
    assert!(regressions(&baseline(), &current, |_| Some("same".to_owned())).is_empty());
    let a = current.get_mut("crates/rig-core/src/a.rs").unwrap();
    a.lines.insert(4, false);
    a.lines.insert(2, true);
    a.branches.insert((2, 0, 0), false);
    a.branches.insert((2, 0, 1), true);
    // Same counts, different sets: still two regressions.
    assert_eq!(a.counts(), report()["crates/rig-core/src/a.rs"].counts());
    let found = regressions(&baseline(), &current, |_| Some("same".to_owned()));
    assert_eq!(
        found,
        [
            "crates/rig-core/src/a.rs: lines 4 no longer covered",
            "crates/rig-core/src/a.rs: branches 2.0.0 no longer covered"
        ]
    );
}

#[test]
fn only_what_the_baseline_covered_and_is_still_instrumented_counts() {
    let mut current = report();
    let a = current.get_mut("crates/rig-core/src/a.rs").unwrap();
    a.branches.insert((2, 0, 1), false);
    a.branches.insert((7, 0, 0), false);
    a.lines.insert(7, false);
    a.lines.remove(&1);
    a.branches.remove(&(2, 0, 0));
    assert!(regressions(&baseline(), &current, |_| Some("same".to_owned())).is_empty());
}

#[test]
fn a_changed_file_fails_only_when_a_ratio_falls() {
    let mut current = report();
    let a = current.get_mut("crates/rig-core/src/a.rs").unwrap();
    a.lines.insert(1, false);
    a.lines.insert(2, true);
    assert!(regressions(&baseline(), &current, |_| Some("edited".to_owned())).is_empty());
    let a = current.get_mut("crates/rig-core/src/a.rs").unwrap();
    a.lines.insert(9, false);
    let found = regressions(&baseline(), &current, |_| Some("edited".to_owned()));
    assert_eq!(found, ["crates/rig-core/src/a.rs: lines 2/3 -> 2/4"]);
}

#[test]
fn a_deleted_file_is_not_a_regression_but_an_unmeasured_one_is() {
    let empty = BTreeMap::new();
    assert!(regressions(&baseline(), &empty, |_| None).is_empty());
    assert_eq!(
        regressions(&baseline(), &empty, |_| Some("same".to_owned())),
        ["crates/rig-core/src/a.rs: no longer measured"]
    );
}

#[test]
fn intersecting_runs_keeps_what_both_covered() {
    let mut first = report()["crates/rig-core/src/a.rs"].clone();
    let mut second = first.clone();
    second.lines.insert(4, false);
    second.branches.insert((2, 0, 1), true);
    first.intersect(&second);
    assert_eq!(first.covered_lines().collect::<Vec<_>>(), [1]);
    assert!(!first.branches[&(2, 0, 1)]);
}

#[test]
fn per_crate_totals_add_up() {
    let totals = per_crate(&report());
    assert_eq!(totals["rig-core"].lines, (2, 3));
}

#[test]
fn the_wrapper_config_extends_the_repository_config() {
    let config = wrapper_config("[profile.local]\nretries = 0\n", Path::new("/bin/xtask"));
    assert!(config.starts_with("experimental = [\"wrapper-scripts\"]\n[profile.local]"));
    assert!(config.contains("command-line = \"/bin/xtask coverage --wrap\""));
    assert!(
        config
            .contains("[[profile.local.scripts]]\nfilter = 'all()'\nrun-wrapper = 'rig-coverage'")
    );
}

#[test]
fn a_per_test_record_lists_covered_files_only() {
    let mut files = report();
    files.insert("crates/x/src/none.rs".to_owned(), FileCoverage::default());
    assert_eq!(
        per_test_record("rig-core", "a::b", &files),
        "rig-core\ta::b\tcrates/rig-core/src/a.rs\t1,4\t2.0.0\n"
    );
    assert_eq!(sanitize("rig::core"), "rig~~core");
}
