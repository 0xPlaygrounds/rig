use super::*;

#[test]
fn a_manifest_reads_back_as_written() {
    let mut rows = BTreeMap::new();
    rows.insert(
        (
            "test".to_owned(),
            "rig-cassette::openai openai::a".to_owned(),
        ),
        "rig-cassette::openai openai::b".to_owned(),
    );
    rows.insert(
        ("fixture".to_owned(), "openai/a.yaml".to_owned()),
        "-".to_owned(),
    );
    let text = render(&rows);
    assert!(text.starts_with(PREAMBLE), "the rule heads the manifest");
    let parsed = parse(&text).expect("the manifest parses");
    assert_eq!(
        parsed,
        [
            Row {
                kind: "fixture".into(),
                item: "openai/a.yaml".into(),
                covered: "-".into(),
            },
            Row {
                kind: "test".into(),
                item: "rig-cassette::openai openai::a".into(),
                covered: "rig-cassette::openai openai::b".into(),
            },
        ]
    );
    assert!(parse(&format!("{PREAMBLE}test\tonly two\n")).is_err());
}

#[test]
fn a_sweep_is_named_by_target_and_prefix() {
    assert!(is_sweep("rig-cassette::chat_parity", "anything"));
    assert!(is_sweep(
        "rig-cassette::verify",
        "corpus_oracle::bus_engine::x"
    ));
    assert!(!is_sweep("rig-cassette::verify", "golden_replay::x"));
    assert!(is_sweep(
        "rig-cassette::openai",
        "cassette_safety::cassette_files_match_registered_scenarios"
    ));
    assert!(!is_sweep("rig-core", "cassette_safety::x"));
    assert!(!is_sweep(
        "rig-cassette::openai",
        "openai::cassette::streaming::smoke"
    ));
}

#[test]
fn only_a_provider_targets_own_test_can_be_stale() {
    let scanned = BTreeSet::from([(
        "rig-cassette::openai".to_owned(),
        "openai::cassette::kept".to_owned(),
    )]);
    let id = |binary: &str, test: &str| (binary.to_owned(), test.to_owned());
    assert!(!is_stale(
        &id("rig-cassette::openai", "openai::cassette::kept"),
        &scanned
    ));
    assert!(is_stale(
        &id("rig-cassette::openai", "openai::cassette::gone"),
        &scanned
    ));
    // A shared module's test, compiled into the provider target.
    assert!(!is_stale(
        &id("rig-cassette::openai", "corpus_matrix::tests::x"),
        &scanned
    ));
    assert!(!is_stale(&id("rig-core", "openai::x"), &scanned));
}

#[test]
fn a_fixture_takes_its_snapshot_and_clock_and_a_golden_its_one_file() {
    assert_eq!(
        sidecars("openai/a/b.yaml"),
        [
            "openai/a/b.yaml",
            "openai/a/b.requests.json",
            "openai/a/b.clock.json"
        ]
    );
    let golden = golden_of("deepseek_x");
    assert_eq!(
        golden.files(),
        [format!("{}/deepseek_x.effects.json", names::EFFECTS)]
    );
    assert_eq!(golden.label(), "deepseek_x");
}

#[test]
fn the_keep_list_reads_the_first_word_of_each_line() {
    let text = "# a comment\n\nxai/a/b.yaml   the reason\n  golden_name reason too\n";
    assert_eq!(parse_keep(text), ["xai/a/b.yaml", "golden_name"]);
}
