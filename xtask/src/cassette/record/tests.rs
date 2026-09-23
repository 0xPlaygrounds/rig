use super::*;

#[test]
fn attempts_are_counted_per_test_and_unrun_lines_do_not_count() {
    let ledger = format!(
        "{LEDGER_HEADER}\n\
         groq\tgroq/a.yaml\tgroq::a\t1\t1\tlog x\n\
         groq\tgroq/a.yaml\tgroq::a\t2\t0\tlog y\n\
         groq\tgroq/b.yaml\tgroq::b\t1\t0\tlog z\n\
         groq\tgroq/a.yaml\tgroq::a\t0\tskipped\tcap\n"
    );
    assert_eq!(prior_attempts(&ledger, "groq::a"), 2);
    assert_eq!(prior_attempts(&ledger, "groq::b"), 1);
    assert_eq!(prior_attempts(&ledger, "groq::c"), 0);
}

#[test]
fn an_interrupted_run_still_counts() {
    let interrupted = format!(
        "{LEDGER_HEADER}\n\
         groq\tgroq/a.yaml\tgroq::a\t1\t1\tlog x\n\
         groq\tgroq/a.yaml\tgroq::a\t2\tstarted\t-\n"
    );
    assert_eq!(prior_attempts(&interrupted, "groq::a"), 2);
    assert_eq!(next_attempt(&interrupted, "groq::a", 2), None);
}

#[test]
fn a_test_at_the_cap_is_not_run_again() {
    let failed_twice = format!(
        "{LEDGER_HEADER}\n\
         groq\tgroq/a.yaml\tgroq::a\t1\t1\tlog x\n\
         groq\tgroq/a.yaml\tgroq::a\t2\t1\tlog y\n"
    );
    assert_eq!(next_attempt(&failed_twice, "groq::a", 6), Some(3));
    assert_eq!(next_attempt(&failed_twice, "groq::a", 2), None);
    assert_eq!(next_attempt(&failed_twice, "groq::b", 2), Some(1));
    assert_eq!(next_attempt(LEDGER_HEADER, "groq::a", 0), None);
}

#[test]
fn fixture_arguments_accept_both_forms() {
    assert_eq!(
        parse_fixture("openai/raw_capture_matrix/raw_round_trips.yaml"),
        Some(("openai".into(), "raw_capture_matrix/raw_round_trips".into()))
    );
    assert_eq!(
        parse_fixture("crates/rig-cassette/fixtures/cassettes/gemini/a/b.yaml"),
        Some(("gemini".into(), "a/b".into()))
    );
    assert_eq!(parse_fixture("openai/no_extension"), None);
    assert_eq!(parse_fixture("lonely.yaml"), None);
}

#[test]
fn options_default_to_the_cap_of_six_and_cleanup() {
    let options = parse_options(&["openai/a/b.yaml".into()]).expect("options");
    assert_eq!(options.cap, 6);
    assert!(options.cleanup && !options.dry_run);
    let options = parse_options(
        &[
            "--cap",
            "2",
            "--pause",
            "30",
            "--dry-run",
            "--no-cleanup",
            "x/y.yaml",
        ]
        .map(String::from),
    )
    .expect("options");
    assert_eq!((options.cap, options.pause_seconds), (2, 30));
    assert!(options.dry_run && !options.cleanup);
    assert!(parse_options(&[]).is_err());
    assert!(parse_options(&["--cap", "7", "x/y.yaml"].map(String::from)).is_err());
}

#[test]
fn a_failed_run_restores_the_snapshot_taken_just_before_it() {
    let dir = std::env::temp_dir().join(format!("xtask-restore-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let fixtures = dir.join("openai");
    std::fs::create_dir_all(fixtures.join("cells")).expect("dir");
    let fixture = fixtures.join("cells/a.yaml");
    std::fs::write(&fixture, "original").expect("fixture");
    std::fs::write(fixtures.join("cells/other.yaml"), "untouched").expect("other");

    // Attempt 1 passed and wrote a new recording.
    std::fs::write(&fixture, "recording 1").expect("pass");

    // Attempt 2 starts from that recording and fails without writing: the
    // good recording stays.
    let before = snapshot(&fixtures).expect("snapshot");
    let kept = dir.join("kept2");
    assert!(
        restore_snapshot(&fixtures, &before, &kept)
            .expect("restore")
            .is_empty()
    );
    assert_eq!(
        std::fs::read_to_string(&fixture).expect("read"),
        "recording 1"
    );

    // Attempt 3 fails after rewriting the fixture, writing a fixture nobody
    // named and deleting another: all of it is undone and kept aside.
    let before = snapshot(&fixtures).expect("snapshot");
    std::fs::write(&fixture, "half-written").expect("fail");
    std::fs::write(fixtures.join("cells/new.yaml"), "stray").expect("stray");
    std::fs::remove_file(fixtures.join("cells/other.yaml")).expect("remove");
    let kept = dir.join("kept3");
    let restored = restore_snapshot(&fixtures, &before, &kept).expect("restore");
    assert_eq!(
        restored,
        [
            PathBuf::from("cells/a.yaml"),
            PathBuf::from("cells/new.yaml"),
            PathBuf::from("cells/other.yaml")
        ]
    );
    assert_eq!(
        std::fs::read_to_string(&fixture).expect("read"),
        "recording 1"
    );
    assert_eq!(
        std::fs::read_to_string(fixtures.join("cells/other.yaml")).expect("read"),
        "untouched"
    );
    assert!(!fixtures.join("cells/new.yaml").exists());
    assert_eq!(
        std::fs::read_to_string(kept.join("cells/a.yaml")).expect("read"),
        "half-written"
    );
    assert_eq!(
        std::fs::read_to_string(kept.join("cells/new.yaml")).expect("read"),
        "stray"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_run_that_ran_no_test_is_not_a_recording() {
    assert!(ran_nothing(
        "    Starting 0 tests across 1 binary (5000 skipped)"
    ));
    assert!(ran_nothing(
        "     Summary [   0.001s] 0 tests run: 0 passed, 12 skipped"
    ));
    assert!(!ran_nothing(
        "     Summary [   3.2s] 1 test run: 1 passed, 5000 skipped"
    ));
}

#[test]
fn a_restore_puts_originals_back_even_when_keeping_a_copy_fails() {
    let dir = std::env::temp_dir().join(format!("xtask-restore-kept-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let fixtures = dir.join("groq");
    std::fs::create_dir_all(&fixtures).expect("dir");
    std::fs::write(fixtures.join("a.yaml"), "original").expect("fixture");
    let before = snapshot(&fixtures).expect("snapshot");
    std::fs::write(fixtures.join("a.yaml"), "half-written").expect("fail");
    // A file where the kept directory should be: every copy fails.
    let kept = dir.join("kept");
    std::fs::write(&kept, "in the way").expect("blocker");
    let error = restore_snapshot(&fixtures, &before, &kept).expect_err("keeping fails");
    assert!(error.contains("keeping"), "{error}");
    assert!(!error.contains("restoring"), "{error}");
    assert_eq!(
        std::fs::read_to_string(fixtures.join("a.yaml")).expect("read"),
        "original"
    );
    let _ = std::fs::remove_dir_all(&dir);
}
