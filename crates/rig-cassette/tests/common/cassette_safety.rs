//! Each compiled provider binary checks its own fixture partition. The xtask
//! inventory check verifies that every provider directory has an executable
//! binary containing both safety tests, without parsing Rust source.

use std::{fs, path::Path};

#[test]
fn cassettes_do_not_contain_obvious_secrets() {
    let root = crate::cassettes::cassette_root();
    let mut failures = Vec::new();
    for entry in fs::read_dir(&root).expect("cassette root should be readable") {
        let entry = entry.expect("cassette root entry should be readable");
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if !entry.path().is_dir()
            || name.is_empty()
            || !name
                .chars()
                .all(|ch| ch.is_ascii_lowercase() || ch.is_ascii_digit() || ch == '_')
        {
            failures.push(format!(
                "{} is not a canonical provider directory",
                entry.path().display()
            ));
        }
    }
    let own_dir = root.join(env!("CARGO_CRATE_NAME"));
    if own_dir.is_dir() {
        scan_dir(&own_dir, &mut failures);
    } else {
        // A first-capture-only suite has no directory yet. Every absent file
        // must still have an explicit allowance in its compiled declaration.
        rig_test_support::recording::check_inventory(&root, env!("CARGO_CRATE_NAME"))
            .expect("missing provider directory must have explicit first-capture declarations");
    }
    assert!(
        failures.is_empty(),
        "cassette secret scan failed:\n{}",
        failures.join("\n")
    );
}

#[test]
fn cassette_files_match_registered_scenarios() -> anyhow::Result<()> {
    rig_test_support::recording::check_inventory(
        &crate::cassettes::cassette_root(),
        env!("CARGO_CRATE_NAME"),
    )
}

fn scan_dir(dir: &Path, failures: &mut Vec<String>) {
    for entry in fs::read_dir(dir).expect("cassette directory should be readable") {
        let path = entry
            .expect("cassette directory entry should be readable")
            .path();
        if path.is_dir() {
            scan_dir(&path, failures);
        } else if path.extension().is_some_and(|ext| ext == "yaml") {
            let contents = fs::read_to_string(&path).expect("cassette should be readable as UTF-8");
            failures.extend(crate::cassettes::cassette_safety_failures(&path, &contents));
        }
    }
}
