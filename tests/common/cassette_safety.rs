//! Safety checks for committed cassette fixtures.

use std::path::Path;

const CASSETTE_ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/cassettes");
const REWRITE_ENV: &str = "RIG_REWRITE_CASSETTES";

#[test]
fn cassettes_do_not_contain_obvious_secrets() {
    let root = Path::new(CASSETTE_ROOT);
    if !root.exists() {
        return;
    }

    let mut failures = Vec::new();
    scan_dir(root, &mut failures);

    assert!(
        failures.is_empty(),
        "cassette secret scan failed:\n{}",
        failures.join("\n")
    );
}

fn scan_dir(dir: &Path, failures: &mut Vec<String>) {
    for entry in std::fs::read_dir(dir).expect("cassette directory should be readable") {
        let entry = entry.expect("cassette directory entry should be readable");
        let path = entry.path();

        if path.is_dir() {
            scan_dir(&path, failures);
            continue;
        }

        if path.extension().and_then(|ext| ext.to_str()) != Some("yaml") {
            continue;
        }

        let contents =
            std::fs::read_to_string(&path).expect("cassette should be readable as UTF-8");
        let scrubbed = crate::cassettes::scrub_cassette_contents(&contents);

        if std::env::var_os(REWRITE_ENV).is_some() && scrubbed != contents {
            std::fs::write(&path, scrubbed).expect("scrubbed cassette should be writable");
            continue;
        }

        failures.extend(crate::cassettes::cassette_safety_failures(&path, &contents));
    }
}
