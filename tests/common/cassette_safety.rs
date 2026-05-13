//! Safety checks for committed cassette fixtures.

use std::path::Path;

const CASSETTE_ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/cassettes");

const FORBIDDEN_PATTERNS: &[&str] = &[
    "Authorization:",
    "Bearer ",
    "sk-",
    "x-api-key:",
    "x-goog-api-key:",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
    "AIza",
    "__cf_bm=",
    "proj_",
];

const REDACTED_HEADER_NAMES: &[&str] = &[
    "openai-organization",
    "openai-project",
    "set-cookie",
    "x-api-key",
    "x-goog-api-key",
];

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

        for pattern in FORBIDDEN_PATTERNS {
            if contents.contains(pattern) {
                failures.push(format!("{} contains {pattern:?}", path.display()));
            }
        }

        for header in REDACTED_HEADER_NAMES {
            let leaked_values = contents.lines().enumerate().any(|(index, line)| {
                line.trim() == format!("- name: {header}")
                    && contents
                        .lines()
                        .nth(index + 1)
                        .is_some_and(|next| next.trim() != "value: '[REDACTED]'")
            });

            if leaked_values {
                failures.push(format!(
                    "{} contains unredacted {header} header",
                    path.display()
                ));
            }
        }
    }
}
