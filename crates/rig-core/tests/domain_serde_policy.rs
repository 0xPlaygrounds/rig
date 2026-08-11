//! Structural guard for public Serde enum representations.
//!
//! Rig-owned public domain, persistence, and event enums must never return to
//! variant-trial-order deserialization. Public untagged provider wire unions
//! are allowed only when their exact type and upstream-owned shape are recorded
//! in `domain_serde_policy_allowlist.txt`.

#![allow(clippy::expect_used)]

use std::path::{Path, PathBuf};

const SOURCE_ROOTS: &[&str] = &["src", "crates"];
const SKIPPED_DIRS: &[&str] = &[".git", "target", "tests", "examples"];

struct AllowlistEntry {
    path_suffix: String,
    enum_name: String,
    used: bool,
}

fn parse_allowlist(raw: &str) -> Vec<AllowlistEntry> {
    raw.lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .map(|line| {
            let mut fields = line.splitn(3, '|').map(str::trim);
            let path_suffix = fields.next().unwrap_or_default().to_owned();
            let enum_name = fields.next().unwrap_or_default().to_owned();
            let justification = fields.next().unwrap_or_default();
            assert!(
                !path_suffix.is_empty() && !enum_name.is_empty() && !justification.is_empty(),
                "malformed domain_serde_policy_allowlist.txt entry (need `path | enum | justification`): {line}"
            );
            AllowlistEntry {
                path_suffix,
                enum_name,
                used: false,
            }
        })
        .collect()
}

fn public_untagged_enums(source: &str) -> Vec<(String, usize)> {
    let mut found = Vec::new();
    let mut cursor = 0;

    while let Some(relative_start) = source[cursor..].find("#[serde") {
        let start = cursor + relative_start;
        cursor = start + "#[serde".len();

        // Attribute markers in rustdoc or string literals are prose, not code.
        let line_start = source[..start].rfind('\n').map_or(0, |index| index + 1);
        if !source[line_start..start].trim().is_empty() {
            continue;
        }

        let Some(relative_end) = source[start..].find(']') else {
            continue;
        };
        let end = start + relative_end + 1;
        let attribute: String = source[start..end]
            .chars()
            .filter(|character| !character.is_whitespace())
            .collect();
        if !attribute.starts_with("#[serde(") || !attribute.contains("untagged") {
            continue;
        }

        // A serde enum attribute is followed by any remaining attributes and
        // then the declaration. Stop at a semicolon so a non-enum item cannot
        // make the scanner drift into a later declaration.
        let after = &source[end..];
        let brace = after.find('{');
        let semicolon = after.find(';');
        let declaration_end = match (brace, semicolon) {
            (Some(brace), Some(semicolon)) if semicolon < brace => continue,
            (Some(brace), _) => brace,
            (None, _) => continue,
        };
        let declaration = &after[..declaration_end];
        let Some(name_start) = declaration.find("pub enum ") else {
            continue;
        };
        // A variant-level serde attribute can be followed by the closing brace
        // of its enum and then an unrelated public enum. It is not an enum
        // representation attribute and must not be attributed to the later type.
        if declaration[..name_start].contains('}') {
            continue;
        }
        let name = declaration[name_start + "pub enum ".len()..]
            .chars()
            .take_while(|character| character.is_ascii_alphanumeric() || *character == '_')
            .collect::<String>();
        if !name.is_empty() {
            found.push((name, source[..start].lines().count()));
        }
    }

    found
}

fn scan_rust_sources(
    workspace: &Path,
    dir: &Path,
    allowlist: &mut [AllowlistEntry],
    visited: &mut Vec<String>,
    violations: &mut Vec<String>,
) {
    for entry in std::fs::read_dir(dir).expect("source directory should be readable") {
        let entry = entry.expect("source entry should be readable");
        let path = entry.path();
        if path.is_dir() {
            let name = entry.file_name();
            if !SKIPPED_DIRS.contains(&name.to_string_lossy().as_ref()) {
                scan_rust_sources(workspace, &path, allowlist, visited, violations);
            }
            continue;
        }
        if path.extension().is_none_or(|extension| extension != "rs") {
            continue;
        }

        let relative = path
            .strip_prefix(workspace)
            .expect("scanned source should be under the workspace")
            .to_string_lossy()
            .replace('\\', "/");
        visited.push(relative.clone());
        let source = std::fs::read_to_string(&path).expect("source file should be readable");
        for (enum_name, line) in public_untagged_enums(&source) {
            if let Some(entry) = allowlist.iter_mut().find(|entry| {
                relative.ends_with(&entry.path_suffix) && enum_name == entry.enum_name
            }) {
                assert!(
                    !entry.used,
                    "duplicate public untagged enum matched one allowlist entry: {relative}:{line} {enum_name}"
                );
                entry.used = true;
            } else {
                violations.push(format!("{relative}:{line} {enum_name}"));
            }
        }
    }
}

#[test]
fn rig_owned_public_serde_enums_are_not_untagged() {
    let workspace = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("rig-core must live under the workspace crates directory")
        .to_path_buf();
    let allowlist_path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/domain_serde_policy_allowlist.txt");
    let raw = std::fs::read_to_string(&allowlist_path)
        .expect("domain Serde policy allowlist should be readable");
    let mut allowlist = parse_allowlist(&raw);
    let mut visited = Vec::new();
    let mut violations = Vec::new();

    for relative in SOURCE_ROOTS {
        let root = workspace.join(relative);
        assert!(
            root.is_dir(),
            "domain policy root moved or vanished: {}",
            root.display()
        );
        scan_rust_sources(
            &workspace,
            &root,
            &mut allowlist,
            &mut visited,
            &mut violations,
        );
    }

    for required in [
        "crates/rig-core/src/completion/message.rs",
        "crates/rig-core/src/streaming/mod.rs",
        "crates/rig-agent/src/agent/prompt_request/streaming.rs",
        "crates/rig-sqlite/src/lib.rs",
    ] {
        assert!(
            visited.iter().any(|path| path.ends_with(required)),
            "domain Serde scan did not visit required floor file `{required}`"
        );
    }

    assert!(
        violations.is_empty(),
        "public `#[serde(untagged)]` enum is not a justified provider-wire exception; use an explicit discriminator or add an exact `path | enum | justification` entry to domain_serde_policy_allowlist.txt:\n{}",
        violations.join("\n")
    );

    let stale = allowlist
        .iter()
        .filter(|entry| !entry.used)
        .map(|entry| format!("{}::{}", entry.path_suffix, entry.enum_name))
        .collect::<Vec<_>>();
    assert!(
        stale.is_empty(),
        "stale domain Serde policy allowlist entries (the public untagged enum moved or was removed; delete the entry): {stale:?}"
    );
}

#[test]
fn policy_scanner_recognizes_multiline_attributes_and_only_public_enums() {
    let source = r#"
        /// Prose mentioning #[serde(untagged)] is ignored.
        #[derive(serde::Deserialize)]
        #[serde(
            rename_all = "snake_case",
            untagged
        )]
        pub enum PublicWire { Text(String) }

        #[serde(untagged)]
        enum PrivateDetail { Text(String) }
    "#;

    assert_eq!(
        public_untagged_enums(source),
        vec![("PublicWire".to_owned(), 4)]
    );
}
