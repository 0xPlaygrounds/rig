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

/// Every `pub enum` in `source` that derives a Serde trait, has at least one
/// data-bearing variant, and declares **no** representation (`tag` or
/// `untagged`) — so Serde falls back to implicit external tagging
/// (`{"VariantName": …}`), decided by variant order rather than by a stable
/// discriminator.
///
/// This is the check [`public_untagged_enums`] cannot make. That one blacklists
/// one attribute; every enum #2281 retagged (`MediaType`, `ToolChoice`,
/// `FinishReason`, `Filter`, `ModelListingError`, `ToolCallDeltaContent`) carried
/// **no** `untagged` attribute at all and sailed past it. Verified against
/// `03643160^`: this scanner flags exactly those six there and none on the
/// retagged tree.
fn public_enums_without_representation(source: &str) -> Vec<(String, usize)> {
    let mut found = Vec::new();
    let mut cursor = 0;

    while let Some(relative) = source[cursor..].find("pub enum ") {
        let declaration = cursor + relative;
        cursor = declaration + "pub enum ".len();

        // `pub enum` must open the line: a match arm or string literal mentioning
        // it is prose, not a declaration.
        let line_start = source[..declaration]
            .rfind('\n')
            .map_or(0, |index| index + 1);
        if !source[line_start..declaration].trim().is_empty() {
            continue;
        }

        let name = source[cursor..]
            .chars()
            .take_while(|character| character.is_ascii_alphanumeric() || *character == '_')
            .collect::<String>();
        if name.is_empty() {
            continue;
        }

        let attributes = attribute_block_before(source, line_start);
        if !attributes.contains("Serialize") && !attributes.contains("Deserialize") {
            continue;
        }
        // A declared representation is the whole point of the policy; either
        // spelling satisfies it (`untagged` is separately policed by
        // `public_untagged_enums` and its allowlist).
        if attributes.contains("tag") || attributes.contains("untagged") {
            continue;
        }

        let Some(body) = enum_body(source, declaration) else {
            // An unbalanced body means the scanner cannot see the variants. Report
            // it rather than skipping: a silent skip reads as a pass, which is one
            // of the two ways the original scanner could go quietly blind.
            found.push((
                format!("{name} (UNPARSED BODY — scanner could not match braces)"),
                source[..declaration].lines().count(),
            ));
            continue;
        };
        if has_data_bearing_variant(&body) {
            found.push((name, source[..declaration].lines().count()));
        }
    }

    found
}

/// The contiguous run of attribute and doc-comment lines directly above `offset`.
fn attribute_block_before(source: &str, offset: usize) -> String {
    let mut lines: Vec<&str> = Vec::new();
    for line in source[..offset].lines().rev() {
        let trimmed = line.trim();
        let is_attribute_or_doc = trimmed.starts_with("#[")
            || trimmed.starts_with("//")
            || trimmed.starts_with(']')
            || trimmed.starts_with(")]")
            || trimmed.ends_with(',')
            || trimmed.ends_with('(')
            || trimmed.contains('=');
        if trimmed.is_empty() || !is_attribute_or_doc {
            break;
        }
        lines.push(line);
    }
    lines.reverse();
    lines.join("\n")
}

/// The `{ … }` body of the enum declared at `declaration`, brace-matched.
fn enum_body(source: &str, declaration: usize) -> Option<String> {
    let open = source[declaration..].find('{')? + declaration;
    let mut depth = 0usize;
    for (index, character) in source[open..].char_indices() {
        match character {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(source[open + 1..open + index].to_owned());
                }
            }
            _ => {}
        }
    }
    None
}

/// Whether any variant carries data — `Variant(T)` or `Variant { .. }`.
///
/// A fieldless enum serializes as a plain string and needs no discriminator, so
/// the policy does not apply to it.
fn has_data_bearing_variant(body: &str) -> bool {
    let mut depth = 0usize;
    for line in body.lines() {
        let trimmed = line.trim();
        // Only consider variants at the enum's own nesting level; a struct
        // variant's fields are not themselves variants.
        let at_top_level = depth == 0;
        depth = depth
            .saturating_add(trimmed.matches('{').count())
            .saturating_sub(trimmed.matches('}').count());

        if !at_top_level || trimmed.is_empty() || trimmed.starts_with("//") {
            continue;
        }
        let without_attribute = if trimmed.starts_with("#[") {
            match trimmed.find(']') {
                Some(end) => trimmed[end + 1..].trim(),
                None => continue,
            }
        } else {
            trimmed
        };
        let mut characters = without_attribute.chars();
        if !characters.next().is_some_and(|c| c.is_ascii_uppercase()) {
            continue;
        }
        let rest: String = characters.collect();
        let name_end = rest
            .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
            .unwrap_or(rest.len());
        if matches!(
            rest[name_end..].trim_start().chars().next(),
            Some('(' | '{')
        ) {
            return true;
        }
    }
    false
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

/// Paths whose enums mirror an upstream SDK's own shapes rather than expressing
/// rig's domain, so rig's discriminator policy does not apply. Provider wire
/// types under `src/providers/` are excluded structurally; these are the
/// provider crates that keep their wire mirror elsewhere.
const WIRE_MIRROR_PATHS: &[(&str, &str)] = &[(
    "crates/rig-bedrock/src/types/converse_output.rs",
    "mirrors the aws-sdk-bedrockruntime Converse output types one-for-one \
     (ContentBlock, StopReason, the Guardrail* family); their shapes are AWS's \
     to choose, and re-tagging them would break the SDK round-trip",
)];

/// The complement of [`rig_owned_public_serde_enums_are_not_untagged`]: that test
/// bans one attribute, this one requires a positive declaration.
///
/// #2281 retagged six enums that the untagged ban would have passed unchanged,
/// because implicit external tagging is a *default*, not an attribute. A guard
/// that only blacklists cannot see a default, so the same breaking-retag cycle
/// could recur; this closes it.
#[test]
fn rig_owned_public_data_bearing_enums_declare_a_representation() {
    let workspace = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("rig-core must live under the workspace crates directory")
        .to_path_buf();

    let mut violations = Vec::new();
    let mut visited_wire_mirrors = Vec::new();
    collect_representation_violations(
        &workspace,
        &workspace.join("crates"),
        &mut violations,
        &mut visited_wire_mirrors,
    );

    assert!(
        violations.is_empty(),
        "public data-bearing enum has no declared Serde representation, so it \
         falls back to implicit external tagging (`{{\"Variant\": …}}`) decided by \
         variant order. Give it `#[serde(tag = \"type\")]`, \
         `#[serde(tag = \"type\", content = \"content\")]`, or an allowlisted \
         `untagged`:\n{}",
        violations.join("\n")
    );

    let unseen = WIRE_MIRROR_PATHS
        .iter()
        .map(|(path, _)| *path)
        .filter(|path| !visited_wire_mirrors.iter().any(|seen| seen == path))
        .collect::<Vec<_>>();
    assert!(
        unseen.is_empty(),
        "stale WIRE_MIRROR_PATHS entry (the file moved or was deleted; delete the entry): {unseen:?}"
    );
}

fn collect_representation_violations(
    workspace: &Path,
    dir: &Path,
    violations: &mut Vec<String>,
    visited_wire_mirrors: &mut Vec<String>,
) {
    for entry in std::fs::read_dir(dir).expect("source directory should be readable") {
        let entry = entry.expect("source entry should be readable");
        let path = entry.path();
        if path.is_dir() {
            let name = entry.file_name();
            if !SKIPPED_DIRS.contains(&name.to_string_lossy().as_ref()) {
                collect_representation_violations(
                    workspace,
                    &path,
                    violations,
                    visited_wire_mirrors,
                );
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

        // Only rig-owned surface: `src/providers/` mirrors provider wires, whose
        // shapes belong to the provider.
        if !relative.contains("/src/") || relative.contains("/src/providers/") {
            continue;
        }
        if let Some((wire_path, _)) = WIRE_MIRROR_PATHS
            .iter()
            .find(|(wire_path, _)| relative.ends_with(wire_path))
        {
            visited_wire_mirrors.push((*wire_path).to_owned());
            continue;
        }

        let source = std::fs::read_to_string(&path).expect("source file should be readable");
        for (enum_name, line) in public_enums_without_representation(&source) {
            violations.push(format!("{relative}:{line} {enum_name}"));
        }
    }
}

#[test]
fn representation_scanner_flags_implicit_external_tagging_only_when_data_bearing() {
    let source = r#"
        #[derive(serde::Serialize, serde::Deserialize)]
        pub enum ImplicitlyTagged { Text(String), Image { url: String } }

        /// Fieldless enums serialize as plain strings and need no discriminator.
        #[derive(serde::Serialize, serde::Deserialize)]
        pub enum PlainStrings { Alpha, Beta }

        #[derive(serde::Serialize, serde::Deserialize)]
        #[serde(tag = "type", content = "content")]
        pub enum Declared { Text(String) }

        #[derive(Debug)]
        pub enum NotSerde { Text(String) }

        #[derive(serde::Serialize)]
        pub enum StructVariantFieldsAreNotVariants {
            Only { lowercase_field: String },
        }
    "#;

    let flagged = public_enums_without_representation(source);
    assert_eq!(
        flagged
            .iter()
            .map(|(name, _)| name.as_str())
            .collect::<Vec<_>>(),
        vec!["ImplicitlyTagged", "StructVariantFieldsAreNotVariants"],
        "flagged: {flagged:?}"
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
