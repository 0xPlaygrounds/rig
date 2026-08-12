//! Structural guard for public Serde enum representations.
//!
//! Rig-owned public domain, persistence, and event enums must never return to
//! variant-trial-order deserialization. Public untagged provider wire unions
//! are allowed only when their exact type and upstream-owned shape are recorded
//! in `domain_serde_policy_allowlist.txt`.

#![allow(clippy::expect_used, clippy::panic)]

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

/// One public enum, as the compiler's own parse sees it.
#[derive(Debug, Clone)]
struct PublicEnum {
    name: String,
    line: usize,
    /// A `#[derive(...)]` list naming `Serialize` or `Deserialize`.
    derives_serde: bool,
    /// Any variant with named or unnamed fields.
    data_bearing: bool,
    /// An enum-level `#[serde(...)]` carrying the `tag` or `untagged` key.
    declares_representation: bool,
    /// An enum-level `#[serde(untagged)]` specifically.
    untagged: bool,
}

/// Every lexically public enum in `source`, read from the AST.
///
/// This deliberately does not scan text. Two successive text scanners were
/// defeated in review by things that look nothing like code: an enum documented
/// "Two-stage handshake" exempted itself because `stage` contains `tag`, and a
/// brace inside a doc comment unbalanced the body matcher in both directions.
/// Every such fix trades one substring hazard for the next — `#[serde(rename =
/// "stage")]` would still fool a `contains("tag")` probe. The parser is the only
/// thing that agrees with rustc about what an attribute is.
///
/// A parse failure is a hard error, never a skip: a file the scanner cannot read
/// is a file the policy is not enforcing, and that must not be able to look like
/// compliance.
fn public_enums(path: &Path, source: &str) -> Result<Vec<PublicEnum>, syn::Error> {
    let file = syn::parse_file(source)?;
    let mut found = Vec::new();
    collect_enums(&file.items, &mut found)?;
    let _ = path;
    Ok(found)
}

/// Descends `mod { … }` blocks: rig declares whole provider wire surfaces in
/// inline modules (`pub mod gemini_api_types { … }`), and a scanner that only
/// looked at top-level items would silently exempt every enum inside them.
fn collect_enums(items: &[syn::Item], found: &mut Vec<PublicEnum>) -> Result<(), syn::Error> {
    for item in items {
        match item {
            syn::Item::Mod(module) => {
                if let Some((_, nested)) = &module.content {
                    collect_enums(nested, found)?;
                }
            }
            syn::Item::Enum(item) => {
                if !matches!(item.vis, syn::Visibility::Public(_)) {
                    continue;
                }
                let keys = serde_keys(&item.attrs)?;
                found.push(PublicEnum {
                    name: item.ident.to_string(),
                    line: item.ident.span().start().line,
                    derives_serde: derives_serde(&item.attrs)?,
                    data_bearing: item
                        .variants
                        .iter()
                        .any(|variant| !matches!(variant.fields, syn::Fields::Unit)),
                    declares_representation: keys
                        .iter()
                        .any(|key| key == "tag" || key == "untagged"),
                    untagged: keys.iter().any(|key| key == "untagged"),
                });
            }
            _ => {}
        }
    }
    Ok(())
}

/// Whether a `#[derive(...)]` list names `Serialize` or `Deserialize`.
///
/// Only `derive` attributes are inspected, so a doc comment or an unrelated
/// attribute mentioning the word cannot make a non-serde enum look serialized.
/// The path's last segment is compared, so `serde::Serialize` counts.
///
/// Parse errors propagate rather than being swallowed — see [`serde_keys`] for
/// what swallowing one costs.
fn derives_serde(attrs: &[syn::Attribute]) -> Result<bool, syn::Error> {
    for attr in attrs.iter().filter(|attr| attr.path().is_ident("derive")) {
        // A bare `#[derive]` carries no list; nothing to read.
        if matches!(attr.meta, syn::Meta::Path(_)) {
            continue;
        }
        let derived = attr.parse_args_with(
            syn::punctuated::Punctuated::<syn::Path, syn::Token![,]>::parse_terminated,
        )?;
        if derived.iter().any(|path| {
            path.segments
                .last()
                .is_some_and(|last| last.ident == "Serialize" || last.ident == "Deserialize")
        }) {
            return Ok(true);
        }
    }
    Ok(false)
}

/// The metadata keys of every enum-level `#[serde(...)]` attribute.
///
/// Keys only — `tag = "type"` yields `tag`, and a *value* of `"untagged"` or
/// `"stage"` yields nothing. That distinction is the whole point: it is what a
/// substring probe over source text cannot make.
///
/// Parsed as a comma-separated list of [`syn::Meta`], which admits all three
/// container-attribute forms serde documents: a bare path (`untagged`), a
/// name-value (`tag = "type"`), and a **nested list** (`bound(serialize = "…")`,
/// `rename(serialize = "wire")`).
///
/// The nested form is why this reads `Meta` rather than driving
/// `parse_nested_meta` with a callback. That callback consumed `key = value` but
/// not `key(...)`, so serde-legal input like
///
/// ```ignore
/// #[serde(bound(serialize = "T: Serialize"), untagged)]
/// ```
///
/// failed mid-parse — and because the error was discarded with `let _ = …`, every
/// key *after* the nested entry went unseen. `untagged` was therefore invisible,
/// which in a provider directory (excluded from the positive representation scan)
/// let an enum bypass the untagged allowlist entirely. Errors now propagate: an
/// attribute this scanner cannot read must fail the test, never quietly narrow it.
fn serde_keys(attrs: &[syn::Attribute]) -> Result<Vec<String>, syn::Error> {
    let mut keys = Vec::new();
    for attr in attrs.iter().filter(|attr| attr.path().is_ident("serde")) {
        // A bare `#[serde]` carries no keys.
        if matches!(attr.meta, syn::Meta::Path(_)) {
            continue;
        }
        let entries = attr.parse_args_with(
            syn::punctuated::Punctuated::<syn::Meta, syn::Token![,]>::parse_terminated,
        )?;
        for entry in entries {
            if let Some(ident) = entry.path().get_ident() {
                keys.push(ident.to_string());
            }
        }
    }
    Ok(keys)
}

/// [`public_enums`] with a parse failure turned into a test failure.
///
/// Requirement 7 of the policy: a file the scanner cannot parse is a file the
/// policy is not enforcing, so it must never read as compliance.
fn parsed_enums(path: &Path, source: &str) -> Vec<PublicEnum> {
    public_enums(path, source).unwrap_or_else(|error| {
        panic!(
            "domain Serde policy could not parse {} as Rust: {error}. \
             A file the scanner cannot read is a file the policy is not enforcing; \
             fix the parse or narrow SKIPPED_DIRS deliberately.",
            path.display()
        )
    })
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
        for found in parsed_enums(&path, &source)
            .into_iter()
            .filter(|found| found.untagged)
        {
            let (enum_name, line) = (found.name, found.line);
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
    let mut visited = Vec::new();
    // Both roots, matching the untagged ban. Scanning only `crates/` left the
    // workspace-root `src/` — the published `rig` facade — outside the wall, so a
    // public data-bearing enum added there would have fallen back to implicit
    // external tagging and passed silently.
    for relative in SOURCE_ROOTS {
        let root = workspace.join(relative);
        assert!(
            root.is_dir(),
            "domain policy root moved or vanished: {}",
            root.display()
        );
        collect_representation_violations(
            &workspace,
            &root,
            &mut violations,
            &mut visited_wire_mirrors,
            &mut visited,
        );
    }

    // Floor files, mirroring the untagged ban's own guard: without this the scan
    // could silently stop reaching the surface it exists to police — which is
    // exactly how the missing `src/` root went unnoticed.
    for required in [
        "crates/rig-core/src/completion/message.rs",
        "crates/rig-core/src/streaming/mod.rs",
        "crates/rig-core/src/vector_store/request.rs",
        "crates/rig-core/src/model/listing.rs",
    ] {
        assert!(
            visited.iter().any(|path| path.ends_with(required)),
            "representation scan did not visit required floor file `{required}`"
        );
    }

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
    visited: &mut Vec<String>,
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
                    visited,
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
        //
        // The root test is `starts_with("src/") || contains("/src/")`: the
        // workspace-root `src/` (the published `rig` facade) has no leading
        // slash, so a `contains("/src/")` test alone silently excluded it even
        // once that root was scanned.
        let under_src = relative.starts_with("src/") || relative.contains("/src/");
        let under_providers =
            relative.starts_with("src/providers/") || relative.contains("/src/providers/");
        if !under_src || under_providers {
            continue;
        }
        visited.push(relative.clone());
        if let Some((wire_path, _)) = WIRE_MIRROR_PATHS
            .iter()
            .find(|(wire_path, _)| relative.ends_with(wire_path))
        {
            visited_wire_mirrors.push((*wire_path).to_owned());
            continue;
        }

        let source = std::fs::read_to_string(&path).expect("source file should be readable");
        for found in parsed_enums(&path, &source).into_iter().filter(|found| {
            found.derives_serde && found.data_bearing && !found.declares_representation
        }) {
            violations.push(format!("{relative}:{} {}", found.line, found.name));
        }
    }
}

#[test]
fn ast_scanner_reads_syntax_not_text() {
    let source = r#"
        /// Two-stage handshake result. Mentions untagged and tagged in prose.
        #[derive(serde::Serialize, serde::Deserialize)]
        pub enum StageProse { A(u32) }

        #[derive(serde::Serialize, serde::Deserialize)]

        pub enum BlankLineBeforeDeclaration { A(u32) }

        #[derive(serde::Serialize, serde::Deserialize)]
        #[serde(rename_all = "snake_case")]
        pub enum RenameIsNotARepresentation { A(u32) }

        /// A value that merely spells a representation is not one.
        #[derive(serde::Serialize, serde::Deserialize)]
        #[serde(rename = "untagged")]
        pub enum ValueSaysUntagged { A(u32) }

        #[derive(serde::Serialize, serde::Deserialize)]
        #[serde(tag = "type", content = "content")]
        pub enum Declared {
            /// Rendered as `{` in the output.
            A(u32),
        }

        #[derive(serde::Serialize, serde::Deserialize)]
        pub enum FieldlessNeedsNothing { Alpha, Beta }

        #[derive(Debug)]
        pub enum NotSerde { A(u32) }

        #[derive(serde::Serialize, serde::Deserialize)]
        enum PrivateDetail { A(u32) }

        pub mod nested {
            #[derive(serde::Serialize, serde::Deserialize)]
            pub enum InsideAnInlineModule { A(u32) }
        }
    "#;

    let parsed = public_enums(Path::new("probe.rs"), source).expect("probe should parse");
    let needs_representation = parsed
        .iter()
        .filter(|found| found.derives_serde && found.data_bearing && !found.declares_representation)
        .map(|found| found.name.as_str())
        .collect::<Vec<_>>();

    assert_eq!(
        needs_representation,
        vec![
            // Prose containing `tag` is prose. `stage` is not a representation.
            "StageProse",
            // A blank line between the derive and the declaration is legal Rust.
            "BlankLineBeforeDeclaration",
            // `rename_all` is a key, but not `tag`/`untagged`.
            "RenameIsNotARepresentation",
            // `rename = "untagged"` is a *value*; only keys count.
            "ValueSaysUntagged",
            // An inline module is still rig's surface.
            "InsideAnInlineModule",
        ],
        "parsed: {parsed:#?}"
    );

    // A brace in a doc comment is invisible to the parser, so `Declared` neither
    // fails nor hides its neighbours.
    assert!(parsed.iter().any(|found| found.name == "Declared"
        && found.declares_representation
        && found.data_bearing));
    // Fieldless enums serialize as plain strings; the policy does not apply.
    assert!(
        parsed
            .iter()
            .any(|found| found.name == "FieldlessNeedsNothing" && !found.data_bearing)
    );
    // Only `#[derive(...)]` decides serde-ness, and only `pub` is public.
    assert!(
        parsed
            .iter()
            .any(|found| found.name == "NotSerde" && !found.derives_serde)
    );
    assert!(!parsed.iter().any(|found| found.name == "PrivateDetail"));
}

#[test]
fn ast_scanner_reports_untagged_from_the_attribute_not_the_text() {
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

    let parsed = public_enums(Path::new("probe.rs"), source).expect("probe should parse");
    assert_eq!(
        parsed
            .iter()
            .filter(|found| found.untagged)
            .map(|found| found.name.as_str())
            .collect::<Vec<_>>(),
        vec!["PublicWire"],
        "parsed: {parsed:#?}"
    );
}

/// Serde's documented **nested** container-attribute forms, which a
/// `parse_nested_meta` callback could not consume.
///
/// Both were reproduced before the fix. `bound(...)` / `rename(...)` made the
/// parse fail mid-attribute, and because that error was discarded, every key
/// after the nested entry went unseen:
///
/// - `#[serde(bound(...), untagged)]` reported `untagged = false` — a **silent**
///   bypass of the untagged allowlist, and provider directories are excluded from
///   the positive representation scan, so nothing else would have caught it
/// - `#[serde(rename(...), tag = "type")]` reported no representation — a
///   spurious violation
#[test]
fn ast_scanner_reads_nested_serde_metadata() {
    let source = r#"
        #[derive(serde::Serialize, serde::Deserialize)]
        #[serde(bound(serialize = "T: serde::Serialize"), untagged)]
        pub enum NestedThenUntagged<T> { A(T) }

        #[derive(serde::Serialize, serde::Deserialize)]
        #[serde(rename(serialize = "wire"), tag = "type")]
        pub enum NestedThenTag { A(u32) }

        #[derive(serde::Serialize, serde::Deserialize)]
        #[serde(bound(serialize = "T: serde::Serialize"))]
        pub enum NestedOnly<T> { A(T) }
    "#;

    let parsed = public_enums(Path::new("probe.rs"), source).expect("probe should parse");
    let by_name = |name: &str| {
        parsed
            .iter()
            .find(|found| found.name == name)
            .unwrap_or_else(|| panic!("{name} missing from {parsed:#?}"))
    };

    assert!(
        by_name("NestedThenUntagged").untagged,
        "a key after a nested entry must still be seen, or the untagged allowlist \
         is bypassed silently: {parsed:#?}"
    );
    assert!(by_name("NestedThenTag").declares_representation);
    // `bound` alone is neither a tag nor untagged, so this one legitimately owes
    // a representation.
    assert!(!by_name("NestedOnly").declares_representation);
    assert!(!by_name("NestedOnly").untagged);
}

/// A malformed attribute is a hard failure, not a quiet narrowing of the policy.
#[test]
fn ast_scanner_propagates_attribute_parse_errors() {
    let source = r#"
        #[derive(serde::Serialize)]
        #[serde(tag = )]
        pub enum BrokenAttribute { A(u32) }
    "#;
    assert!(
        public_enums(Path::new("probe.rs"), source).is_err(),
        "an unreadable serde attribute must fail the scan"
    );
}

#[test]
fn ast_scanner_treats_an_unparsable_file_as_a_failure() {
    assert!(
        public_enums(Path::new("broken.rs"), "pub enum Broken { ").is_err(),
        "a file the scanner cannot parse must not read as compliance"
    );
}
