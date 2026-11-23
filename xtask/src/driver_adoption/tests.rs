use super::*;

fn source(path: &str, shipped: &str) -> Source {
    Source {
        path: path.to_owned(),
        shipped: shipped.to_owned(),
    }
}

#[test]
fn the_committed_allowlist_parses() {
    assert!(!parse_allowlist(SERDE_POLICY_ALLOWLIST).unwrap().is_empty());
    assert!(parse_allowlist("providers/x.rs | snippet").is_err());
}

#[test]
fn the_triage_guard_flags_a_restated_policy_table() {
    let home = source(
        "crates/rig-core/src/driver.rs",
        "match event { WireEvent::Unknown(_) => {} }",
    );
    let provider = source(
        "crates/rig-core/src/providers/x/streaming.rs",
        "fn f() {\n    if let WireEvent::Corrupt(_) = event {}\n}\n",
    );
    let error = every_triage_site_runs_on_the_single_policy_driver(&[home, provider]).unwrap_err();
    assert!(
        error.contains("crates/rig-core/src/providers/x/streaming.rs:2:"),
        "{error}"
    );
    // A tree whose homes never name the variants is a vacuous scan.
    assert!(
        every_triage_site_runs_on_the_single_policy_driver(&[source("a.rs", "")])
            .unwrap_err()
            .contains("vacuous")
    );
}

#[test]
fn the_warn_scan_catches_every_capture_spelling() {
    // Positional, named, format-string, and rustfmt-wrapped multi-line
    // captures are all flagged...
    for leaking in [
        "tracing::warn!(?frame, \"skipping\");",
        "tracing::warn!(payload = ?frame, \"skipping\");",
        "tracing::warn!(\"bad frame: {:?}\", frame);",
        "tracing::warn!(\n    ?frame,\n    \"skipping\"\n);",
        "tracing::warn!(\n    payload =\n        ?frame,\n    \"skipping\"\n);",
        // Inline format captures Debug-print just as loudly.
        "tracing::warn!(\"bad frame: {frame:?}\");",
        "tracing::warn!(\"bad frame: {frame:#?}\");",
        // Hex, width and precision specs still Debug-print the payload.
        "tracing::warn!(\"bad frame: {frame:x?}\");",
        "tracing::warn!(\"bad frame: {frame:X?}\");",
        "tracing::warn!(\"bad frame: {frame:>10?}\");",
        "tracing::warn!(\"bad frame: {frame:.3?}\");",
        // A stray `)` in a comment must not truncate the scanned body
        // ahead of the capture.
        "tracing::warn!(\n    // see step 3)\n    ?frame,\n    \"skipping\"\n);",
        "tracing::warn!(\n    /* note (a) */\n    ?frame,\n    \"skipping\"\n);",
        // `event!` at WARN level is the same macro.
        "tracing::event!(tracing::Level::WARN, ?frame, \"skipping\");",
    ] {
        let start = leaking
            .find("warn!")
            .or_else(|| leaking.find("event!"))
            .expect("fixture contains a warn-level macro");
        assert!(
            body_debug_captures(&macro_body(leaking, start)),
            "must flag: {leaking}"
        );
    }
    // ...while structural logging, discriminant captures, and prose
    // question marks are not.
    for clean in [
        "tracing::warn!(kind, payload_bytes = size, \"skipping unmodeled wire payload\");",
        "tracing::warn!(step_index, \"arguments_delta for an unopened step?\");",
        "tracing::warn!(\n    kind,\n    payload_bytes = bytes,\n    \"skipping\"\n);",
        "tracing::warn!(\n    delta = ?std::mem::discriminant(&unknown),\n    \"skipping\"\n);",
        // Display captures and string forms that could desync a naive
        // parser stay clean.
        "tracing::warn!(\"dropped {count} frames\");",
        "tracing::warn!(r#\"marker \"quoted\" text\"#, count);",
    ] {
        assert!(
            !body_debug_captures(&macro_body(
                clean,
                clean.find("warn!").expect("fixture contains warn!")
            )),
            "must not flag: {clean}"
        );
    }
}

/// A synthetic streaming module with an unlisted `serde_json::from_str` and
/// a `#[serde(other)]` fails, and the allowlist covers exactly what it names.
#[test]
fn serde_policy_scanner_catches_raw_parses() {
    let bad_source = r#"
        fn sneak_a_policy_site(data: &str) {
            let value = serde_json::from_str::<serde_json::Value>(data);
        }
        #[serde(other)]
        struct Marker;
    "#;

    let violations = scan_streaming_source(
        "crates/rig-core/src/providers/fake/streaming.rs",
        bad_source,
        &mut [],
    );
    assert_eq!(
        violations.len(),
        2,
        "the scanner must flag both the raw parse and the serde(other) fallback: {violations:?}"
    );

    // An allowlist entry covers its named line and nothing else.
    let mut allowlist = parse_allowlist(
        "providers/fake/streaming.rs | serde_json::from_str::<serde_json::Value>(data) | synthetic",
    )
    .unwrap();
    let violations = scan_streaming_source(
        "crates/rig-core/src/providers/fake/streaming.rs",
        bad_source,
        &mut allowlist,
    );
    assert_eq!(
        violations.len(),
        1,
        "only the serde(other) line stays flagged"
    );
    assert!(allowlist.iter().all(|entry| entry.used));

    // The classify layer's own file is out of scope by construction, even
    // though its content mentions the machinery.
    assert!(!is_serde_wall_target(
        "crates/rig-core/src/providers/internal/wire.rs",
        "fn classify(frame: WireFrame) -> WireEvent { todo!() }",
    ));
    assert!(is_serde_wall_target(
        "crates/rig-core/src/providers/openai/responses_api/websocket.rs",
        "",
    ));
}

/// A helper whose basename the streaming pattern misses is scanned once its
/// shipped content touches the wire machinery, and not before.
#[test]
fn serde_wall_scopes_by_machinery_content() {
    let compat = "crates/rig-core/src/providers/openai/wire/chat/finish.rs";
    assert!(
        is_serde_wall_target(compat, "use crate::wire::WireFrame;"),
        "a compat helper referencing the machinery must be scanned"
    );
    let future_helper = "crates/rig-core/src/providers/somegateway/sse.rs";
    assert!(
        is_serde_wall_target(
            future_helper,
            "fn classify(&self, frame: WireFrame) -> WireEvent<Chunk> {"
        ),
        "any future compat/sse helper opts in the moment it names the machinery"
    );
    assert!(
        !is_serde_wall_target(future_helper, "fn plain_request_builder() {}"),
        "machinery-free helpers stay out of scope"
    );
}

/// The named-policy-home exemption is by full path suffix: a foreign
/// `adapter.rs` is scanned by both guards.
#[test]
fn foreign_adapter_files_are_not_exempt() {
    let foreign = "crates/rig-bedrock/src/streaming/adapter.rs";
    assert!(
        !is_policy_home(foreign),
        "guard 1 must scan a foreign adapter.rs for restated policy tables"
    );
    assert!(
        is_serde_wall_target(foreign, "match event { WireEvent::Unknown(_) => {} }"),
        "guard 2 must scan a foreign adapter.rs that touches the machinery"
    );
    assert!(is_policy_home(
        "crates/rig-core/src/providers/internal/wire.rs"
    ));
}

/// A provider's own `wire.rs` (and the modules under `wire/`) is the triage
/// guard's policy home and still a serde wall target, so one predicate
/// cannot serve both guards.
#[test]
fn the_two_guards_have_different_scopes() {
    let classify_layer = "fn classify(frame: WireFrame) -> WireEvent<Event> { todo!() }";
    for path in [
        "crates/rig-core/src/providers/x/wire.rs",
        "crates/rig-core/src/providers/x/wire/chat.rs",
    ] {
        assert!(
            is_policy_home(path),
            "{path}: a provider's classify layer names its own triage variants"
        );
        assert!(
            is_serde_wall_target(path, classify_layer),
            "{path}: the classify layer still owes the serde wall a \
             justified allowlist entry per raw parse"
        );
    }

    // The four named homes are the exemption the two guards do share.
    for suffix in ALLOWED_POLICY_HOMES {
        let path = format!("crates/{suffix}");
        assert!(is_policy_home(&path), "{suffix} states the policy table");
        assert!(
            !is_serde_wall_target(&path, classify_layer),
            "{suffix} is exempt from both guards"
        );
    }
}

/// Only an attribute in attribute position removes an item; a comment
/// mentioning `#[cfg(test)]` does not exempt the code after it.
#[test]
fn shipped_portion_ignores_cfg_test_mentions_in_comments() {
    let source = "\
/// This helper is exercised under #[cfg(test)] elsewhere.
fn shipped_code() { let _ = WireEvent::Unknown; }
#[cfg(test)]
mod tests {
    fn test_only() { let _ = WireEvent::Corrupt; }
}
";
    let shipped = shipped_portion(source);
    assert!(
        shipped.contains("WireEvent::Unknown"),
        "code after a doc-comment mention still ships"
    );
    assert!(
        !shipped.contains("WireEvent::Corrupt"),
        "the real attribute-position marker still removes its item"
    );

    // Indented attribute position (inside an impl block) removes its item.
    let indented = "fn a() {}\n    #[cfg(test)]\n    fn b() {}\n";
    assert_eq!(shipped_portion(indented), "fn a() {}\n\n\n");

    // No marker at all: the whole file ships.
    let plain = "fn a() {}\n";
    assert_eq!(shipped_portion(plain), plain);

    // A block comment quoting the attribute at column 0 is inert.
    let block_commented = "\
/*
#[cfg(test)]
*/
fn shipped_code() { let _ = WireEvent::Unknown; }
";
    assert!(
        shipped_portion(block_commented).contains("WireEvent::Unknown"),
        "an attribute inside a block comment must not gate the code below it"
    );
}

#[test]
fn declares_module_matches_whole_line_declarations_only() {
    assert!(declares_module("mod tests;\n", "tests"));
    assert!(declares_module("    pub(crate) mod tests;\n", "tests"));
    assert!(declares_module("pub mod tests ;\n", "tests"));
    assert!(!declares_module("mod tests {\n", "tests"));
    assert!(!declares_module("mod tests_extra;\n", "tests"));
    assert!(!declares_module("// mod tests;\n", "tests"));
}

/// A sibling-file module is test-only exactly when every path rustc would
/// take to its parent declares it under `#[cfg(test)]`.
#[test]
fn is_test_only_module_follows_the_parent_declaration() {
    let root = std::env::temp_dir().join(format!("xtask-driver-adoption-{}", std::process::id()));
    let src = root.join("src");
    std::fs::create_dir_all(src.join("gated")).unwrap();
    std::fs::create_dir_all(src.join("open")).unwrap();

    // `src/gated.rs` declares its sibling under a gate; `src/open.rs` does not.
    std::fs::write(
        src.join("gated.rs"),
        "pub fn shipped() {}\n\n#[cfg(test)]\nmod tests;\n",
    )
    .unwrap();
    std::fs::write(src.join("gated").join("tests.rs"), "use super::*;\n").unwrap();
    std::fs::write(
        src.join("open.rs"),
        "mod helpers;\n#[cfg(test)]\nmod tests;\n",
    )
    .unwrap();
    std::fs::write(src.join("open").join("helpers.rs"), "").unwrap();
    std::fs::write(src.join("open").join("tests.rs"), "").unwrap();
    // Crate root: `src/lib.rs` gating `src/tests.rs`, and a comment mention
    // that must not count as a declaration.
    std::fs::write(
        src.join("lib.rs"),
        "// mod imaginary;\nmod open;\nmod gated;\n#[cfg(test)]\nmod tests;\n",
    )
    .unwrap();
    std::fs::write(src.join("tests.rs"), "").unwrap();
    std::fs::write(src.join("orphan.rs"), "").unwrap();

    let results = [
        is_test_only_module(&src.join("gated").join("tests.rs")),
        is_test_only_module(&src.join("open").join("tests.rs")),
        is_test_only_module(&src.join("open").join("helpers.rs")),
        is_test_only_module(&src.join("tests.rs")),
        is_test_only_module(&src.join("open.rs")),
        is_test_only_module(&src.join("gated.rs")),
        is_test_only_module(&src.join("orphan.rs")),
    ];
    std::fs::remove_dir_all(&root).unwrap();
    assert_eq!(results, [true, true, false, true, false, false, false]);
}

/// Gating is item-scoped: a `#[cfg(test)]` helper mid-file removes only
/// that helper, and everything after it is scanned again.
#[test]
fn shipped_portion_is_item_scoped() {
    let source = "\
fn before() { let _ = WireEvent::Unknown; }

#[cfg(test)]
pub(super) fn gated_helper() -> u8 {
    let braces_in_a_string = \"unbalanced { brace\";
    let raw = r#\"also unbalanced } here\"#;
    let _ = (braces_in_a_string, raw);
    7
}

fn after() { let _ = drive_frames(); }

#[cfg(test)]
mod tests {
    fn test_only() { let _ = WireEvent::Corrupt; }
}

fn last() { let _ = triage_frame(); }
";
    let shipped = shipped_portion(source);
    assert!(shipped.contains("WireEvent::Unknown"), "{shipped}");
    assert!(
        shipped.contains("drive_frames"),
        "shipped code AFTER a gated helper must stay visible: {shipped}"
    );
    assert!(
        shipped.contains("triage_frame"),
        "shipped code after a gated `mod tests` must stay visible too: {shipped}"
    );
    assert!(
        !shipped.contains("gated_helper") && !shipped.contains("WireEvent::Corrupt"),
        "gated items themselves must be blanked: {shipped}"
    );
    assert_eq!(
        shipped.lines().count(),
        source.lines().count(),
        "blanking must preserve line numbering so violations cite real lines"
    );

    // Content scoping sees machinery named only after a gated item.
    assert!(
        is_serde_wall_target(
            "crates/rig-core/src/providers/somegateway/compat.rs",
            &shipped,
        ),
        "a file whose only machinery reference sits after a gated helper must still be scanned"
    );

    // A brace-less gated item ends at its semicolon, not at end of file.
    let brace_less = "#[cfg(test)]\nuse std::fmt;\nfn after() { let _ = WireEvent::Unknown; }\n";
    assert!(shipped_portion(brace_less).contains("WireEvent::Unknown"));
}
