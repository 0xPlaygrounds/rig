use super::*;

fn facts(name: &str, source: &str) -> Facts {
    let item: ItemFn = syn::parse_str(source).expect("the test function parses");
    of(name, &item, false)
}

fn reason(name: &str, source: &str) -> Option<Reason> {
    facts(name, source).contract.map(|contract| contract.reason)
}

#[test]
fn a_loop_over_a_table_of_cases_is_table_driven() {
    let cases = [
        (
            "array",
            "for (input, want) in [(1, 2), (3, 4)] { assert_eq!(input + 1, want); }",
            true,
        ),
        (
            "reference",
            "for case in &[1, 2] { assert!(*case > 0); }",
            true,
        ),
        (
            "vec",
            "for case in vec![1, 2].into_iter() { assert!(case > 0); }",
            true,
        ),
        (
            "named",
            "for (name, case) in CASES.iter().enumerate() { assert!(case.ok(), \"{name}\"); }",
            true,
        ),
        ("range", "for n in 0..3 { assert!(n < 3); }", false),
        ("none", "assert_eq!(1 + 1, 2);", false),
    ];
    for (name, body, table) in cases {
        let found = facts("rig x", &format!("#[test] fn t() {{ {body} }}"));
        assert_eq!(found.table, table, "{name}");
    }
}

#[test]
fn size_counts_the_attributes_and_the_body() {
    let found = facts(
        "rig x",
        "#[test]\n#[ignore]\nfn t() {\n    assert!(true);\n}",
    );
    assert_eq!(found.lines, 5);
}

#[test]
fn contract_reasons_follow_the_name_and_the_tokens() {
    let cases = [
        (
            "wasm",
            "rig x",
            "#[wasm_bindgen_test] #[test] fn t() {}",
            Some(Reason::Wasm),
        ),
        (
            "dual",
            "rig x",
            "#[cfg_attr(target_family = \"wasm\", wasm_bindgen_test::wasm_bindgen_test)] #[test] fn t() {}",
            Some(Reason::Wasm),
        ),
        (
            "trybuild",
            "rig x",
            "#[test] fn t() { let t = trybuild::TestCases::new(); }",
            Some(Reason::CompileFail),
        ),
        (
            "api",
            "rig::api_surface x",
            "#[test] fn t() { assert!(ok()); }",
            Some(Reason::ApiShape),
        ),
        (
            "scrub name",
            "rig a::redacts_the_key",
            "#[test] fn t() { assert!(ok()); }",
            Some(Reason::Security),
        ),
        (
            "scrub body",
            "rig x",
            "#[test] fn t() { assert!(scrub(text).is_empty()); }",
            Some(Reason::Security),
        ),
        (
            "round trip name",
            "rig a::history_round_trips",
            "#[test] fn t() { assert!(ok()); }",
            Some(Reason::StoredFormat),
        ),
        (
            "round trip body",
            "rig x",
            "#[test] fn t() { let v = serde_json::to_value(&x).unwrap(); let y: X = serde_json::from_value(v).unwrap(); assert_eq!(x, y); }",
            Some(Reason::StoredFormat),
        ),
        (
            "error message",
            "rig x",
            "#[test] fn t() { let error = run().unwrap_err(); assert_eq!(error.to_string(), \"boom\"); }",
            Some(Reason::ErrorMessage),
        ),
        (
            "panic message only",
            "rig x",
            "#[test] fn t() { let error = run().unwrap_err(); assert!(error.is_fatal(), \"{}\", error.to_string()); }",
            None,
        ),
        (
            "decode only",
            "rig x",
            "#[test] fn t() { let x: X = serde_json::from_str(\"{}\").unwrap(); assert_eq!(x.n, 1); }",
            None,
        ),
        (
            "golden name",
            "rig core::x_effect_log_is_the_golden_fixture",
            "#[test] fn t() { assert!(ok()); }",
            Some(Reason::StoredFormat),
        ),
        (
            "fixture path",
            "rig x",
            "#[test] fn t() { let text = read(\"../fixtures/effects/a.effects.json\").unwrap(); assert!(text.ok()); }",
            Some(Reason::StoredFormat),
        ),
        (
            "compile only",
            "rig x",
            "#[test] fn t() { let _ = AgentRun::new(\"x\"); }",
            Some(Reason::ApiShape),
        ),
        (
            "a fallible call is a check",
            "rig x",
            "#[test] fn t() -> Result<(), E> { run()?; Ok(()) }",
            None,
        ),
        (
            "rendered",
            "rig x",
            "#[test] fn t() { assert_eq!(doc.to_string(), \"<file>\"); }",
            Some(Reason::RenderedText),
        ),
        (
            "rendered message only",
            "rig x",
            "#[test] fn t() { assert!(doc.ok(), \"{}\", doc.to_string()); }",
            None,
        ),
    ];
    for (name, test, source, want) in cases {
        assert_eq!(reason(test, source), want, "{name}");
    }
}

#[test]
fn an_error_message_contract_is_its_literal_assertions() {
    let found = facts(
        "rig x",
        "#[test] fn t() { let error = run().unwrap_err(); assert!(error.is_fatal()); assert_eq!(error.to_string(), \"boom\"); }",
    );
    let contract = found.contract.expect("an error-message contract");
    assert_eq!(
        contract.assertions,
        ["assert_eq!(error . to_string () , \"boom\")"]
    );
    assert_eq!(found.assertions.len(), 2);
}

#[test]
fn only_a_positive_wasm_cfg_gates_to_wasm() {
    let gated = |source: &str| {
        let item: syn::ItemMod = syn::parse_str(source).expect("the module parses");
        gates_wasm(&item.attrs)
    };
    assert!(gated("#[cfg(target_family = \"wasm\")] mod m;"));
    assert!(gated("#[cfg(all(test, target_arch = \"wasm32\"))] mod m;"));
    assert!(!gated("#[cfg(not(target_family = \"wasm\"))] mod m;"));
    assert!(!gated("#[cfg(test)] mod m;"));
}
