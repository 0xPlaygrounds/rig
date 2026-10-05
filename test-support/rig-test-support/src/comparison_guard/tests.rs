use super::*;

const VOLATILE: &[&str] = &[
    "created",
    "created_at",
    "completed_at",
    "updated",
    "updated_at",
];

fn findings(source: &str) -> Vec<String> {
    exact_volatile_comparisons(source, VOLATILE).expect("valid Rust")
}

#[test]
fn a_recorded_document_compared_exactly_is_one_finding() {
    for (case, source, finding) in [
        (
            // A raw-capture cell's key-by-key loop.
            "a key by key document loop is flagged",
            r#"
            fn raw_is_the_verbatim_response_body() {
                for key in body.as_object().expect("object").keys().filter(|key| key.as_str() != "id") {
                    assert_eq!(raw.get(key), body.get(key), "raw should carry `{key}` unchanged");
                }
            }
        "#,
            "raw_is_the_verbatim_response_body:",
        ),
        (
            // A terminal-metadata cell comparing a live index with a recorded document.
            "a whole recorded document compared exactly is flagged",
            r#"
            fn assert_cell() {
                assert_eq!(
                    observation.raw["additional_params"],
                    recorded_additional_params(&chunks),
                    "{scenario}: every unmodeled top-level SSE field"
                );
                assert_eq!(recorded_body(&chunks)["model"], raw["model"]);
                // A recorded premise checked against a literal or an expectation.
                assert_eq!(recorded_provider(scenario, transport), "OpenAI");
                assert_eq!(recorded_finish_reason(scenario, transport), expected_finish);
                if CassetteMode::current() == CassetteMode::Replay {
                    assert_eq!(raw, recorded_body(&chunks));
                }
            }
        "#,
            "assert_cell: compares a whole recorded document",
        ),
    ] {
        let found = findings(source);
        assert_eq!(found.len(), 1, "{case}: {found:?}");
        assert!(found[0].starts_with(finding), "{case}: {found:?}");
    }
}

#[test]
fn sources_the_guard_leaves_alone_have_no_findings() {
    for (case, source) in [
        (
            "a loop that consults the volatile keys passes",
            r#"
            fn helper() {
                for key in recorded.as_object().unwrap().keys() {
                    if is_volatile_json_key(key) {
                        assert_wire_value_matches(live, recorded, key);
                    } else {
                        assert_eq!(live.get(key), recorded.get(key));
                    }
                }
            }
        "#,
        ),
        (
            "a replay arm or branch may compare exactly",
            r#"
            fn created_round_trips() {
                match CassetteMode::current() {
                    CassetteMode::Replay => assert_eq!(params.get("created"), Some(&created)),
                    CassetteMode::Record => assert!(params.get("created").is_some()),
                }
                if mode == CassetteMode::Replay {
                    assert_eq!(raw["created_at"], body["created_at"]);
                }
            }
        "#,
        ),
        (
            "other keys and loops are left alone",
            r#"
            fn fine() {
                assert_eq!(raw["model"], body["model"]);
                assert_eq!(notes.first(), Some(&(0, serde_json::json!("created"))));
                for name in names.iter() {
                    assert_eq!(a.get(name), b.get(name));
                }
                for key in map.keys() {
                    assert!(key.len() > 0);
                }
            }
        "#,
        ),
        (
            "literal keys without a volatile one are left alone",
            r#"
            fn fine() {
                // Recorded text compared with a decoded field is not a document.
                let wire_text = recorded_text(scenario, transport);
                assert_eq!(observation.text, wire_text);
                assert_eq!(observation.reasoning, recorded_reasoning(SCENARIO));
                for field in ["id", "model"] {
                    assert_eq!(raw.get(field), body.get(field));
                }
                assert!(raw["model"] == body["model"]);
            }
        "#,
        ),
    ] {
        let found = findings(source);
        assert!(found.is_empty(), "{case}: {found:?}");
    }
}

#[test]
fn volatile_comparisons_outside_a_replay_branch_are_flagged() {
    for (case, source, count) in [
        (
            "an exact volatile comparison is flagged outside replay",
            r#"
            fn terminal() {
                assert_eq!(raw["created"], body["created"]);
                assert_ne!(params.get("updated_at"), body.get("updated_at"));
                assert_eq!(raw["Created"], body["Created"]);
                // A presence check holds in both modes.
                assert_ne!(params.get("updated_at"), None);
            }
        "#,
            3,
        ),
        (
            "a recording branch is not a replay branch",
            r#"
            fn record_only() {
                if mode != CassetteMode::Replay {
                    assert_eq!(raw["created"], body["created"]);
                }
                if !matches!(mode, CassetteMode::Replay) {
                    assert_eq!(raw["created_at"], body["created_at"]);
                }
                if matches!(mode, CassetteMode::Replay) && strict {
                    assert_eq!(raw["updated"], body["updated"]);
                }
            }
        "#,
            2,
        ),
    ] {
        let found = findings(source);
        assert_eq!(found.len(), count, "{case}: {found:?}");
    }
}

#[test]
fn the_remaining_exact_shapes_are_flagged() {
    let source = r#"
        fn bound_first() {
            let recorded = recorded_additional_params(&chunks);
            assert_eq!(observation.raw["additional_params"], recorded);
        }
        fn literal_keys() {
            for field in ["created", "id"] {
                assert_eq!(raw.get(field), body.get(field));
            }
        }
        fn plain_assert() {
            assert!(raw["created"] == body["created"]);
        }
        fn mentions_only() {
            let _ = is_volatile_json_key;
            for key in body.as_object().unwrap().keys() {
                assert_eq!(raw.get(key), body.get(key));
            }
        }
        struct Cell;
        impl Cell {
            fn method() {
                assert_eq!(raw["created"], body["created"]);
            }
        }
    "#;
    let found = findings(source);
    let functions: Vec<&str> = found
        .iter()
        .map(|finding| finding.split(':').next().unwrap_or_default())
        .collect();
    assert_eq!(
        functions,
        [
            "bound_first",
            "literal_keys",
            "plain_assert",
            "mentions_only",
            "method"
        ],
        "{found:?}"
    );
}

#[test]
fn only_the_arm_pattern_decides_a_replay_arm() {
    let source = r#"
        fn arms() {
            match CassetteMode::current() {
                CassetteMode::Record => {
                    if mode == CassetteMode::Replay {
                        unreachable!();
                    }
                    assert_eq!(raw["created"], body["created"]);
                }
                CassetteMode::Replay => assert_eq!(raw["created"], body["created"]),
            }
            match mode {
                CassetteMode::Record | CassetteMode::Replay => {
                    assert_eq!(raw["created_at"], body["created_at"]);
                }
            }
        }
    "#;
    assert_eq!(findings(source).len(), 2, "{:?}", findings(source));
}
