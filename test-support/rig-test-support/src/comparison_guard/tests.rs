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

fn functions(found: &[String]) -> Vec<&str> {
    found
        .iter()
        .map(|finding| finding.split(':').next().unwrap_or_default())
        .collect()
}

#[test]
fn the_else_of_a_branch_that_rules_replay_out_runs_in_replay() {
    let source = r#"
        fn ruled_out() {
            if mode != CassetteMode::Replay {
                assert!(raw.get("created").is_some());
            } else {
                assert_eq!(raw["created"], body["created"]);
            }
            if !matches!(mode, CassetteMode::Replay) {
                assert!(raw.get("created_at").is_some());
            } else {
                assert_eq!(raw["created_at"], body["created_at"]);
            }
        }
        fn replay_first() {
            // The `else` of a replay branch is the recording pass.
            if mode == CassetteMode::Replay {
                assert_eq!(raw["updated"], body["updated"]);
            } else {
                assert_eq!(raw["updated_at"], body["updated_at"]);
            }
        }
    "#;
    let found = findings(source);
    assert_eq!(functions(&found), ["replay_first"], "{found:?}");
}

#[test]
fn a_rebound_name_no_longer_holds_a_recorded_document() {
    let source = r#"
        fn rebound() {
            let recorded = recorded_additional_params(&chunks);
            let recorded = recorded["service_tier"].clone();
            assert_eq!(observation.raw["service_tier"], recorded);
        }
        fn rebound_by_a_tuple() {
            let body = recorded_json_response("x", "cell");
            let (body, _other) = (live_document(), 0);
            assert_eq!(raw["extras"], body);
        }
        fn still_recorded() {
            let recorded = recorded_additional_params(&chunks);
            assert_eq!(observation.raw["additional_params"], recorded);
        }
        // A rebinding inside a block ends with the block.
        fn shadowed_inside() {
            let body = recorded_json_response("x", "cell");
            for item in items { let body = item; let _ = body; }
            { let body = live_document(); }
            assert_eq!(raw["extras"], body);
        }
        // A loop, `while let`, match arm or closure parameter rebinds the
        // name within its own scope.
        fn rebound_by_inner_patterns() {
            let body = recorded_json_response("x", "cell");
            for body in items { assert_eq!(raw["extras"], body); }
            for (body, _) in pairs { assert_eq!(raw["extras"], body); }
            while let Some(body) = next() { assert_eq!(raw["extras"], body); }
            match live() { Some(body) => assert_eq!(raw["extras"], body), None => {} }
            items.iter().for_each(|body| assert_eq!(raw["extras"], body));
        }
    "#;
    let found = findings(source);
    assert_eq!(
        functions(&found),
        ["still_recorded", "shadowed_inside"],
        "{found:?}"
    );
}

#[test]
fn a_pattern_that_also_matches_recording_is_not_replay() {
    let source = r#"
        fn either_mode() {
            if matches!(mode, CassetteMode::Replay | CassetteMode::Record) {
                assert_eq!(raw["created"], body["created"]);
            }
            if matches!(mode, Some(CassetteMode::Replay | _)) {
                assert_eq!(raw["created_at"], body["created_at"]);
            }
            if let (CassetteMode::Record, CassetteMode::Replay) = (mode, fallback) {
                assert_eq!(raw["updated"], body["updated"]);
            }
        }
        fn replay_only() {
            if matches!(mode, Some(CassetteMode::Replay) if strict) {
                assert_eq!(raw["created"], body["created"]);
            }
            if let Some(CassetteMode::Replay) = mode {
                assert_eq!(raw["created_at"], body["created_at"]);
            }
        }
    "#;
    let found = findings(source);
    assert_eq!(
        functions(&found),
        ["either_mode", "either_mode", "either_mode"],
        "{found:?}"
    );
}

#[test]
fn an_entry_by_entry_loop_over_a_recorded_object_is_flagged() {
    let source = r#"
        fn entries() {
            let body = recorded_json_response("x", "cell");
            for (key, value) in body.as_object().unwrap() {
                assert_eq!(raw.get(key), Some(value), "{key}");
            }
        }
        fn bound_object() {
            let body = recorded_json_response("x", "cell");
            let object = body.as_object().expect("an object");
            for (key, value) in object {
                assert_eq!(raw.get(key), Some(value));
            }
        }
        fn live_entries_against_a_recording() {
            let body = recorded_json_response("x", "cell");
            for (key, value) in raw.as_object().unwrap() {
                assert_eq!(body.get(key), Some(value));
            }
        }
        fn entries_through_iter() {
            for (key, value) in recorded_json_response("x", "cell").as_object().unwrap().iter() {
                assert!(value == &live[key]);
            }
        }
        fn entries_treated() {
            let body = recorded_json_response("x", "cell");
            for (key, value) in body.as_object().unwrap() {
                if !is_volatile_json_key(key) {
                    assert_eq!(raw.get(key), Some(value));
                }
            }
        }
        // A live document against a re-serialization of itself holds in
        // both modes.
        fn live_against_itself() {
            let reserialized = serde_json::to_value(&typed).unwrap();
            for (field, value) in reserialized.as_object().unwrap() {
                assert_eq!(raw.get(field), Some(value));
            }
        }
    "#;
    let found = findings(source);
    assert_eq!(
        functions(&found),
        [
            "entries",
            "bound_object",
            "live_entries_against_a_recording",
            "entries_through_iter"
        ],
        "{found:?}"
    );
}
