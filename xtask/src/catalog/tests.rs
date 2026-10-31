use serde_json::json;

use super::guard::{self, Comparison, Reading, Reasoning};
use super::{Rows, generate, is_rfc3339_utc, models_dev_rows, pin, read_rows, render, rfc3339};

/// A row's `rig` facts without the keys it pins.
fn facts(row: &serde_json::Map<String, serde_json::Value>) -> serde_json::Value {
    let mut facts = row.get("rig").cloned().unwrap_or_default();
    if let Some(facts) = facts.as_object_mut() {
        facts.retain(|candidate, _| candidate != "pinned");
    }
    facts
}

fn models_dev() -> serde_json::Value {
    json!({
        "anthropic": {"name": "Anthropic", "models": {
            "claude-x-1": {
                "id": "claude-x-1", "name": "Claude X 1", "family": "claude",
                "reasoning": true,
                "reasoning_options": [{"type": "effort", "values": ["low", "high"], "extra": 1}],
                "modalities": {"input": ["text", "image"], "output": ["text"]},
                "limit": {"context": 200000, "input": 180000, "output": 64000},
                "cost": {
                    "input": 3, "output": 15,
                    "tiers": [{"input": 6, "output": 22.5, "tier": {"type": "context", "size": 200000}}]
                },
                "release_date": "2026-01-01",
                "experimental": {"modes": {"fast": {"cost": {"input": 6, "output": 30, "future_rate": 1}}}},
                "future_field": {"a": 1}
            }
        }},
        "amazon-bedrock": {"models": {
            "us.anthropic.claude-x-1-v1:0": {
                "name": "Claude X 1", "reasoning": true,
                "canonical_model_id": "anthropic/claude-x-1-20260101"
            },
            "anthropic.claude-x-1-v2": {"name": "Claude X 1 v2", "reasoning": true},
            "amazon.nova-pro-v1:0": {"name": "Nova Pro", "canonical_model_id": "amazon/nova-pro"}
        }},
        "openrouter": {"models": {
            "anthropic/claude-x.1": {
                "name": "Claude X 1", "reasoning": true,
                "reasoning_options": [{"type": "toggle"}],
                "canonical_model_id": "anthropic/claude-x-1"
            },
            "openai/gpt-x": {
                "name": "GPT X", "reasoning": true,
                "reasoning_options": [{"type": "effort", "values": ["low", "high", "max"]}]
            },
            "vendor/thinker": {
                "name": "Thinker", "reasoning": true,
                "reasoning_options": [{"type": "budget_tokens", "min": 1024}]
            },
            "vendor/unknown": {"name": "Unknown", "reasoning": true},
            "vendor/plain": {
                "name": "Plain", "reasoning": false,
                "reasoning_options": [{"type": "toggle"}]
            }
        }},
        "google-vertex": {"models": {
            "claude-x-1@default": {"name": "Claude X 1", "canonical_model_id": "anthropic/claude-x-1"}
        }},
        "github-copilot": {"models": {
            "claude-x.1": {"name": "Claude X 1", "canonical_model_id": "anthropic/claude-x-1"},
            "gpt-x": {"name": "GPT X", "canonical_model_id": "openai/gpt-x"}
        }},
        "azure": {"models": {
            "gpt-x": {"name": "GPT X", "canonical_model_id": "openai/gpt-x", "rig": {"cache": ["short"]}}
        }},
        "somebody-else": {"models": {"m": {"name": "M"}}}
    })
}

#[test]
fn sync_keeps_rigs_providers_and_every_field_of_their_rows() {
    let rows = generate(&models_dev(), &json!({})).expect("generates");
    assert_eq!(
        rows.keys().collect::<Vec<_>>(),
        [
            "anthropic",
            "aws_bedrock",
            "azure.openai",
            "copilot",
            "openrouter",
            "vertexai"
        ],
        "models.dev keys are renamed, unknown providers dropped"
    );
    let mut row = models_dev()["anthropic"]["models"]["claude-x-1"].clone();
    if let Some(row) = row.as_object_mut() {
        row.retain(|candidate, _| candidate != "id");
    }
    let claude = serde_json::Value::Object(rows["anthropic"]["claude-x-1"].clone());
    assert_eq!(
        claude, row,
        "every field but the id, which is the row's key"
    );
    assert_eq!(
        super::unread_keys(&rows),
        [
            "experimental.modes.*.cost.future_rate",
            "future_field",
            "reasoning_options.*.extra"
        ]
    );
}

/// A gateway's row that models.dev reaches through Google's API is left
/// out, since rig would send it to another API; its other rows stay.
#[test]
fn a_row_reached_through_an_api_rig_serves_elsewhere_is_left_out() {
    let models_dev = json!({"opencode": {"models": {
        "gemini-x": {"name": "Gemini X", "provider": {"npm": "@ai-sdk/google"}},
        "claude-x": {"name": "Claude X", "provider": {"npm": "@ai-sdk/anthropic"}},
        "glm-x": {"name": "GLM X"}
    }}});
    let rows = generate(&models_dev, &json!({})).expect("generates");
    assert_eq!(
        rows["opencode"].keys().collect::<Vec<_>>(),
        ["claude-x", "glm-x"]
    );
}

#[test]
fn a_reviewed_row_merges_over_models_dev_and_drops_its_review_keys() {
    let review = json!({
        "anthropic": {"models": {
            "claude-x-1": {
                "source": "https://example.com/models",
                "limit": {"output": 128000},
                "rig": {"adaptive_thinking": true}
            },
            "claude-x-0": {"from": "anthropic/claude-x-1", "name": "Claude X 0"}
        }}
    });
    let rows = generate(&models_dev(), &review).expect("generates");
    let claude = &rows["anthropic"]["claude-x-1"];
    assert_eq!(
        claude["limit"],
        json!({"context": 200000, "input": 180000, "output": 128000})
    );
    assert_eq!(claude.get("source"), None);
    let copy = &rows["anthropic"]["claude-x-0"];
    assert_eq!(copy["name"], "Claude X 0");
    assert_eq!(
        copy["limit"], claude["limit"],
        "`from` starts from the named row"
    );
}

#[test]
fn rows_whose_canonical_id_is_an_anthropic_model_take_its_facts() {
    let review = json!({
        "anthropic": {"models": {"claude-x-1": {"rig": {"binds_context": true}}}},
        "vertexai": {"models": {"claude-x-1@default": {"rig": {"binds_context": false}}}}
    });
    let rows = generate(&models_dev(), &review).expect("generates");
    let claude = &rows["anthropic"]["claude-x-1"];

    let bedrock = &rows["aws_bedrock"]["us.anthropic.claude-x-1-v1:0"];
    assert_eq!(
        bedrock["canonical_model_id"], "anthropic/claude-x-1-20260101",
        "the canonical id is kept"
    );
    assert_eq!(
        bedrock["reasoning_options"], claude["reasoning_options"],
        "a dated canonical id finds its model"
    );
    assert_eq!(facts(bedrock), facts(claude));

    let openrouter = &rows["openrouter"]["anthropic/claude-x.1"];
    assert_eq!(facts(openrouter), facts(claude));
    assert_eq!(
        openrouter["reasoning_options"],
        json!([
            {"type": "effort", "values": ["minimal", "low", "medium", "high", "xhigh"]},
            {"type": "budget_tokens"},
            {"type": "toggle"}
        ]),
        "OpenRouter keeps its own reasoning options, as it translates them"
    );

    let vertex = &rows["vertexai"]["claude-x-1@default"];
    assert_eq!(vertex["reasoning_options"], claude["reasoning_options"]);
    assert_eq!(
        facts(vertex),
        json!({"binds_context": false}),
        "a reviewed fact wins over the Anthropic one"
    );

    for (vendor, id) in [
        ("aws_bedrock", "anthropic.claude-x-1-v2"),
        ("aws_bedrock", "amazon.nova-pro-v1:0"),
        ("copilot", "claude-x.1"),
    ] {
        assert_eq!(rows[vendor][id].get("rig"), None, "{vendor}/{id}");
    }
}

#[test]
fn a_review_sync_cannot_read_is_an_error() {
    for review in [
        json!({"anthropic": {"models": {"claude-x-1": {"limits": {}}}}}),
        json!({"anthropic": {"models": {"claude-x-1": {"rig": {"adaptive": true}}}}}),
        json!({"anthropic": {"models": {
            "claude-x-0": {"from": "anthropic/claude-x-1"},
            "claude-x-1": {"from": "anthropic/claude-x-2", "name": "Claude X 1"}
        }}}),
    ] {
        let error = generate(&models_dev(), &review).expect_err("refused");
        assert!(error.contains("claude-x-1"), "{error}");
    }
}

#[test]
fn the_rendered_catalog_reads_back_to_itself() {
    let rows = generate(&models_dev(), &json!({})).expect("generates");
    let rendered = render(&rows);
    let read = read_rows(&serde_json::from_str(&rendered).expect("valid JSON")).expect("rows");
    assert_eq!(render(&read), rendered);
    assert_eq!(
        rendered.lines().count(),
        27,
        "one line per model:\n{rendered}"
    );
}

#[test]
fn a_reviewed_reasoning_control_reaches_the_row() {
    let review = json!({
        "anthropic": {"models": {"claude-x-1": {
            "reasoning_options": [],
            "rig": {"reasoning_control": "none"}
        }}}
    });
    let rows = generate(&models_dev(), &review).expect("generates");
    let claude = &rows["anthropic"]["claude-x-1"];
    assert_eq!(claude["reasoning_options"], json!([]));
    assert_eq!(facts(claude), json!({"reasoning_control": "none"}));
}

#[test]
fn a_reviewed_format_reaches_the_row() {
    let review = json!({
        "anthropic": {"models": {"claude-x-1": {"rig": {"format": "anthropic"}}}}
    });
    let rows = generate(&models_dev(), &review).expect("generates");
    assert_eq!(
        rows["anthropic"]["claude-x-1"]["rig"],
        json!({"format": "anthropic"})
    );
}

#[test]
fn a_gateway_that_translates_reasoning_takes_every_effort_and_a_budget() {
    let rows = generate(&models_dev(), &json!({})).expect("generates");
    let openrouter = &rows["openrouter"];
    assert_eq!(
        openrouter["openai/gpt-x"]["reasoning_options"],
        json!([{"type": "effort", "values": ["minimal", "low", "medium", "high", "xhigh", "max"]}]),
        "an OpenAI upstream takes no budget, and keeps the efforts it lists"
    );
    assert_eq!(
        openrouter["vendor/thinker"]["reasoning_options"],
        json!([
            {"type": "effort", "values": ["minimal", "low", "medium", "high", "xhigh"]},
            {"type": "budget_tokens", "min": 1024}
        ]),
        "a listed budget is kept, and reasoning still cannot be turned off"
    );
    assert_eq!(
        openrouter["vendor/unknown"].get("reasoning_options"),
        None,
        "a row that lists nothing stays unknown"
    );
    assert_eq!(
        openrouter["vendor/plain"]["reasoning_options"],
        json!([{"type": "toggle"}]),
        "a model that does not reason is left alone"
    );
}

#[test]
fn rows_whose_canonical_id_is_an_openai_model_take_its_sampling_rule() {
    let review = json!({
        "openai": {"models": {"gpt-x": {
            "name": "GPT X",
            "rig": {"sampling": "reasoning_off", "reasoning_default": "medium", "cache": ["long"]}
        }}}
    });
    let rows = generate(&models_dev(), &review).expect("generates");
    assert_eq!(
        rows["copilot"]["gpt-x"]["rig"],
        json!({"sampling": "reasoning_off", "reasoning_default": "medium"})
    );
    assert_eq!(
        rows["azure.openai"]["gpt-x"]["rig"],
        json!({"cache": ["short"], "sampling": "reasoning_off", "reasoning_default": "medium"}),
        "the row's own facts stay, and only the sampling rule joins them"
    );
}

/// A mirror vendor lists every reviewed row of its origin, with the
/// mirror's own review laid on top.
#[test]
fn a_mirror_vendor_lists_every_reviewed_row_of_its_origin() {
    let models_dev = json!({"google": {"models": {
        "gemini-x": {"name": "Gemini X", "cost": {"input": 1, "output": 2}}
    }}});
    let review = json!({
        "gcp.gemini": {"models": {"gemini-x": {"rig": {"reasoning_default": "high"}}}},
        "gemini-grpc": {"models": {"gemini-x": {"limit": {"output": 64}}}}
    });
    let rows = generate(&models_dev, &review).expect("generates");
    let mirrored = &rows["gemini-grpc"]["gemini-x"];
    assert_eq!(mirrored["cost"], json!({"input": 1, "output": 2}));
    assert_eq!(facts(mirrored), json!({"reasoning_default": "high"}));
    assert_eq!(
        mirrored["limit"],
        json!({"output": 64}),
        "its own review lays over"
    );
}

/// Every key the review or a join sets is pinned, so a refresh from
/// models.dev keeps it, and nothing else is; `limit` and `cost` pin by part.
/// Pinning again changes nothing, which is what `catalog check` relies on.
#[test]
fn the_keys_the_review_and_joins_set_are_pinned() {
    let review = json!({
        "anthropic": {"models": {
            "claude-x-1": {
                "source": "https://example.com/models",
                "reasoning_options": [{"type": "toggle"}],
                "limit": {"output": 128000},
                "rig": {"adaptive_thinking": true}
            },
            "claude-x-0": {"from": "anthropic/claude-x-1", "name": "Claude X 0"}
        }}
    });
    let rows = generate(&models_dev(), &review).expect("generates");
    let pinned = |vendor: &str, id: &str| {
        rows[vendor][id]
            .get("rig")
            .and_then(|facts| facts.get("pinned"))
            .cloned()
    };
    assert_eq!(
        pinned("anthropic", "claude-x-1"),
        Some(json!(["reasoning_options", "limit.output"])),
        "the reviewed keys, and not `source` or `rig`"
    );
    assert_eq!(
        pinned("anthropic", "claude-x-0"),
        Some(json!([
            "name",
            "family",
            "reasoning",
            "reasoning_options",
            "modalities",
            "limit.context",
            "limit.input",
            "limit.output",
            "cost.input",
            "cost.output",
            "cost.tiers",
            "release_date",
            "experimental"
        ])),
        "a copy pins every key it holds"
    );
    assert_eq!(
        pinned("aws_bedrock", "us.anthropic.claude-x-1-v1:0"),
        Some(json!(["reasoning_options"])),
        "a served row pins what it takes from its origin"
    );
    assert_eq!(
        pinned("openrouter", "vendor/thinker"),
        Some(json!(["reasoning_options"])),
        "a gateway pins the controls it translates"
    );
    for (vendor, id) in [
        ("openrouter", "vendor/unknown"),
        ("openrouter", "vendor/plain"),
        ("copilot", "gpt-x"),
        ("aws_bedrock", "amazon.nova-pro-v1:0"),
    ] {
        assert_eq!(pinned(vendor, id), None, "{vendor}/{id}");
    }
    assert_eq!(rows["copilot"]["claude-x.1"].get("rig"), None);

    let mut again = rows.clone();
    pin(&mut again, &read_rows(&review).expect("rows"));
    assert_eq!(again, rows);
}

/// Sync writes when its data was read in the one form rig-core reads, and
/// `catalog check` accepts that form only.
#[test]
fn the_generation_time_is_written_as_an_rfc_3339_utc_time() {
    use std::time::{Duration, UNIX_EPOCH};
    for (seconds, written) in [
        (0, "1970-01-01T00:00:00Z"),
        (951_868_800, "2000-03-01T00:00:00Z"),
        (1_709_208_000, "2024-02-29T12:00:00Z"),
        (1_791_355_811, "2026-10-07T06:50:11Z"),
    ] {
        let time = UNIX_EPOCH + Duration::from_secs(seconds);
        assert_eq!(rfc3339(time).as_deref(), Ok(written));
        assert!(is_rfc3339_utc(written), "{written}");
    }
    for not_a_time in [
        "",
        "2026-10-07",
        "2026-10-07T06:50:11+00:00",
        "2026-1O-07T06:50:11Z",
    ] {
        assert!(!is_rfc3339_utc(not_a_time), "{not_a_time}");
    }
}

#[test]
fn a_review_of_a_model_models_dev_dropped_needs_its_facts() {
    let review = json!({"anthropic": {"models": {"claude-gone": {"rig": {"cache": ["long"]}}}}});
    let error = generate(&models_dev(), &review).expect_err("refused");
    assert!(error.contains("claude-gone"), "{error}");
    let review = json!({"anthropic": {"models": {"claude-gone": {"name": "Claude Gone"}}}});
    let rows = generate(&models_dev(), &review).expect("generates");
    assert_eq!(rows["anthropic"]["claude-gone"]["name"], "Claude Gone");
}

/// A models.dev-shaped document holding the one row `id` of `provider`.
fn one(provider: &str, id: &str, row: serde_json::Value) -> serde_json::Value {
    json!({provider: {"models": {id: row}}})
}

/// The guard's comparison of the catalog sync wrote from `before` under
/// `old_review` with the one it writes from `after` under `review`.
fn compared(
    before: &serde_json::Value,
    old_review: &serde_json::Value,
    after: &serde_json::Value,
    review: &serde_json::Value,
) -> Comparison {
    let old = generate(before, old_review).expect("old catalog");
    let new = generate(after, review).expect("new catalog");
    guard::compare(
        &old,
        &new,
        &read_rows(review).expect("review"),
        &models_dev_rows(after).expect("models.dev rows"),
    )
}

/// The refused losses, as `vendor/id: what`.
fn refused(comparison: &Comparison) -> Vec<String> {
    comparison
        .refused
        .iter()
        .flat_map(|(key, losses)| {
            losses
                .iter()
                .map(move |loss| format!("{key}: {}", loss.what))
        })
        .collect()
}

/// Sync refuses a row that loses what rig reads, as rig reads it: through
/// a gateway's generated reasoning options too.
#[test]
fn a_lost_capability_is_refused() {
    let effort = |values: serde_json::Value| json!([{"type": "effort", "values": values}]);
    let reasoning =
        |options: serde_json::Value| json!({"reasoning": true, "reasoning_options": options});
    for (provider, id, before, after, lost) in [
        (
            "venice",
            "m",
            reasoning(effort(json!(["low", "high"]))),
            reasoning(json!([])),
            "venice/m: reasoning controls become unknown (were effort low high)",
        ),
        (
            "venice",
            "m",
            reasoning(effort(json!(["none", "low", "high"]))),
            reasoning(effort(json!(["low", "high"]))),
            "venice/m: can no longer turn reasoning off",
        ),
        (
            "venice",
            "m",
            reasoning(json!([{"type": "budget_tokens"}, {"type": "effort", "values": ["low"]}])),
            reasoning(effort(json!(["low"]))),
            "venice/m: reasoning budget lost",
        ),
        (
            "venice",
            "m",
            json!({"reasoning": true}),
            json!({"reasoning": false}),
            "venice/m: stops reasoning",
        ),
        (
            "ollama-cloud",
            "m",
            json!({"modalities": {"input": ["text", "pdf"], "output": ["text"]}}),
            json!({"modalities": {"input": ["text"], "output": ["text"]}}),
            "ollama/m: input lost: pdf",
        ),
        (
            "venice",
            "m",
            json!({"tool_call": true}),
            json!({"tool_call": false}),
            "venice/m: tool calls lost",
        ),
        (
            "openrouter",
            "vendor/m",
            json!({"structured_output": true}),
            json!({}),
            "openrouter/vendor/m: structured output lost",
        ),
        (
            "venice",
            "m",
            json!({"temperature": true}),
            json!({"temperature": false}),
            "venice/m: sampling any -> unknown",
        ),
        (
            "venice",
            "m",
            json!({}),
            json!({"rig": {"sampling": "never"}}),
            "venice/m: sampling unknown -> never",
        ),
        // OpenRouter translates an OpenAI model's listed efforts and adds
        // its own, so `max` is what a refresh loses.
        (
            "openrouter",
            "openai/gpt-x",
            reasoning(effort(json!(["low", "high", "max"]))),
            reasoning(effort(json!(["low", "high"]))),
            "openrouter/openai/gpt-x: effort lost: max",
        ),
    ] {
        let comparison = compared(
            &one(provider, id, before),
            &json!({}),
            &one(provider, id, after),
            &json!({}),
        );
        assert_eq!(refused(&comparison), [lost]);
        assert!(
            comparison
                .refusal()
                .is_some_and(|refusal| refusal.contains(lost))
        );
    }
}

/// What rig reads the same is no change: a list in another order, `toggle`
/// for effort `none`, and an absent flag for `false`.
#[test]
fn a_change_rig_reads_the_same_is_no_change() {
    for (before, after) in [
        (
            json!({"reasoning": true, "reasoning_options": [
                {"type": "effort", "values": ["low", "high"]}, {"type": "budget_tokens"}
            ]}),
            json!({"reasoning": true, "reasoning_options": [
                {"type": "budget_tokens"}, {"type": "effort", "values": ["high", "low"]}
            ]}),
        ),
        (
            json!({"reasoning": true, "reasoning_options": [
                {"type": "toggle"}, {"type": "effort", "values": ["low"]}
            ]}),
            json!({"reasoning": true, "reasoning_options": [
                {"type": "effort", "values": ["none", "low"]}
            ]}),
        ),
        (json!({}), json!({"structured_output": false})),
        (json!({}), json!({"temperature": false})),
    ] {
        let comparison = compared(
            &one("venice", "m", before),
            &json!({}),
            &one("venice", "m", after),
            &json!({}),
        );
        assert_eq!(refused(&comparison), Vec::<String>::new());
        assert!(comparison.gains.is_empty(), "{:?}", comparison.gains);
    }
}

/// A loss is answered by the row's own review setting the key that
/// decides the value, with a source: not by a value a join copies from a
/// reviewed row (a Bedrock row taking Anthropic's reasoning options), not
/// by a key the value does not follow, and not without a source.
#[test]
fn only_a_review_value_with_a_source_answers_a_loss() {
    let claude = json!({
        "anthropic": {"models": {"claude-x-1": {
            "reasoning": true,
            "reasoning_options": [{"type": "effort", "values": ["low", "high"]}]
        }}},
        "amazon-bedrock": {"models": {"us.claude-x-1": {
            "reasoning": true, "canonical_model_id": "anthropic/claude-x-1"
        }}}
    });
    let pdf = one(
        "venice",
        "m",
        json!({"modalities": {"input": ["text", "pdf"], "output": ["text"]}}),
    );
    let text = one(
        "venice",
        "m",
        json!({"modalities": {"input": ["text"], "output": ["text"]}}),
    );
    let sampled = one("venice", "m", json!({"temperature": true}));
    let never = one(
        "venice",
        "m",
        json!({"temperature": true, "rig": {"sampling": "never"}}),
    );
    // A `null` rule is no rule, so `temperature` decides.
    let no_rule = one(
        "venice",
        "m",
        json!({"temperature": false, "rig": {"sampling": null}}),
    );
    let venice = |row: serde_json::Value| one("venice", "m", row);
    for (before, after, review, unanswered) in [
        (
            &claude,
            &claude,
            json!({"anthropic": {"models": {"claude-x-1": {
                "source": "https://example.com/x",
                "reasoning_options": [{"type": "effort", "values": ["high"]}]
            }}}}),
            vec!["aws_bedrock/us.claude-x-1: effort lost: low"],
        ),
        (
            &claude,
            &claude,
            json!({"anthropic": {"models": {"claude-x-1": {
                "source": "",
                "reasoning_options": [{"type": "effort", "values": ["high"]}]
            }}}}),
            vec![
                "anthropic/claude-x-1: effort lost: low",
                "aws_bedrock/us.claude-x-1: effort lost: low",
            ],
        ),
        (
            &pdf,
            &pdf,
            venice(json!({"source": "https://example.com/m", "modalities": null})),
            vec![],
        ),
        (
            &pdf,
            &text,
            venice(
                json!({"source": "https://example.com/m", "modalities": {"output": ["text", "image"]}}),
            ),
            vec!["venice/m: input lost: pdf"],
        ),
        (
            &sampled,
            &never,
            venice(json!({"source": "https://example.com/m", "temperature": true})),
            vec!["venice/m: sampling any -> never"],
        ),
        (
            &sampled,
            &no_rule,
            venice(json!({"source": "https://example.com/m", "temperature": false})),
            vec![],
        ),
    ] {
        let comparison = compared(before, &json!({}), after, &review);
        assert_eq!(refused(&comparison), unanswered, "{review}");
    }
}

/// A removed row, limit and budget changes and a price change are reported,
/// not refused; a narrower reasoning budget is called out.
#[test]
fn removals_limits_and_prices_are_reported_not_refused() {
    let thinker = |output: u64| {
        json!({
            "reasoning": true,
            "reasoning_options": [{"type": "budget_tokens", "min": 1024}],
            "limit": {"context": 200000, "output": output}
        })
    };
    let budget = |max: u64| {
        json!({
            "reasoning": true,
            "reasoning_options": [{"type": "budget_tokens", "min": 1024, "max": max}]
        })
    };
    let before = json!({"venice": {"models": {
        "kept": {"name": "Kept", "cost": {"input": 1, "output": 2}},
        "gone": {"name": "Gone", "tool_call": true},
        "thinker": thinker(64000),
        "wider": budget(4096)
    }}});
    let after = json!({"venice": {"models": {
        "kept": {"name": "Kept", "cost": {"input": 1, "output": 3}},
        "new": {"name": "New"},
        "thinker": thinker(32000),
        "wider": budget(8192)
    }}});
    let comparison = compared(&before, &json!({}), &after, &json!({}));
    assert_eq!(comparison.refusal(), None);
    assert_eq!(comparison.removed, ["venice/gone"]);
    assert_eq!(comparison.added, [("venice/new".to_owned(), false)]);
    assert!(comparison.gains.is_empty());
    assert_eq!(
        comparison.limits,
        [
            (
                "venice/thinker".to_owned(),
                vec![
                    "output 64000 -> 32000".to_owned(),
                    "**reasoning budget 1024..64000 -> 1024..32000, narrower**".to_owned()
                ]
            ),
            (
                "venice/wider".to_owned(),
                vec!["reasoning budget 1024..4096 -> 1024..8192".to_owned()]
            )
        ]
    );
    assert_eq!(
        comparison.counted.into_iter().collect::<Vec<_>>(),
        [("prices".to_owned(), 1)]
    );
}

/// Each value a review row sets on a model models.dev lists, a `from`
/// row's own included, is listed as agreeing with models.dev or not.
#[test]
fn reviewed_values_are_compared_with_models_dev() {
    let models_dev = json!({"venice": {"models": {
        "base": {"name": "Base", "tool_call": false, "limit": {"context": 8192, "output": 1024}},
        "alias": {"name": "Alias", "tool_call": false}
    }}});
    let review = json!({"venice": {"models": {
        "base": {"source": "https://example.com/m", "limit": {"output": 1024}, "tool_call": true},
        "alias": {"from": "venice/base", "source": "https://example.com/m", "tool_call": true}
    }}});
    let comparison = compared(&models_dev, &review, &models_dev, &review);
    assert_eq!(comparison.agrees, ["`venice/base` `limit.output`"]);
    assert_eq!(
        comparison.disagrees,
        [
            "`venice/alias` `tool_call`: review true, models.dev false",
            "`venice/base` `tool_call`: review true, models.dev false"
        ]
    );
}

/// A review row needs a source when it sets a capability or a `rig` fact,
/// or copies a row with `from`, unless the unsourced list holds it as it
/// was; a `from` row takes its origin review's. A listed row that changes, or gains a source, leaves the list.
#[test]
fn a_new_or_changed_review_row_needs_a_source() {
    let base = json!({"venice": {"models": {
        "old": {"tool_call": true},
        "sourced": {"source": "https://example.com/m", "reasoning": true}
    }}});
    let review = |extra: serde_json::Value| -> Rows {
        let mut review = base.clone();
        if let (Some(models), Some(extra)) = (
            review
                .pointer_mut("/venice/models")
                .and_then(|models| models.as_object_mut()),
            extra.as_object(),
        ) {
            models.extend(extra.clone());
        }
        read_rows(&review).expect("review")
    };
    let unsourced = format!(
        "# rows without a source\n{} venice/old\n",
        guard::fingerprint(&review(json!({}))["venice"]["old"])
    );
    for (extra, refused) in [
        (json!({}), None),
        (json!({"named": {"name": "Named"}}), None),
        (
            json!({"copy": {"from": "venice/sourced", "rig": {"cache": ["short"]}}}),
            None,
        ),
        (json!({"copy": {"from": "venice/sourced"}}), None),
        (
            json!({"copy": {"from": "venice/unreviewed"}}),
            Some("venice/copy: sets from"),
        ),
        (
            json!({"new": {"source": "unverified: copied from venice/old", "tool_call": true}}),
            None,
        ),
        (
            json!({"new": {"rig": {"cache": ["short"]}}}),
            Some("venice/new: sets rig"),
        ),
        (
            json!({"old": {"tool_call": false}}),
            Some("venice/old: changed"),
        ),
        (
            json!({"old": {"source": "https://example.com/old", "tool_call": true}}),
            Some("venice/old: has a source now"),
        ),
    ] {
        let result = guard::provenance(&review(extra), &unsourced);
        match refused {
            None => assert_eq!(result, Ok(())),
            Some(row) => assert!(result.is_err_and(|error| error.contains(row))),
        }
    }
}

/// What rig-core reads from `spec`, in the guard's terms.
fn rig_core_reading(spec: &rig_core::catalog::ModelSpec) -> Option<Reading> {
    use rig_core::catalog::ReasoningSupport;
    let efforts = ["minimal", "low", "medium", "high", "xhigh", "max"];
    let reasoning = match &spec.reasoning {
        ReasoningSupport::None => Reasoning::None,
        ReasoningSupport::Unknown { .. } => Reasoning::Unknown,
        ReasoningSupport::Listed {
            levels,
            budget,
            can_disable,
            ..
        } => Reasoning::Listed {
            levels: efforts
                .into_iter()
                .filter(|word| levels.iter().any(|level| level.as_str() == *word))
                .collect(),
            off: *can_disable,
            budget: budget.as_ref().map(|range| (*range.start(), *range.end())),
        },
        _ => return None,
    };
    let input = spec.input;
    Some(Reading {
        reasoning,
        input: [
            (input.text, "text"),
            (input.image, "image"),
            (input.audio, "audio"),
            (input.video, "video"),
            (input.pdf, "pdf"),
        ]
        .into_iter()
        .filter_map(|(listed, word)| listed.then_some(word))
        .collect(),
        tools: spec.tools,
        structured_output: spec.structured_output,
        sampling: spec.sampling.and_then(|sampling| {
            serde_json::to_value(sampling)
                .ok()
                .and_then(|word| word.as_str().map(str::to_owned))
        }),
    })
}

/// The guard reads every row of the checked-in catalog as rig-core does,
/// and the rows that catalog lacks the shape of: reasoning options on a
/// model that does not reason, unknown and repeated words, several budgets,
/// zero bounds, an output limit rig-core cannot hold, no modalities.
#[test]
fn the_reading_is_rig_cores() {
    use rig_core::catalog::Catalog;
    let budget = |options: serde_json::Value, output: u64| json!({"reasoning": true, "reasoning_options": options, "limit": {"context": 8192, "output": output}});
    let shapes = json!({"venice": {"models": {
        "absent": {},
        "not-reasoning": {"reasoning": false, "reasoning_options": [{"type": "toggle"}]},
        "empty": {"reasoning": true, "reasoning_options": []},
        "no-control": {"reasoning": true, "reasoning_options": [], "rig": {"reasoning_control": "none"}},
        "words": {"reasoning": true, "reasoning_options": [
            {"type": "effort", "values": ["default", "none", "high", "high", "low"]},
            {"type": "adaptive"}
        ]},
        "budgets": budget(json!([{"type": "budget_tokens", "min": 1, "max": 10}, {"type": "budget_tokens"}]), 4096),
        "zero-bounds": budget(json!([{"type": "budget_tokens", "min": 0, "max": 0}]), 4096),
        "zero-output": budget(json!([{"type": "budget_tokens", "min": 1024}]), 0),
        "huge-output": budget(json!([{"type": "budget_tokens"}]), 5_000_000_000),
        "null-modalities": {"modalities": null, "tool_call": false},
        "other-words": {"modalities": {"input": ["file", "pdf", "text"]}, "structured_output": true},
        "sampling": {"temperature": true, "rig": {"sampling": "reasoning_off"}},
        "temperature": {"temperature": true, "tool_call": true}
    }}})
    .to_string();
    let checked_in = include_str!("../../../crates/rig-core/src/catalog/models.json");
    for text in [checked_in, shapes.as_str()] {
        let rows = read_rows(&serde_json::from_str(text).expect("JSON")).expect("rows");
        let (catalog, skipped) = Catalog::from_models_dev(text).expect("loads");
        assert!(skipped.is_empty(), "{skipped:?}");
        let mut differ = Vec::new();
        let mut read = 0;
        for spec in catalog.iter() {
            let (vendor, id) = (spec.provider.vendor(), &spec.id);
            let row = rows.get(vendor).and_then(|models| models.get(id));
            let guard = row.map(guard::reading);
            let rig_core = rig_core_reading(spec);
            if guard.is_none() || guard != rig_core {
                differ.push(format!(
                    "{vendor}/{id}: guard {guard:?}, rig-core {rig_core:?}"
                ));
            }
            read += 1;
        }
        assert_eq!(differ, Vec::<String>::new());
        assert_eq!(
            read,
            rows.values().map(|models| models.len()).sum::<usize>()
        );
    }
}
