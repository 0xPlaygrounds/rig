use serde_json::json;

use super::{generate, read_rows, render};

fn models_dev() -> serde_json::Value {
    json!({
        "anthropic": {"name": "Anthropic", "models": {
            "claude-x-1": {
                "id": "claude-x-1", "name": "Claude X 1", "family": "claude",
                "reasoning": true,
                "reasoning_options": [{"type": "effort", "values": ["low", "high"], "extra": 1}],
                "modalities": {"input": ["text", "image"], "output": ["text"]},
                "limit": {"context": 200000, "output": 64000},
                "cost": {"input": 3, "output": 15, "tiers": []}
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
fn sync_keeps_rigs_providers_and_the_fields_the_catalog_reads() {
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
    let claude = &rows["anthropic"]["claude-x-1"];
    assert_eq!(claude.get("family"), None);
    assert_eq!(claude["modalities"], json!({"input": ["text", "image"]}));
    assert_eq!(claude["cost"], json!({"input": 3, "output": 15}));
    assert_eq!(
        claude["reasoning_options"],
        json!([{"type": "effort", "values": ["low", "high"]}])
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
        json!({"context": 200000, "output": 128000})
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
    assert_eq!(bedrock["rig"], claude["rig"]);

    let openrouter = &rows["openrouter"]["anthropic/claude-x.1"];
    assert_eq!(openrouter["rig"], claude["rig"]);
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
        vertex["rig"],
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
fn a_reviewed_key_the_catalog_does_not_read_is_an_error() {
    for review in [
        json!({"anthropic": {"models": {"claude-x-1": {"limits": {}}}}}),
        json!({"anthropic": {"models": {"claude-x-1": {"rig": {"adaptive": true}}}}}),
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
    assert_eq!(claude["rig"], json!({"reasoning_control": "none"}));
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
