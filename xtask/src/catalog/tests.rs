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
            "us.anthropic.claude-x-1-v1:0": {"name": "Claude X 1", "reasoning": true},
            "amazon.nova-pro-v1:0": {"name": "Nova Pro"}
        }},
        "somebody-else": {"models": {"m": {"name": "M"}}}
    })
}

#[test]
fn sync_keeps_rigs_providers_and_the_fields_the_catalog_reads() {
    let rows = generate(&models_dev(), &json!({})).expect("generates");
    assert_eq!(
        rows.keys().collect::<Vec<_>>(),
        ["anthropic", "aws_bedrock"],
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
fn bedrock_rows_of_an_anthropic_model_take_its_reasoning_and_facts() {
    let review = json!({
        "anthropic": {"models": {"claude-x-1": {"rig": {"binds_context": true}}}},
        "aws_bedrock": {"models": {"us.anthropic.claude-x-1-v1:0": {"rig": {"binds_context": false}}}}
    });
    let rows = generate(&models_dev(), &review).expect("generates");
    let bedrock = &rows["aws_bedrock"]["us.anthropic.claude-x-1-v1:0"];
    assert_eq!(
        bedrock["reasoning_options"],
        rows["anthropic"]["claude-x-1"]["reasoning_options"]
    );
    assert_eq!(
        bedrock["rig"],
        json!({"binds_context": false}),
        "a reviewed Bedrock fact wins over the Anthropic one"
    );
    assert_eq!(rows["aws_bedrock"]["amazon.nova-pro-v1:0"].get("rig"), None);
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
        9,
        "one line per model:\n{rendered}"
    );
}
