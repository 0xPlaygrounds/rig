use super::{Cost, Usage};
use anyhow::{Context, ensure};
use serde_json::json;

/// These cases exercise normalized usage's Serde contract without provider I/O.
#[test]
fn numeric_reported_cost_normalizes_totals_without_inventing_parts() -> anyhow::Result<()> {
    for total in [json!(0.00001386), json!(0), json!(2), json!(-0.5)] {
        let reported_total = total.as_f64().context("numeric total")?;
        let usage: Usage = serde_json::from_value(json!({
            "input_tokens": 330,
            "output_tokens": 21,
            "cost": total,
        }))?;
        let cost = usage.cost.context("reported total")?;
        ensure!(cost == Cost::from_total(reported_total));
        ensure!(cost.is_complete());
        ensure!(usage.input_tokens == Some(330) && usage.output_tokens == Some(21));

        let serialized = serde_json::to_value(usage)?;
        ensure!(serialized.get("cost") == Some(&json!({"total": reported_total})));
        let restored: Usage = serde_json::from_value(serialized)?;
        ensure!(restored == usage);
    }
    Ok(())
}

/// Structured costs must retain every part and their partial-pricing flag.
#[test]
fn numeric_reported_cost_support_keeps_structured_costs_unchanged() -> anyhow::Result<()> {
    let value = json!({
        "input": 0.2,
        "output": 0.3,
        "cache_read": 0.1,
        "cache_write": 0.4,
        "total": 1.0,
        "incomplete": true,
    });
    let usage: Usage = serde_json::from_value(json!({"cost": value}))?;
    let cost = usage.cost.context("structured cost")?;
    ensure!(cost.input == Some(0.2) && cost.output == Some(0.3));
    ensure!(cost.cache_read == Some(0.1) && cost.cache_write == Some(0.4));
    ensure!(cost.total == 1.0 && !cost.is_complete());
    ensure!(serde_json::to_value(usage)?.get("cost") == Some(&value));

    let total_only: Usage = serde_json::from_value(json!({"cost": {"total": 1.0}}))?;
    ensure!(total_only.cost == Some(Cost::from_total(1.0)));
    Ok(())
}

/// Optional accounting stays absent when the provider supplies no cost.
#[test]
fn numeric_reported_cost_support_preserves_missing_and_null() -> anyhow::Result<()> {
    for value in [json!({}), json!({"cost": null})] {
        let usage: Usage = serde_json::from_value(value)?;
        ensure!(usage.cost.is_none());
        ensure!(serde_json::to_value(usage)?.get("cost").is_none());
    }
    Ok(())
}

/// Accepting scalar totals must not turn malformed accounting into a known cost.
#[test]
fn numeric_reported_cost_support_rejects_malformed_values() {
    for cost in [
        json!("0.5"),
        json!(true),
        json!([]),
        json!({}),
        json!({"input": 0.5}),
        json!({"total": "0.5"}),
    ] {
        assert!(serde_json::from_value::<Usage>(json!({"cost": cost})).is_err());
    }
}
