use super::UsageRecord;

#[test]
fn a_logged_usage_record_restores_the_whole_totals() -> Result<(), serde_json::Error> {
    // A usage record as a session logged it, summed over two models.
    let logged = r#"{
        "models": {
            "deepseek/deepseek-flash": {
                "tokens": {"input_tokens": 1285279, "output_tokens": 20201,
                           "total_tokens": 1305480, "cached_input_tokens": 1253632},
                "cost": 0.02, "unpriced": 0, "calls": 48, "context": 47924
            },
            "local/model": {
                "tokens": {"input_tokens": 100, "output_tokens": 10},
                "cost": 0.0, "unpriced": 2, "calls": 2, "context": 110
            }
        },
        "context": 47924
    }"#;
    let usage: UsageRecord = serde_json::from_str(logged)?;
    let total = usage.total();
    assert_eq!(total.calls, 50);
    assert_eq!(total.unpriced, 2);
    assert_eq!(total.context, Some(47924));
    assert_eq!(total.tokens.output_tokens, Some(20211));
    assert_eq!(total.cost_label().as_deref(), Some("$0.020+"));
    Ok(())
}
