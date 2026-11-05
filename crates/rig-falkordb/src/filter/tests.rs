use super::*;
use serde_json::json;

#[test]
fn nested_filters_bind_values_in_order() -> anyhow::Result<()> {
    let injected = "a\\' OR true //";
    let filter = FalkorDBSearchFilter::eq("display name", json!(injected)).and(
        FalkorDBSearchFilter::gt("rating", json!(3))
            .or(FalkorDBSearchFilter::lt("rating", json!(10))),
    );
    let (condition, params) = filter.compile()?;
    anyhow::ensure!(
        condition
            == "(node.`display name` = $filter_0) AND ((node.`rating` > $filter_1) OR (node.`rating` < $filter_2))"
    );
    anyhow::ensure!(!condition.contains(injected));
    anyhow::ensure!(
        params
            == vec![
                ("filter_0".into(), FalkorValue::String(injected.into())),
                ("filter_1".into(), FalkorValue::I64(3)),
                ("filter_2".into(), FalkorValue::I64(10)),
            ]
    );
    let round_trip: FalkorDBSearchFilter = serde_json::from_value(serde_json::to_value(filter)?)?;
    anyhow::ensure!(round_trip.compile()? == (condition, params));
    Ok(())
}

#[test]
fn null_equality_uses_is_null() -> anyhow::Result<()> {
    let (condition, params) = FalkorDBSearchFilter::eq("missing", Value::Null).compile()?;
    anyhow::ensure!(condition == "node.`missing` IS NULL");
    anyhow::ensure!(params.is_empty());
    Ok(())
}

#[test]
fn invalid_property_names_are_filter_errors() {
    for key in ["", "a` OR true //", "a\0b"] {
        assert!(matches!(
            FalkorDBSearchFilter::eq(key, json!(1)).compile(),
            Err(VectorStoreError::FilterError(_))
        ));
    }
}
