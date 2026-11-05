use super::*;

#[test]
fn nonnumeric_range_bounds_are_errors() -> anyhow::Result<()> {
    for name in ["gt", "gte", "lt", "lte"] {
        for value in [json!("50"), json!(null), json!(true), json!([]), json!({})] {
            let filter = QdrantFilter(json!({"key": "price", "range": {name: value}}));
            let result = filter.interpret();
            anyhow::ensure!(
                matches!(result, Err(FilterError::Expected { .. })),
                "{name} accepted a nonnumeric bound: {result:?}"
            );
        }
    }
    Ok(())
}

#[test]
fn invalid_bound_is_not_dropped_from_a_partial_or_nested_range() {
    let partial = QdrantFilter::range_inclusive("price", &json!(10), &json!("50"));
    assert!(partial.clone().interpret().is_err());
    assert!(
        QdrantFilter::eq("active", json!(true))
            .and(partial.clone())
            .interpret()
            .is_err()
    );
    assert!(
        QdrantFilter::eq("active", json!(true))
            .or(partial.clone())
            .interpret()
            .is_err()
    );
    assert!(partial.not().interpret().is_err());
    assert!(QdrantFilter::gt("price", json!("50")).interpret().is_err());
    assert!(QdrantFilter::lt("price", json!("50")).interpret().is_err());
}

#[test]
fn numeric_ranges_keep_present_bounds_and_leave_absent_bounds_unset() -> anyhow::Result<()> {
    for (range, expected) in [
        (
            json!({"gt": -3, "lte": 50.5}),
            Range {
                gt: Some(-3.0),
                lte: Some(50.5),
                ..Default::default()
            },
        ),
        (
            json!({"gte": 0, "lt": 100}),
            Range {
                gte: Some(0.0),
                lt: Some(100.0),
                ..Default::default()
            },
        ),
    ] {
        let filter = QdrantFilter(json!({"key": "price", "range": range})).interpret()?;
        let actual = filter
            .as_ref()
            .and_then(|filter| filter.must.first())
            .and_then(|condition| condition.condition_one_of.as_ref());
        let Some(ConditionOneOf::Field(field)) = actual else {
            anyhow::bail!("missing field condition: {actual:?}");
        };
        anyhow::ensure!(
            field.range.as_ref() == Some(&expected),
            "range changed: {field:?}"
        );
    }
    Ok(())
}
