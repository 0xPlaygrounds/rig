use super::*;

/// A range the filter cannot send must fail the search, not widen it. Numeric
/// bounds still reach their own range fields.
#[test]
fn unusable_range_bounds_are_refused() {
    let mut ranges = vec![json!({"gte": 10, "lte": "50"})];
    for name in ["gt", "gte", "lt", "lte"] {
        for value in [json!("50"), json!(null), json!(true), json!([]), json!({})] {
            ranges.push(json!({name: value}));
        }
    }

    for range in ranges {
        let leaf = QdrantFilter(json!({"key": "price", "range": range}));
        let active = QdrantFilter::eq("active", json!(true));
        for filter in [
            leaf.clone(),
            active.clone().and(leaf.clone()),
            active.or(leaf.clone()),
            leaf.not(),
        ] {
            let shown = filter.0.clone();
            let result = filter.interpret();
            assert!(
                matches!(result, Err(FilterError::Expected { .. })),
                "range constraint dropped from {shown}: {result:?}"
            );
        }
    }

    let filter = QdrantFilter(json!({
        "must": [
            {"key": "a", "range": {"gt": -3, "lte": 50.5}},
            {"key": "b", "range": {"gte": 0, "lt": 100}},
        ]
    }));
    let ranges: Vec<_> = filter
        .interpret()
        .ok()
        .flatten()
        .map(|filter| filter.must)
        .unwrap_or_default()
        .into_iter()
        .filter_map(|condition| match condition.condition_one_of {
            Some(ConditionOneOf::Field(field)) => field.range,
            _ => None,
        })
        .collect();

    assert_eq!(
        ranges,
        [
            Range {
                gt: Some(-3.0),
                lte: Some(50.5),
                ..Default::default()
            },
            Range {
                gte: Some(0.0),
                lt: Some(100.0),
                ..Default::default()
            },
        ]
    );
}
