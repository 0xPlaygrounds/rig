use super::*;
use rig_core::vector_store::request::SearchFilter;
use serde_json::json;

#[test]
fn test_eq_filter() {
    let filter = VectorizeFilter::eq("category", json!("programming"));
    assert_eq!(
        filter.into_inner(),
        json!({ "category": { "$eq": "programming" } })
    );
}

#[test]
fn test_gt_filter() {
    let filter = VectorizeFilter::gt("score", json!(0.5));
    assert_eq!(filter.into_inner(), json!({ "score": { "$gt": 0.5 } }));
}

#[test]
fn test_lt_filter() {
    let filter = VectorizeFilter::lt("price", json!(100));
    assert_eq!(filter.into_inner(), json!({ "price": { "$lt": 100 } }));
}

#[test]
fn test_ne_filter() {
    let filter = VectorizeFilter::ne("status", &json!("deleted"));
    assert_eq!(
        filter.into_inner(),
        json!({ "status": { "$ne": "deleted" } })
    );
}

#[test]
fn test_gte_filter() {
    let filter = VectorizeFilter::gte("count", &json!(10));
    assert_eq!(filter.into_inner(), json!({ "count": { "$gte": 10 } }));
}

#[test]
fn test_lte_filter() {
    let filter = VectorizeFilter::lte("age", &json!(65));
    assert_eq!(filter.into_inner(), json!({ "age": { "$lte": 65 } }));
}

#[test]
fn test_in_filter() {
    let filter = VectorizeFilter::in_values("category", &[json!("a"), json!("b"), json!("c")]);
    assert_eq!(
        filter.into_inner(),
        json!({ "category": { "$in": ["a", "b", "c"] } })
    );
}

#[test]
fn test_nin_filter() {
    let filter = VectorizeFilter::nin("status", &[json!("deleted"), json!("archived")]);
    assert_eq!(
        filter.into_inner(),
        json!({ "status": { "$nin": ["deleted", "archived"] } })
    );
}

#[test]
fn test_and_filter() {
    let filter1 = VectorizeFilter::eq("category", json!("programming"));
    let filter2 = VectorizeFilter::gt("score", json!(0.5));
    let combined = filter1.and(filter2);

    let result = combined.into_inner();
    let Some(obj) = result.as_object() else {
        assert!(
            result.is_object(),
            "combined filter should serialize to an object"
        );
        return;
    };

    // Both keys should be present (implicit AND)
    assert!(obj.contains_key("category"));
    assert!(obj.contains_key("score"));
    assert_eq!(obj.get("category"), Some(&json!({ "$eq": "programming" })));
    assert_eq!(obj.get("score"), Some(&json!({ "$gt": 0.5 })));
}

#[test]
fn test_or_filter_validation_fails() {
    let filter1 = VectorizeFilter::eq("a", json!(1));
    let filter2 = VectorizeFilter::eq("b", json!(2));
    let combined = filter1.or(filter2);

    // OR should create an invalid filter
    let result = combined.validate();
    assert!(
        matches!(
            &result,
            Err(VectorizeError::UnsupportedFilterOperation(msg)) if msg.contains("OR")
        ),
        "expected UnsupportedFilterOperation error mentioning OR, got {result:?}"
    );
}

#[test]
fn test_empty_filter() {
    let filter = VectorizeFilter::new();
    assert!(filter.is_empty());
    assert_eq!(filter.into_inner(), json!({}));
}

#[test]
fn test_non_empty_filter() {
    let filter = VectorizeFilter::eq("key", json!("value"));
    assert!(!filter.is_empty());
}

#[test]
fn test_multiple_and_filters() {
    let filter = VectorizeFilter::eq("category", json!("tech"))
        .and(VectorizeFilter::gt("score", json!(0.5)))
        .and(VectorizeFilter::lt("price", json!(100)));

    let result = filter.into_inner();
    let Some(obj) = result.as_object() else {
        assert!(
            result.is_object(),
            "combined filter should serialize to an object"
        );
        return;
    };

    assert_eq!(obj.len(), 3);
    assert!(obj.contains_key("category"));
    assert!(obj.contains_key("score"));
    assert!(obj.contains_key("price"));
}
