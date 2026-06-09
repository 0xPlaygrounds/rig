#![allow(clippy::unwrap_used)]

use rig_core::vector_store::request::{Filter as CoreFilter, SearchFilter};
use rig_redis::filter::{Filter, RedisNumber, RedisValue};

#[test]
fn test_filter_eq_string() {
    let filter = Filter::eq("category", RedisValue::String("electronics".to_string())).unwrap();
    assert_eq!(filter.into_inner(), "@category:{electronics}");
}

#[test]
fn test_filter_eq_string_escapes_tag_value() {
    let filter = Filter::eq(
        "category",
        RedisValue::String("audio | video - {new}, @home".to_string()),
    )
    .unwrap();
    assert_eq!(
        filter.into_inner(),
        r"@category:{audio\ \|\ video\ \-\ \{new\}\,\ \@home}"
    );
}

#[test]
fn test_filter_escapes_field_name_segments() {
    let filter = Filter::eq(
        "metadata.category-name",
        RedisValue::String("electronics".to_string()),
    )
    .unwrap();
    assert_eq!(
        filter.into_inner(),
        r"@metadata.category\-name:{electronics}"
    );
}

#[test]
fn test_filter_eq_number() {
    let filter = Filter::eq("price", RedisValue::Number(99.99)).unwrap();
    // Numeric equality uses range syntax: @field:[val val]
    assert_eq!(filter.into_inner(), "@price:[99.99 99.99]");
}

#[test]
fn test_filter_eq_nan_errors() {
    let err = Filter::eq("price", RedisValue::Number(f64::NAN)).unwrap_err();
    assert!(err.to_string().contains("finite numeric value"));
}

#[test]
fn test_filter_gt() {
    let filter = Filter::gt("price", RedisValue::Number(50.0)).unwrap();
    assert_eq!(filter.into_inner(), "@price:[(50 +inf]");
}

#[test]
fn test_search_filter_trait_gt_is_numeric() {
    let filter = <Filter as SearchFilter>::gt("price", RedisNumber::new(50.0).unwrap());
    assert_eq!(filter.into_inner(), "@price:[(50 +inf]");
}

#[test]
fn test_redis_number_rejects_non_finite_values() {
    assert!(RedisNumber::new(f64::NAN).is_err());
    assert!(RedisNumber::new(f64::INFINITY).is_err());
    assert!(RedisNumber::new(f64::NEG_INFINITY).is_err());
}

#[test]
fn test_filter_lt() {
    let filter = Filter::lt("price", RedisValue::Number(100.0)).unwrap();
    assert_eq!(filter.into_inner(), "@price:[-inf (100]");
}

#[test]
fn test_filter_gte() {
    let filter = Filter::gte("price", RedisValue::Number(50.0)).unwrap();
    assert_eq!(filter.into_inner(), "@price:[50 +inf]");
}

#[test]
fn test_filter_lte() {
    let filter = Filter::lte("price", RedisValue::Number(100.0)).unwrap();
    assert_eq!(filter.into_inner(), "@price:[-inf 100]");
}

#[test]
fn test_filter_range() {
    let filter = Filter::range("price", 50.0, 100.0).unwrap();
    assert_eq!(filter.into_inner(), "@price:[50 100]");
}

#[test]
fn test_filter_range_rejects_non_finite_min() {
    let err = Filter::range("price", f64::NAN, 100.0).unwrap_err();
    assert!(err.to_string().contains("finite numeric value"));
}

#[test]
fn test_filter_range_exclusive() {
    let filter = Filter::range_exclusive("price", 50.0, 100.0).unwrap();
    assert_eq!(filter.into_inner(), "@price:[(50 (100]");
}

#[test]
fn test_filter_range_exclusive_rejects_non_finite_max() {
    let err = Filter::range_exclusive("price", 50.0, f64::INFINITY).unwrap_err();
    assert!(err.to_string().contains("finite numeric value"));
}

#[test]
fn test_filter_and() {
    let filter1 = Filter::eq("category", RedisValue::String("electronics".to_string())).unwrap();
    let filter2 = Filter::gt("price", RedisValue::Number(50.0)).unwrap();
    let combined = filter1.and(filter2);
    assert_eq!(
        combined.into_inner(),
        "(@category:{electronics} @price:[(50 +inf])"
    );
}

#[test]
fn test_filter_or() {
    let filter1 = Filter::eq("category", RedisValue::String("electronics".to_string())).unwrap();
    let filter2 = Filter::eq("category", RedisValue::String("books".to_string())).unwrap();
    let combined = filter1.or(filter2);
    assert_eq!(
        combined.into_inner(),
        "(@category:{electronics} | @category:{books})"
    );
}

#[test]
fn test_filter_not() {
    let filter = Filter::eq("category", RedisValue::String("electronics".to_string())).unwrap();
    let negated = filter.not();
    assert_eq!(negated.into_inner(), "-@category:{electronics}");
}

#[test]
fn test_filter_tag_in() {
    let filter = Filter::tag_in("tags", vec!["new".to_string(), "sale".to_string()]);
    assert_eq!(filter.into_inner(), "@tags:{new | sale}");
}

#[test]
fn test_filter_tag_in_escapes_values() {
    let filter = Filter::tag_in(
        "tags",
        vec![
            "new items".to_string(),
            "sale|clearance".to_string(),
            "a,b.c".to_string(),
        ],
    );
    assert_eq!(
        filter.into_inner(),
        r"@tags:{new\ items | sale\|clearance | a\,b\.c}"
    );
}

#[test]
fn test_filter_text_contains() {
    let filter = Filter::text_contains("description", "laptop");
    assert_eq!(filter.into_inner(), "@description:(laptop)");
}

#[test]
fn test_filter_text_contains_groups_multi_word_query() {
    let filter = Filter::text_contains("description", "gaming laptop");
    assert_eq!(filter.into_inner(), "@description:(gaming laptop)");
}

#[test]
fn test_filter_text_contains_escapes_query_syntax_inside_group() {
    let filter = Filter::text_contains("description", "laptop - @home | office");
    assert_eq!(
        filter.into_inner(),
        r"@description:(laptop \- \@home \| office)"
    );
}

#[test]
fn test_filter_text_contains_escapes_parentheses_and_quotes_inside_group() {
    let filter = Filter::text_contains("description", r#"laptop ("portable")"#);
    assert_eq!(
        filter.into_inner(),
        r#"@description:(laptop \(\"portable\"\))"#
    );
}

#[test]
fn test_complex_filter() {
    let category_filter =
        Filter::eq("category", RedisValue::String("electronics".to_string())).unwrap();
    let price_min = Filter::gte("price", RedisValue::Number(50.0)).unwrap();
    let price_max = Filter::lte("price", RedisValue::Number(200.0)).unwrap();

    let combined = category_filter.and(price_min).and(price_max);

    assert_eq!(
        combined.into_inner(),
        "((@category:{electronics} @price:[50 +inf]) @price:[-inf 200])"
    );
}

#[test]
fn test_core_filter_conversion() {
    let core_filter: CoreFilter<serde_json::Value> =
        CoreFilter::eq("category", serde_json::json!("electronics"));

    let redis_filter = Filter::try_from(core_filter).unwrap();
    assert_eq!(redis_filter.into_inner(), "@category:{electronics}");
}

#[test]
fn test_core_filter_gt_conversion() {
    let core_filter: CoreFilter<serde_json::Value> =
        CoreFilter::gt("price", serde_json::json!(50.0));

    let redis_filter = Filter::try_from(core_filter).unwrap();
    assert_eq!(redis_filter.into_inner(), "@price:[(50 +inf]");
}

#[test]
fn test_core_filter_gt_rejects_non_numeric_value() {
    let core_filter: CoreFilter<serde_json::Value> =
        CoreFilter::gt("status", serde_json::json!("active"));

    assert!(Filter::try_from(core_filter).is_err());
}

#[test]
fn test_core_filter_lt_rejects_non_numeric_value() {
    let core_filter: CoreFilter<serde_json::Value> =
        CoreFilter::lt("status", serde_json::json!("inactive"));

    assert!(Filter::try_from(core_filter).is_err());
}

#[test]
fn test_core_filter_eq_numeric_conversion() {
    let core_filter: CoreFilter<serde_json::Value> =
        CoreFilter::eq("price", serde_json::json!(99.99));

    let redis_filter = Filter::try_from(core_filter).unwrap();
    assert_eq!(redis_filter.into_inner(), "@price:[99.99 99.99]");
}

#[test]
fn test_core_filter_and_conversion() {
    let filter1: CoreFilter<serde_json::Value> =
        CoreFilter::eq("category", serde_json::json!("electronics"));
    let filter2: CoreFilter<serde_json::Value> = CoreFilter::gt("price", serde_json::json!(50.0));
    let combined = CoreFilter::and(filter1, filter2);

    let redis_filter = Filter::try_from(combined).unwrap();
    assert_eq!(
        redis_filter.into_inner(),
        "(@category:{electronics} @price:[(50 +inf])"
    );
}

#[test]
fn test_redis_value_bool() {
    let filter = Filter::eq("in_stock", RedisValue::Bool(true)).unwrap();
    assert_eq!(filter.into_inner(), "@in_stock:{1}");
}

#[test]
fn test_redis_value_from_str_ref() {
    let value: RedisValue = "hello".into();
    assert_eq!(value, RedisValue::String("hello".to_string()));
}

#[test]
fn test_filter_tag_in_single_value() {
    let filter = Filter::tag_in("tags", vec!["only".to_string()]);
    assert_eq!(filter.into_inner(), "@tags:{only}");
}

#[test]
fn test_filter_gt_non_numeric_errors() {
    let err = Filter::gt("status", RedisValue::String("active".to_string())).unwrap_err();
    assert!(err.to_string().contains("greater-than filter"));
}

#[test]
fn test_filter_gt_nan_errors() {
    let err = Filter::gt("price", RedisValue::Number(f64::NAN)).unwrap_err();
    assert!(err.to_string().contains("finite numeric value"));
}

#[test]
fn test_filter_gte_infinity_errors() {
    let err = Filter::gte("price", RedisValue::Number(f64::INFINITY)).unwrap_err();
    assert!(err.to_string().contains("finite numeric value"));
}

#[test]
fn test_filter_lte_negative_infinity_errors() {
    let err = Filter::lte("price", RedisValue::Number(f64::NEG_INFINITY)).unwrap_err();
    assert!(err.to_string().contains("finite numeric value"));
}

#[test]
fn test_filter_lt_non_numeric_errors() {
    let err = Filter::lt("status", RedisValue::String("inactive".to_string())).unwrap_err();
    assert!(err.to_string().contains("less-than filter"));
}

#[test]
fn test_filter_gte_bool_errors() {
    let err = Filter::gte("enabled", RedisValue::Bool(true)).unwrap_err();
    assert!(err.to_string().contains("greater-than-or-equal filter"));
}

#[test]
fn test_filter_lte_bool_errors() {
    let err = Filter::lte("enabled", RedisValue::Bool(false)).unwrap_err();
    assert!(err.to_string().contains("less-than-or-equal filter"));
}
