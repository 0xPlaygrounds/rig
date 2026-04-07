use rig::vector_store::request::{Filter as CoreFilter, SearchFilter};
use rig_redis::filter::{Filter, RedisValue};

#[test]
fn test_filter_eq_string() {
    let filter = Filter::eq("category", RedisValue::String("electronics".to_string()));
    assert_eq!(filter.into_inner(), "@category:{electronics}");
}

#[test]
fn test_filter_eq_number() {
    let filter = Filter::eq("price", RedisValue::Number(99.99));
    assert_eq!(filter.into_inner(), "@price:99.99");
}

#[test]
fn test_filter_gt() {
    let filter = Filter::gt("price", RedisValue::Number(50.0));
    assert_eq!(filter.into_inner(), "@price:[(50 +inf]");
}

#[test]
fn test_filter_lt() {
    let filter = Filter::lt("price", RedisValue::Number(100.0));
    assert_eq!(filter.into_inner(), "@price:[-inf (100]");
}

#[test]
fn test_filter_gte() {
    let filter = Filter::gte("price", RedisValue::Number(50.0));
    assert_eq!(filter.into_inner(), "@price:[50 +inf]");
}

#[test]
fn test_filter_lte() {
    let filter = Filter::lte("price", RedisValue::Number(100.0));
    assert_eq!(filter.into_inner(), "@price:[-inf 100]");
}

#[test]
fn test_filter_range() {
    let filter = Filter::range("price", 50.0, 100.0);
    assert_eq!(filter.into_inner(), "@price:[50 100]");
}

#[test]
fn test_filter_range_exclusive() {
    let filter = Filter::range_exclusive("price", 50.0, 100.0);
    assert_eq!(filter.into_inner(), "@price:[(50 (100]");
}

#[test]
fn test_filter_and() {
    let filter1 = Filter::eq("category", RedisValue::String("electronics".to_string()));
    let filter2 = Filter::gt("price", RedisValue::Number(50.0));
    let combined = filter1.and(filter2);
    assert_eq!(
        combined.into_inner(),
        "(@category:{electronics} @price:[(50 +inf])"
    );
}

#[test]
fn test_filter_or() {
    let filter1 = Filter::eq("category", RedisValue::String("electronics".to_string()));
    let filter2 = Filter::eq("category", RedisValue::String("books".to_string()));
    let combined = filter1.or(filter2);
    assert_eq!(
        combined.into_inner(),
        "(@category:{electronics} | @category:{books})"
    );
}

#[test]
fn test_filter_not() {
    let filter = Filter::eq("category", RedisValue::String("electronics".to_string()));
    let negated = filter.not();
    assert_eq!(negated.into_inner(), "-@category:{electronics}");
}

#[test]
fn test_filter_tag_in() {
    let filter = Filter::tag_in("tags", vec!["new".to_string(), "sale".to_string()]);
    assert_eq!(filter.into_inner(), "@tags:{new | sale}");
}

#[test]
fn test_filter_text_contains() {
    let filter = Filter::text_contains("description", "laptop");
    assert_eq!(filter.into_inner(), "@description:laptop");
}

#[test]
fn test_complex_filter() {
    let category_filter = Filter::eq("category", RedisValue::String("electronics".to_string()));
    let price_min = Filter::gte("price", RedisValue::Number(50.0));
    let price_max = Filter::lte("price", RedisValue::Number(200.0));

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
    let filter = Filter::eq("in_stock", RedisValue::Bool(true));
    assert_eq!(filter.into_inner(), "@in_stock:1");
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
