use super::*;
use serde_json::json;

#[test]
fn test_data_collection_serialization() {
    assert_eq!(
        serde_json::to_string(&DataCollection::Allow).unwrap(),
        r#""allow""#
    );
    assert_eq!(
        serde_json::to_string(&DataCollection::Deny).unwrap(),
        r#""deny""#
    );
}

#[test]
fn test_data_collection_default() {
    assert_eq!(DataCollection::default(), DataCollection::Allow);
}

#[test]
fn test_quantization_serialization() {
    assert_eq!(
        serde_json::to_string(&Quantization::Int4).unwrap(),
        r#""int4""#
    );
    assert_eq!(
        serde_json::to_string(&Quantization::Int8).unwrap(),
        r#""int8""#
    );
    assert_eq!(
        serde_json::to_string(&Quantization::Fp16).unwrap(),
        r#""fp16""#
    );
    assert_eq!(
        serde_json::to_string(&Quantization::Bf16).unwrap(),
        r#""bf16""#
    );
    assert_eq!(
        serde_json::to_string(&Quantization::Fp32).unwrap(),
        r#""fp32""#
    );
    assert_eq!(
        serde_json::to_string(&Quantization::Fp8).unwrap(),
        r#""fp8""#
    );
    assert_eq!(
        serde_json::to_string(&Quantization::Unknown).unwrap(),
        r#""unknown""#
    );
}

#[test]
fn test_provider_sort_strategy_serialization() {
    assert_eq!(
        serde_json::to_string(&ProviderSortStrategy::Price).unwrap(),
        r#""price""#
    );
    assert_eq!(
        serde_json::to_string(&ProviderSortStrategy::Throughput).unwrap(),
        r#""throughput""#
    );
    assert_eq!(
        serde_json::to_string(&ProviderSortStrategy::Latency).unwrap(),
        r#""latency""#
    );
}

#[test]
fn test_sort_partition_serialization() {
    assert_eq!(
        serde_json::to_string(&SortPartition::Model).unwrap(),
        r#""model""#
    );
    assert_eq!(
        serde_json::to_string(&SortPartition::None).unwrap(),
        r#""none""#
    );
}

#[test]
fn test_provider_sort_simple() {
    let sort = ProviderSort::Simple(ProviderSortStrategy::Latency);
    let json = serde_json::to_value(&sort).unwrap();
    assert_eq!(json, "latency");
}

#[test]
fn test_provider_sort_complex() {
    let sort = ProviderSort::Complex(
        ProviderSortConfig::new(ProviderSortStrategy::Price).partition(SortPartition::None),
    );
    let json = serde_json::to_value(&sort).unwrap();
    assert_eq!(json["by"], "price");
    assert_eq!(json["partition"], "none");
}

#[test]
fn test_provider_sort_complex_without_partition() {
    let sort = ProviderSort::Complex(ProviderSortConfig::new(ProviderSortStrategy::Throughput));
    let json = serde_json::to_value(&sort).unwrap();
    assert_eq!(json["by"], "throughput");
    assert!(json.get("partition").is_none());
}

#[test]
fn test_provider_sort_from_strategy() {
    let sort: ProviderSort = ProviderSortStrategy::Price.into();
    assert_eq!(sort, ProviderSort::Simple(ProviderSortStrategy::Price));
}

#[test]
fn test_provider_sort_from_config() {
    let config = ProviderSortConfig::new(ProviderSortStrategy::Latency);
    let sort: ProviderSort = config.into();
    match sort {
        ProviderSort::Complex(c) => assert_eq!(c.by, ProviderSortStrategy::Latency),
        _ => panic!("Expected Complex variant"),
    }
}

#[test]
fn test_percentile_thresholds_builder() {
    let thresholds = PercentileThresholds::new()
        .p50(10.0)
        .p75(25.0)
        .p90(50.0)
        .p99(100.0);

    assert_eq!(thresholds.p50, Some(10.0));
    assert_eq!(thresholds.p75, Some(25.0));
    assert_eq!(thresholds.p90, Some(50.0));
    assert_eq!(thresholds.p99, Some(100.0));
}

#[test]
fn test_percentile_thresholds_default() {
    let thresholds = PercentileThresholds::default();
    assert_eq!(thresholds.p50, None);
    assert_eq!(thresholds.p75, None);
    assert_eq!(thresholds.p90, None);
    assert_eq!(thresholds.p99, None);
}

#[test]
fn test_throughput_threshold_simple() {
    let threshold = ThroughputThreshold::Simple(50.0);
    let json = serde_json::to_value(&threshold).unwrap();
    assert_eq!(json, 50.0);
}

#[test]
fn test_throughput_threshold_percentile() {
    let threshold = ThroughputThreshold::Percentile(PercentileThresholds::new().p90(50.0));
    let json = serde_json::to_value(&threshold).unwrap();
    assert_eq!(json["p90"], 50.0);
}

#[test]
fn test_latency_threshold_simple() {
    let threshold = LatencyThreshold::Simple(0.5);
    let json = serde_json::to_value(&threshold).unwrap();
    assert_eq!(json, 0.5);
}

#[test]
fn test_latency_threshold_percentile() {
    let threshold = LatencyThreshold::Percentile(PercentileThresholds::new().p50(0.1).p99(1.0));
    let json = serde_json::to_value(&threshold).unwrap();
    assert_eq!(json["p50"], 0.1);
    assert_eq!(json["p99"], 1.0);
}

#[test]
fn test_max_price_builder() {
    let price = MaxPrice::new().prompt(0.001).completion(0.002);

    assert_eq!(price.prompt, Some(0.001));
    assert_eq!(price.completion, Some(0.002));
    assert_eq!(price.request, None);
    assert_eq!(price.image, None);
}

#[test]
fn test_max_price_all_fields() {
    let price = MaxPrice::new()
        .prompt(0.001)
        .completion(0.002)
        .request(0.01)
        .image(0.05);

    let json = serde_json::to_value(&price).unwrap();
    assert_eq!(json["prompt"], 0.001);
    assert_eq!(json["completion"], 0.002);
    assert_eq!(json["request"], 0.01);
    assert_eq!(json["image"], 0.05);
}

#[test]
fn test_max_price_default() {
    let price = MaxPrice::default();
    assert_eq!(price.prompt, None);
    assert_eq!(price.completion, None);
    assert_eq!(price.request, None);
    assert_eq!(price.image, None);
}

#[test]
fn test_provider_preferences_default() {
    let prefs = ProviderPreferences::default();
    assert!(prefs.order.is_none());
    assert!(prefs.only.is_none());
    assert!(prefs.ignore.is_none());
    assert!(prefs.allow_fallbacks.is_none());
    assert!(prefs.require_parameters.is_none());
    assert!(prefs.data_collection.is_none());
    assert!(prefs.zdr.is_none());
    assert!(prefs.sort.is_none());
    assert!(prefs.preferred_min_throughput.is_none());
    assert!(prefs.preferred_max_latency.is_none());
    assert!(prefs.max_price.is_none());
    assert!(prefs.quantizations.is_none());
}

#[test]
fn test_provider_preferences_order_with_fallbacks() {
    let prefs = ProviderPreferences::new()
        .order(["anthropic", "openai"])
        .allow_fallbacks(true);

    let json = prefs.to_json();
    let provider = &json["provider"];

    assert_eq!(provider["order"], json!(["anthropic", "openai"]));
    assert_eq!(provider["allow_fallbacks"], true);
}

#[test]
fn test_provider_preferences_only_allowlist() {
    let prefs = ProviderPreferences::new()
        .only(["azure", "together"])
        .allow_fallbacks(false);

    let json = prefs.to_json();
    let provider = &json["provider"];

    assert_eq!(provider["only"], json!(["azure", "together"]));
    assert_eq!(provider["allow_fallbacks"], false);
}

#[test]
fn test_provider_preferences_ignore() {
    let prefs = ProviderPreferences::new().ignore(["deepinfra"]);

    let json = prefs.to_json();
    let provider = &json["provider"];

    assert_eq!(provider["ignore"], json!(["deepinfra"]));
}

#[test]
fn test_provider_preferences_sort_latency() {
    let prefs = ProviderPreferences::new().sort(ProviderSortStrategy::Latency);

    let json = prefs.to_json();
    let provider = &json["provider"];

    assert_eq!(provider["sort"], "latency");
}

#[test]
fn test_provider_preferences_price_with_throughput() {
    let prefs = ProviderPreferences::new()
        .sort(ProviderSortStrategy::Price)
        .preferred_min_throughput(ThroughputThreshold::Percentile(
            PercentileThresholds::new().p90(50.0),
        ));

    let json = prefs.to_json();
    let provider = &json["provider"];

    assert_eq!(provider["sort"], "price");
    assert_eq!(provider["preferred_min_throughput"]["p90"], 50.0);
}

#[test]
fn test_provider_preferences_require_parameters() {
    let prefs = ProviderPreferences::new().require_parameters(true);

    let json = prefs.to_json();
    let provider = &json["provider"];

    assert_eq!(provider["require_parameters"], true);
}

#[test]
fn test_provider_preferences_data_policy_and_zdr() {
    let prefs = ProviderPreferences::new()
        .data_collection(DataCollection::Deny)
        .zdr(true);

    let json = prefs.to_json();
    let provider = &json["provider"];

    assert_eq!(provider["data_collection"], "deny");
    assert_eq!(provider["zdr"], true);
}

#[test]
fn test_provider_preferences_quantizations() {
    let prefs = ProviderPreferences::new().quantizations([Quantization::Int8, Quantization::Fp16]);

    let json = prefs.to_json();
    let provider = &json["provider"];

    assert_eq!(provider["quantizations"], json!(["int8", "fp16"]));
}

#[test]
fn test_provider_preferences_serialization_skips_none() {
    let prefs = ProviderPreferences::new().sort(ProviderSortStrategy::Price);

    let json = serde_json::to_value(&prefs).unwrap();

    assert_eq!(json["sort"], "price");
    assert!(json.get("order").is_none());
    assert!(json.get("only").is_none());
    assert!(json.get("ignore").is_none());
    assert!(json.get("zdr").is_none());
}

#[test]
fn test_provider_preferences_deserialization() {
    let json = json!({
        "order": ["anthropic", "openai"],
        "sort": "throughput",
        "data_collection": "deny",
        "zdr": true,
        "quantizations": ["int8", "fp16"]
    });

    let prefs: ProviderPreferences = serde_json::from_value(json).unwrap();

    assert_eq!(
        prefs.order,
        Some(vec!["anthropic".to_string(), "openai".to_string()])
    );
    assert_eq!(
        prefs.sort,
        Some(ProviderSort::Simple(ProviderSortStrategy::Throughput))
    );
    assert_eq!(prefs.data_collection, Some(DataCollection::Deny));
    assert_eq!(prefs.zdr, Some(true));
    assert_eq!(
        prefs.quantizations,
        Some(vec![Quantization::Int8, Quantization::Fp16])
    );
}

#[test]
fn test_provider_preferences_deserialization_complex_sort() {
    let json = json!({
        "sort": {
            "by": "latency",
            "partition": "model"
        }
    });

    let prefs: ProviderPreferences = serde_json::from_value(json).unwrap();

    match prefs.sort {
        Some(ProviderSort::Complex(config)) => {
            assert_eq!(config.by, ProviderSortStrategy::Latency);
            assert_eq!(config.partition, Some(SortPartition::Model));
        }
        _ => panic!("Expected Complex sort variant"),
    }
}

#[test]
fn test_provider_preferences_full_integration() {
    let prefs = ProviderPreferences::new()
        .order(["anthropic", "openai"])
        .only(["anthropic", "openai", "google"])
        .sort(ProviderSortStrategy::Throughput)
        .data_collection(DataCollection::Deny)
        .zdr(true)
        .quantizations([Quantization::Int8])
        .allow_fallbacks(false);

    let json = prefs.to_json();

    assert!(json.get("provider").is_some());
    let provider = &json["provider"];
    assert_eq!(provider["order"], json!(["anthropic", "openai"]));
    assert_eq!(provider["only"], json!(["anthropic", "openai", "google"]));
    assert_eq!(provider["sort"], "throughput");
    assert_eq!(provider["data_collection"], "deny");
    assert_eq!(provider["zdr"], true);
    assert_eq!(provider["quantizations"], json!(["int8"]));
    assert_eq!(provider["allow_fallbacks"], false);
}

#[test]
fn test_provider_preferences_max_price() {
    let prefs =
        ProviderPreferences::new().max_price(MaxPrice::new().prompt(0.001).completion(0.002));

    let json = prefs.to_json();
    let provider = &json["provider"];

    assert_eq!(provider["max_price"]["prompt"], 0.001);
    assert_eq!(provider["max_price"]["completion"], 0.002);
}

#[test]
fn test_provider_preferences_preferred_max_latency() {
    let prefs = ProviderPreferences::new().preferred_max_latency(LatencyThreshold::Simple(0.5));

    let json = prefs.to_json();
    let provider = &json["provider"];

    assert_eq!(provider["preferred_max_latency"], 0.5);
}

#[test]
fn test_provider_preferences_empty_arrays() {
    let prefs = ProviderPreferences::new()
        .order(Vec::<String>::new())
        .quantizations(Vec::<Quantization>::new());

    let json = prefs.to_json();
    let provider = &json["provider"];

    assert_eq!(provider["order"], json!([]));
    assert_eq!(provider["quantizations"], json!([]));
}
