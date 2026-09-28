use super::*;

#[test]
fn snake_case_splits_lower_to_upper_boundaries() {
    assert_eq!(snake("thoughtSignature"), "thought_signature");
    assert_eq!(snake("_responseJsonSchema"), "response_json_schema");
    assert_eq!(snake("int64Value"), "int64_value");
}

#[test]
fn reserved_and_legacy_fields_get_distinct_idents() {
    let taken = BTreeSet::new();
    assert_eq!(field_ident("type", &taken), "r#type");
    assert_eq!(
        field_ident("_responseJsonSchema", &taken),
        "legacy_response_json_schema"
    );
    let taken = BTreeSet::from(["response_json_schema".to_owned()]);
    assert_eq!(
        field_ident("responseJsonSchema", &taken),
        "response_json_schema_"
    );
}

#[test]
fn enum_variants_drop_the_shared_prefix() {
    let values: Vec<String> = ["MEDIA_RESOLUTION_UNSPECIFIED", "MEDIA_RESOLUTION_LOW"]
        .map(str::to_owned)
        .to_vec();
    assert_eq!(common_prefix(&values), "MEDIA_RESOLUTION_");
    assert_eq!(
        variant_ident("MEDIA_RESOLUTION_LOW", "MEDIA_RESOLUTION_"),
        "Low"
    );
    assert_eq!(variant_ident("unknown", ""), "UnknownValue");
    assert_eq!(variant_ident("1080p", ""), "V1080p");
}

#[test]
fn a_prefix_keeps_one_word_in_every_variant() {
    let values: Vec<String> = ["BLOCK", "BLOCK_NONE"].map(str::to_owned).to_vec();
    assert_eq!(common_prefix(&values), "");
}

#[test]
fn bare_code_fences_are_not_doctests() {
    assert_eq!(
        doc("a\n```\n{}\n```", ""),
        "/// a\n/// ```text\n/// {}\n/// ```\n"
    );
}
