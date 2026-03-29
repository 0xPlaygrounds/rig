//! Tests for schemars-based schema generation in `#[rig_tool]`.

use std::collections::HashMap;

use rig::tool::Tool;
use rig_derive::rig_tool;

// --- Doc comment description ---

/// Add two numbers
#[rig_tool]
fn add_doc(
    /// First number
    a: i32,
    /// Second number
    b: i32,
) -> Result<i32, rig::tool::ToolError> {
    Ok(a + b)
}

#[tokio::test]
async fn test_doc_comment_description() {
    let def = AddDoc.definition(String::default()).await;
    assert_eq!(def.description, "Add two numbers");
}

#[tokio::test]
async fn test_param_doc_comments() {
    let def = AddDoc.definition(String::default()).await;
    let props = def.parameters["properties"].as_object().unwrap();
    assert_eq!(props["a"]["description"], "First number");
    assert_eq!(props["b"]["description"], "Second number");
}

// --- Explicit overrides doc ---

/// This doc comment should be ignored
#[rig_tool(
    description = "Override description",
    params(query = "Override param doc")
)]
fn search_override(
    /// This param doc should be ignored
    query: String,
) -> Result<String, rig::tool::ToolError> {
    Ok(query)
}

#[tokio::test]
async fn test_explicit_overrides_doc() {
    let def = SearchOverride.definition(String::default()).await;
    assert_eq!(def.description, "Override description");
    let props = def.parameters["properties"].as_object().unwrap();
    assert_eq!(props["query"]["description"], "Override param doc");
}

// --- Option<T> nullable ---

/// Search documents
#[rig_tool]
fn search_optional(
    /// The search query
    query: String,
    /// Maximum results
    limit: Option<i32>,
) -> Result<String, rig::tool::ToolError> {
    Ok(format!("{query} limit={limit:?}"))
}

#[tokio::test]
async fn test_option_nullable() {
    let def = SearchOptional.definition(String::default()).await;
    let props = def.parameters["properties"].as_object().unwrap();

    // query is a plain string
    assert_eq!(props["query"]["type"], "string");

    // limit is nullable — schemars represents Option<T> as "type": ["integer", "null"]
    let limit = &props["limit"];
    let types = limit["type"].as_array().unwrap();
    let type_names: Vec<&str> = types.iter().filter_map(|v| v.as_str()).collect();
    assert!(
        type_names.contains(&"integer"),
        "expected integer in type array"
    );
    assert!(type_names.contains(&"null"), "expected null in type array");
}

#[tokio::test]
async fn test_option_deserialization() {
    // Option<T> field absent -> None
    let json = serde_json::json!({"query": "test"});
    let params: SearchOptionalParameters = serde_json::from_value(json).unwrap();
    assert_eq!(params.limit, None);

    // Option<T> field null -> None
    let json = serde_json::json!({"query": "test", "limit": null});
    let params: SearchOptionalParameters = serde_json::from_value(json).unwrap();
    assert_eq!(params.limit, None);

    // Option<T> field present -> Some
    let json = serde_json::json!({"query": "test", "limit": 10});
    let params: SearchOptionalParameters = serde_json::from_value(json).unwrap();
    assert_eq!(params.limit, Some(10));
}

// --- Integer vs number ---

/// Test numeric types
#[rig_tool]
fn numeric_types(
    /// An integer
    int_val: i32,
    /// A float
    float_val: f64,
) -> Result<String, rig::tool::ToolError> {
    Ok(format!("{int_val} {float_val}"))
}

#[tokio::test]
async fn test_integer_vs_number() {
    let def = NumericTypes.definition(String::default()).await;
    let props = def.parameters["properties"].as_object().unwrap();
    assert_eq!(props["int_val"]["type"], "integer");
    assert_eq!(props["float_val"]["type"], "number");
}

// --- Vec param ---

/// Sum numbers
#[rig_tool]
fn sum_vec(
    /// Numbers to sum
    numbers: Vec<i64>,
) -> Result<i64, rig::tool::ToolError> {
    Ok(numbers.iter().sum())
}

#[tokio::test]
async fn test_vec_param() {
    let def = SumVec.definition(String::default()).await;
    let props = def.parameters["properties"].as_object().unwrap();
    assert_eq!(props["numbers"]["type"], "array");
    assert_eq!(props["numbers"]["items"]["type"], "integer");
}

// --- No params ---

/// Return constant
#[rig_tool]
fn no_params() -> Result<i32, rig::tool::ToolError> {
    Ok(42)
}

#[tokio::test]
async fn test_no_params() {
    let def = NoParams.definition(String::default()).await;
    let required = def.parameters["required"].as_array().unwrap();
    assert!(required.is_empty());

    // properties may be absent or empty for a zero-field struct
    if let Some(props) = def.parameters["properties"].as_object() {
        assert!(props.is_empty());
    }
}

// --- Bool param ---

/// Toggle flag
#[rig_tool]
fn toggle(
    /// Whether to enable
    enabled: bool,
) -> Result<bool, rig::tool::ToolError> {
    Ok(!enabled)
}

#[tokio::test]
async fn test_bool_param() {
    let def = Toggle.definition(String::default()).await;
    let props = def.parameters["properties"].as_object().unwrap();
    assert_eq!(props["enabled"]["type"], "boolean");
}

// --- Default description fallback ---

#[rig_tool]
fn no_docs(x: i32) -> Result<i32, rig::tool::ToolError> {
    Ok(x)
}

#[tokio::test]
async fn test_default_description_fallback() {
    let def = NoDocs.definition(String::default()).await;
    assert_eq!(def.description, "Function to no_docs");

    let props = def.parameters["properties"].as_object().unwrap();
    assert_eq!(props["x"]["description"], "Parameter x");
}

// --- Schema type is "object" ---

#[tokio::test]
async fn test_schema_type_object() {
    let def = AddDoc.definition(String::default()).await;
    assert_eq!(def.parameters["type"], "object");
}

// --- All params in required by default ---

#[tokio::test]
async fn test_required_all_by_default() {
    let def = AddDoc.definition(String::default()).await;
    let required = def.parameters["required"].as_array().unwrap();
    let names: Vec<&str> = required.iter().filter_map(|v| v.as_str()).collect();
    assert_eq!(names.len(), 2);
    assert!(names.contains(&"a"));
    assert!(names.contains(&"b"));
}

// --- Enum param ---

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub enum SortOrder {
    #[serde(rename = "asc")]
    Ascending,
    #[serde(rename = "desc")]
    Descending,
}

/// Sort items
#[rig_tool]
fn sort_items(
    /// Sort direction
    order: SortOrder,
) -> Result<String, rig::tool::ToolError> {
    Ok(format!("{order:?}"))
}

#[tokio::test]
async fn test_enum_param() {
    let def = SortItems.definition(String::default()).await;
    let schema_str = serde_json::to_string(&def.parameters).unwrap();

    // schemars may use $defs/$ref or inline the enum — verify the renamed
    // variants appear somewhere in the schema
    assert!(schema_str.contains("asc"), "expected 'asc' in schema");
    assert!(schema_str.contains("desc"), "expected 'desc' in schema");
}

// --- HashMap param ---

/// Store metadata
#[rig_tool]
fn store_metadata(
    /// Key-value pairs
    metadata: HashMap<String, String>,
) -> Result<String, rig::tool::ToolError> {
    Ok(format!("{metadata:?}"))
}

#[tokio::test]
async fn test_hashmap_param() {
    let def = StoreMetadata.definition(String::default()).await;
    let props = def.parameters["properties"].as_object().unwrap();
    let meta = &props["metadata"];
    assert_eq!(meta["type"], "object");
    // HashMap<String, String> should have additionalProperties with string type
    assert_eq!(meta["additionalProperties"]["type"], "string");
}

// --- Nested struct param ---

#[derive(serde::Deserialize, schemars::JsonSchema)]
pub struct Coordinates {
    /// Latitude
    pub lat: f64,
    /// Longitude
    pub lng: f64,
}

/// Find nearby places
#[rig_tool]
fn find_nearby(
    /// The location to search from
    location: Coordinates,
    /// Search radius in km
    radius: f64,
) -> Result<Vec<String>, rig::tool::ToolError> {
    Ok(vec![format!(
        "{},{} r={radius}",
        location.lat, location.lng
    )])
}

#[tokio::test]
async fn test_nested_struct_param() {
    let def = FindNearby.definition(String::default()).await;
    let props = def.parameters["properties"].as_object().unwrap();

    // The location field should reference a nested struct definition
    let _location = &props["location"];
    // schemars may inline the struct or use $ref/$defs
    // Either way, the schema should mention lat and lng
    let schema_str = serde_json::to_string(&def.parameters).unwrap();
    assert!(schema_str.contains("lat"), "expected 'lat' in schema");
    assert!(schema_str.contains("lng"), "expected 'lng' in schema");

    // radius should be a number
    assert_eq!(props["radius"]["type"], "number");

    // Verify the nested struct's field descriptions are present somewhere
    assert!(
        schema_str.contains("Latitude"),
        "expected 'Latitude' description in schema"
    );
    assert!(
        schema_str.contains("Longitude"),
        "expected 'Longitude' description in schema"
    );
}

// --- Async tool with doc comments ---

/// Fetch a URL asynchronously
#[rig_tool]
async fn fetch_url(
    /// The URL to fetch
    url: String,
) -> Result<String, rig::tool::ToolError> {
    Ok(format!("fetched: {url}"))
}

#[tokio::test]
async fn test_async_tool_with_docs() {
    let def = FetchUrl.definition(String::default()).await;
    assert_eq!(def.description, "Fetch a URL asynchronously");
    assert_eq!(def.name, "fetch_url");

    let props = def.parameters["properties"].as_object().unwrap();
    assert_eq!(props["url"]["description"], "The URL to fetch");

    // Verify it actually works (async call)
    let result = FetchUrl
        .call(FetchUrlParameters {
            url: "https://example.com".to_string(),
        })
        .await
        .unwrap();
    assert_eq!(result, serde_json::json!("fetched: https://example.com"));
}
