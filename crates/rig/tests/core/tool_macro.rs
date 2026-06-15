#[derive(serde::Deserialize, rig::schemars::JsonSchema)]
struct FacadeCoordinates {
    /// Latitude
    lat: f64,
    /// Longitude
    lng: f64,
}

/// Find nearby places
#[rig::tool_macro]
fn facade_find_nearby(
    /// Location to search from
    location: FacadeCoordinates,
) -> Result<String, rig::tool::ToolError> {
    Ok(format!("{},{}", location.lat, location.lng))
}

#[tokio::test]
async fn test_tool_macro_accepts_facade_schemars_reexport() {
    use rig::tool::Tool;

    let definition = FacadeFindNearby.definition(String::default()).await;
    let schema = serde_json::to_string(&definition.parameters).unwrap();

    assert!(schema.contains("lat"), "expected lat in schema: {schema}");
    assert!(schema.contains("lng"), "expected lng in schema: {schema}");
    assert!(
        schema.contains("Latitude"),
        "expected Latitude description in schema: {schema}"
    );
    assert_eq!(definition.description, "Find nearby places");
}
