use super::*;

/// Serialization shape of the request block is definitory, not observed:
/// the cassette suite pins that Venice *accepts* it, this pins that
/// unset fields stay off the wire entirely rather than being sent null.
#[test]
fn venice_parameters_only_serialize_set_fields() {
    let params = VeniceParameters::new()
        .enable_web_search(WebSearchMode::Auto)
        .disable_thinking(true);

    let json = serde_json::to_value(&params).expect("parameters should serialize");

    assert_eq!(
        json,
        serde_json::json!({
            "enable_web_search": "auto",
            "disable_thinking": true,
        })
    );
}

#[test]
fn venice_parameters_wrap_into_additional_params() {
    let json = VeniceParameters::new()
        .character_slug("venice")
        .into_additional_params();

    assert_eq!(
        json,
        serde_json::json!({ "venice_parameters": { "character_slug": "venice" } })
    );
}
