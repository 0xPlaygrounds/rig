//! Tests for `#[derive(ContextValue)]`.

#![allow(
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::unwrap_used
)]

use rig_core::tool::{ContextValue, ToolContext};

#[derive(Debug, PartialEq, serde::Serialize, serde::Deserialize, rig_core::ContextValue)]
struct SessionId(String);

#[derive(Debug, PartialEq, serde::Serialize, serde::Deserialize, rig_core::ContextValue)]
#[context(key = "session.tenant")]
struct Tenant {
    name: String,
}

#[test]
fn default_key_is_the_type_name() {
    assert_eq!(<SessionId as ContextValue>::KEY, "SessionId");
}

#[test]
fn context_attribute_overrides_the_key() {
    assert_eq!(<Tenant as ContextValue>::KEY, "session.tenant");
}

#[test]
fn derived_values_round_trip_through_a_context_under_their_key() {
    let mut context = ToolContext::new();
    context.insert(SessionId("abc".into())).unwrap();
    context
        .insert(Tenant {
            name: "acme".into(),
        })
        .unwrap();

    assert_eq!(
        context.get::<SessionId>().unwrap(),
        Some(SessionId("abc".into()))
    );
    assert_eq!(
        context.get::<Tenant>().unwrap(),
        Some(Tenant {
            name: "acme".into()
        })
    );

    let json = serde_json::to_value(&context).unwrap();
    assert_eq!(json["inbound"]["SessionId"], serde_json::json!("abc"));
    assert_eq!(
        json["inbound"]["session.tenant"],
        serde_json::json!({"name": "acme"})
    );
}
