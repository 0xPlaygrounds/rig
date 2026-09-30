use rig_core::wire::{Mode, Wire};

use super::{AMAZON_TITAN_EMBED_TEXT_V2_0, Embeddings};

fn first_body(wire: &Embeddings) -> serde_json::Value {
    let batch = wire
        .encode(vec!["text".to_owned()], Mode::Unary)
        .expect("the request encodes");
    let (_, body) = batch.texts.first().expect("one text is one request");
    serde_json::from_str(body).expect("the body is one JSON document")
}

/// Without a width the body names none, so the model answers at its own
/// default instead of being asked for zero dimensions.
#[test]
fn an_unset_width_is_not_sent() {
    let wire = Embeddings::new(AMAZON_TITAN_EMBED_TEXT_V2_0);

    assert_eq!(
        first_body(&wire),
        serde_json::json!({ "inputText": "text", "normalize": true })
    );
    assert_eq!(wire.describe().capabilities.declared, None);
}

#[test]
fn with_ndims_is_sent_and_declared() {
    let wire = Embeddings::new(AMAZON_TITAN_EMBED_TEXT_V2_0).with_ndims(256);

    assert_eq!(
        first_body(&wire).get("dimensions"),
        Some(&serde_json::json!(256))
    );
    assert_eq!(wire.describe().capabilities.ndims, 256);
    assert_eq!(wire.describe().capabilities.declared, Some(256));
}
