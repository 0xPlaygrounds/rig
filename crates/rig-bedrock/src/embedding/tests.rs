use rig_core::wire::{Capabilities, Mode, Wire};

use super::{AMAZON_TITAN_EMBED_TEXT_V2_0, Embeddings};

/// The `InvokeModel` body the wire encodes for one text.
fn sent_body(wire: &Embeddings) -> serde_json::Value {
    let batch = wire
        .encode(vec!["text".to_owned()], Mode::Unary)
        .expect("the batch encodes");
    let (_, body) = batch.texts.first().expect("one request per text");
    serde_json::from_str(body).expect("the body is JSON")
}

/// Titan rejects `dimensions: 0`, so a wire whose caller named no width
/// sends none and the model answers at its default.
#[test]
fn an_unnamed_width_is_not_sent() {
    let wire = Embeddings::new(AMAZON_TITAN_EMBED_TEXT_V2_0);

    assert_eq!(
        sent_body(&wire),
        serde_json::json!({ "inputText": "text", "normalize": true })
    );
    assert_eq!(
        wire.describe().capabilities,
        Capabilities::embedding(1024, 1024)
    );
}

/// The recorded `embeddings_smoke` request body, and a declaration the
/// reply is checked against.
#[test]
fn a_named_width_is_sent_and_declared() {
    let wire = Embeddings::new(AMAZON_TITAN_EMBED_TEXT_V2_0).with_ndims(256);

    assert_eq!(
        sent_body(&wire),
        serde_json::json!({ "inputText": "text", "dimensions": 256, "normalize": true })
    );
    assert_eq!(
        wire.describe().capabilities,
        Capabilities::embedding(1024, 256).declaring(Some(256))
    );
}
