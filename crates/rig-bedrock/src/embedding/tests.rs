use super::*;
use rig_core::embeddings::EmbeddingWidth;

/// The request bodies the batch would send.
fn bodies(wire: &Embeddings) -> Vec<serde_json::Value> {
    wire.encode(vec!["text".to_owned()], Mode::Unary)
        .expect("the batch encodes")
        .texts
        .iter()
        .map(|(_, body)| serde_json::from_str(body).expect("each body is JSON"))
        .collect()
}

/// An undeclared width sends no `dimensions`, so Titan answers at its native
/// width instead of rejecting a zero.
#[test]
fn an_undeclared_width_is_not_sent() {
    let wire = Embeddings::new(AMAZON_TITAN_EMBED_TEXT_V2_0);

    assert_eq!(
        bodies(&wire),
        vec![serde_json::json!({ "inputText": "text", "normalize": true })]
    );
    assert_eq!(
        wire.describe().capabilities,
        Capabilities::embedding(1024, 1024)
    );
}

#[test]
fn a_declared_width_is_sent_and_checked() {
    let wire = Embeddings::new(AMAZON_TITAN_EMBED_TEXT_V2_0).with_ndims(256);

    assert_eq!(
        bodies(&wire),
        vec![serde_json::json!({ "inputText": "text", "dimensions": 256, "normalize": true })]
    );
    assert_eq!(
        wire.describe().capabilities,
        Capabilities::embedding(1024, 256).declaring(Some(256))
    );
}
