use rig_core::wire::{Capabilities, Mode, Wire};

use super::{AMAZON_TITAN_EMBED_TEXT_V2_0, Embeddings};

/// The request bodies `wire` sends for two texts.
fn bodies(wire: &Embeddings) -> Vec<serde_json::Value> {
    wire.encode(vec!["a".to_owned(), "b".to_owned()], Mode::Unary)
        .map(|batch| {
            batch
                .texts
                .iter()
                .filter_map(|(_, body)| serde_json::from_str(body).ok())
                .collect()
        })
        .unwrap_or_default()
}

/// Without a caller's width the request carries no `dimensions`, which Titan
/// reads as its default, and the wire reports that default.
#[test]
fn no_width_is_sent_unless_the_caller_names_one() {
    let wire = Embeddings::new(AMAZON_TITAN_EMBED_TEXT_V2_0, None);

    assert_eq!(
        bodies(&wire),
        vec![
            serde_json::json!({ "inputText": "a", "normalize": true }),
            serde_json::json!({ "inputText": "b", "normalize": true }),
        ]
    );
    assert_eq!(
        wire.describe().capabilities,
        Capabilities::embedding(1024, 1024)
    );
}

/// The caller's width is sent and declared, so a reply of another width
/// fails instead of reaching a vector store.
#[test]
fn a_named_width_is_sent_and_declared() {
    let wire = Embeddings::new(AMAZON_TITAN_EMBED_TEXT_V2_0, Some(256));

    assert_eq!(
        bodies(&wire),
        vec![
            serde_json::json!({ "inputText": "a", "dimensions": 256, "normalize": true }),
            serde_json::json!({ "inputText": "b", "dimensions": 256, "normalize": true }),
        ]
    );
    assert_eq!(
        wire.describe().capabilities,
        Capabilities::embedding(1024, 256).declaring(Some(256))
    );
}
