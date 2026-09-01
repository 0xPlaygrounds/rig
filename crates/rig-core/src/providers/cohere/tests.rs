use super::*;

#[test]
fn embedding_dimensions_cover_every_live_embed_model() {
    assert_eq!(model_dimensions_from_identifier(EMBED_V4), Some(1_536));
    assert_eq!(
        model_dimensions_from_identifier(EMBED_ENGLISH_V3),
        Some(1_024)
    );
    assert_eq!(
        model_dimensions_from_identifier(EMBED_MULTILINGUAL_V3),
        Some(1_024)
    );
    assert_eq!(
        model_dimensions_from_identifier(EMBED_ENGLISH_LIGHT_V3),
        Some(384)
    );
    assert_eq!(
        model_dimensions_from_identifier(EMBED_MULTILINGUAL_LIGHT_V3),
        Some(384)
    );
    assert_eq!(model_dimensions_from_identifier("embed-unknown"), None);
}
