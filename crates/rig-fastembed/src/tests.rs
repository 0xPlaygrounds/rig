use rig_core::wire::{Capabilities, Wire};

use super::{FastembedModel, text_embeddings};

/// A local model cannot change its width, so a caller's width is declared
/// and a vector of any other width fails rather than sizing a vector store
/// wrongly. Without one, the model metadata's width is reported.
#[test]
fn a_callers_width_is_declared_and_the_models_is_reported() {
    let model = FastembedModel::AllMiniLML6V2;
    let capabilities = |ndims| {
        text_embeddings(&model, ndims)
            .map(|wire| wire.describe().capabilities)
            .unwrap_or_default()
    };

    assert_eq!(capabilities(None), Capabilities::embedding(1024, 384));
    assert_eq!(
        capabilities(Some(512)),
        Capabilities::embedding(1024, 512).declaring(Some(512))
    );
}
