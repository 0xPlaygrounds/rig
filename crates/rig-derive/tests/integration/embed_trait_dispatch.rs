use rig_core::embeddings::{Embed, EmbedError, TextEmbedder, to_texts};

struct Leaf;

impl Leaf {
    fn embed(&self, embedder: &mut TextEmbedder) -> Result<(), EmbedError> {
        embedder.embed("inherent".into());
        Ok(())
    }
}

impl Embed for Leaf {
    fn embed(&self, embedder: &mut TextEmbedder) -> Result<(), EmbedError> {
        embedder.embed("trait".into());
        Ok(())
    }
}

#[derive(rig_derive::Embed)]
struct Document {
    #[embed]
    leaf: Leaf,
}

#[test]
fn annotated_field_uses_its_embed_trait() {
    assert!(Leaf.embed(&mut TextEmbedder::default()).is_ok());
    assert_eq!(
        to_texts(Document { leaf: Leaf }).ok(),
        Some(vec!["trait".to_owned()])
    );
}
