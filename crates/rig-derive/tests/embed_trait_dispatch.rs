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
fn annotated_field_uses_its_embed_trait() -> Result<(), EmbedError> {
    Leaf.embed(&mut TextEmbedder::default())?;
    assert_eq!(to_texts(Document { leaf: Leaf })?, vec!["trait"]);
    Ok(())
}
