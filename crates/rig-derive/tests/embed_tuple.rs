use rig_core::embeddings::{EmbedError, TextEmbedder, to_texts};

#[derive(rig_derive::Embed)]
struct BasicTuple(u8, #[embed] String, bool, #[embed] String);

fn embed_count(embedder: &mut TextEmbedder, value: usize) -> Result<(), EmbedError> {
    embedder.embed(format!("count:{value}"));
    Ok(())
}

#[derive(rig_derive::Embed)]
struct MixedTuple<T>(
    bool,
    #[embed] T,
    u8,
    #[embed(embed_with = "embed_count")] usize,
);

#[derive(rig_derive::Embed)]
struct Named<T> {
    #[embed]
    body: T,
    #[embed(embed_with = "embed_count")]
    count: usize,
}

#[test]
fn tuple_fields_keep_their_original_indices() -> Result<(), EmbedError> {
    let document = BasicTuple(7, "first".into(), false, "second".into());
    assert_eq!(document.0, 7);
    assert!(!document.2);
    assert_eq!(to_texts(document)?, vec!["first", "second"]);
    Ok(())
}

#[test]
fn generic_tuple_supports_basic_and_custom_fields() -> Result<(), EmbedError> {
    let document = MixedTuple(false, "body", 9, 42);
    assert!(!document.0);
    assert_eq!(document.2, 9);
    assert_eq!(to_texts(document)?, vec!["body", "count:42"]);
    Ok(())
}

#[test]
fn named_fields_retain_basic_and_custom_dispatch() -> Result<(), EmbedError> {
    assert_eq!(
        to_texts(Named {
            body: "body",
            count: 42
        })?,
        vec!["body", "count:42"],
    );
    Ok(())
}
