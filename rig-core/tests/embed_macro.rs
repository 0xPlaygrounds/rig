use rig::{
    embeddings::{embed::EmbedError, TextEmbedder},
    Embed,
};
use serde::Serialize;

fn serialize(embedder: &mut TextEmbedder, definition: Definition) -> Result<(), EmbedError> {
    embedder.embed(serde_json::to_string(&definition).map_err(EmbedError::new)?);

    Ok(())
}

#[derive(Embed)]
struct FakeDefinition {
    id: String,
    word: String,
    #[embed(embed_with = "serialize")]
    definition: Definition,
}

#[derive(Serialize, Clone)]
struct Definition {
    word: String,
    link: String,
    speech: String,
}

#[test]
fn test_custom_embed() {
    let fake_definition = FakeDefinition {
        id: "doc1".to_string(),
        word: "house".to_string(),
        definition: Definition {
            speech: "noun".to_string(),
            word: "a building in which people live; residence for human beings.".to_string(),
            link: "https://www.dictionary.com/browse/house".to_string(),
        },
    };

    println!(
        "FakeDefinition: {}, {}",
        fake_definition.id, fake_definition.word
    );

    let embedder = &mut TextEmbedder::default();
    fake_definition.embed(embedder).unwrap();

    assert_eq!(
            embedder.texts.first().unwrap().clone(),
            "{\"word\":\"a building in which people live; residence for human beings.\",\"link\":\"https://www.dictionary.com/browse/house\",\"speech\":\"noun\"}".to_string()

        )
}

#[derive(Embed)]
struct FakeDefinition2 {
    id: String,
    #[embed]
    word: String,
    #[embed(embed_with = "serialize")]
    definition: Definition,
}

#[test]
fn test_custom_and_basic_embed() {
    let fake_definition = FakeDefinition2 {
        id: "doc1".to_string(),
        word: "house".to_string(),
        definition: Definition {
            speech: "noun".to_string(),
            word: "a building in which people live; residence for human beings.".to_string(),
            link: "https://www.dictionary.com/browse/house".to_string(),
        },
    };

    println!(
        "FakeDefinition: {}, {}",
        fake_definition.id, fake_definition.word
    );

    let embedder = &mut TextEmbedder::default();
    fake_definition.embed(embedder).unwrap();

    assert_eq!(embedder.texts.first().unwrap().clone(), "house".to_string());

    assert_eq!(
        embedder.texts.last().unwrap().clone(),
        "{\"word\":\"a building in which people live; residence for human beings.\",\"link\":\"https://www.dictionary.com/browse/house\",\"speech\":\"noun\"}".to_string()
    )
}

#[derive(Embed)]
struct FakeDefinition3 {
    id: String,
    word: String,
    #[embed]
    definition: String,
}

#[test]
fn test_single_embed() {
    let definition = "a building in which people live; residence for human beings.".to_string();

    let fake_definition = FakeDefinition3 {
        id: "doc1".to_string(),
        word: "house".to_string(),
        definition: definition.clone(),
    };
    println!(
        "FakeDefinition3: {}, {}",
        fake_definition.id, fake_definition.word
    );

    let embedder = &mut TextEmbedder::default();
    fake_definition.embed(embedder).unwrap();

    assert_eq!(embedder.texts.first().unwrap().clone(), definition)
}

#[derive(Embed)]
struct Company {
    id: String,
    company: String,
    #[embed]
    employee_ages: Vec<i32>,
}

#[test]
fn test_multiple_embed_strings() {
    let company = Company {
        id: "doc1".to_string(),
        company: "Google".to_string(),
        employee_ages: vec![25, 30, 35, 40],
    };

    println!("Company: {}, {}", company.id, company.company);

    let embedder = &mut TextEmbedder::default();
    company.embed(embedder).unwrap();

    assert_eq!(
        embedder.texts,
        vec![
            "25".to_string(),
            "30".to_string(),
            "35".to_string(),
            "40".to_string()
        ]
    );
}

#[derive(Embed)]
struct Company2 {
    id: String,
    #[embed]
    company: String,
    #[embed]
    employee_ages: Vec<i32>,
}

#[test]
fn test_multiple_embed_tags() {
    let company = Company2 {
        id: "doc1".to_string(),
        company: "Google".to_string(),
        employee_ages: vec![25, 30, 35, 40],
    };

    println!("Company: {}", company.id);

    let embedder = &mut TextEmbedder::default();
    company.embed(embedder).unwrap();

    assert_eq!(
        embedder.texts,
        vec![
            "Google".to_string(),
            "25".to_string(),
            "30".to_string(),
            "35".to_string(),
            "40".to_string()
        ]
    );
}
