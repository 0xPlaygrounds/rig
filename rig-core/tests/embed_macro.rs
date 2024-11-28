use rig::{
    embeddings::{embed::EmbedError, TextEmbedder},
    to_texts, Embed,
};
use serde::Serialize;

fn serialize(embedder: &mut TextEmbedder, definition: Definition) -> Result<(), EmbedError> {
    embedder.embed(serde_json::to_string(&definition).map_err(EmbedError::new)?);

    Ok(())
}

#[derive(Embed)]
struct WordDefinition {
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
    let fake_definition = WordDefinition {
        id: "doc1".to_string(),
        word: "house".to_string(),
        definition: Definition {
            speech: "noun".to_string(),
            word: "a building in which people live; residence for human beings.".to_string(),
            link: "https://www.dictionary.com/browse/house".to_string(),
        },
    };

    println!(
        "WordDefinition: {}, {}",
        fake_definition.id, fake_definition.word
    );

    assert_eq!(
        to_texts(fake_definition).unwrap().first().unwrap().clone(),
            "{\"word\":\"a building in which people live; residence for human beings.\",\"link\":\"https://www.dictionary.com/browse/house\",\"speech\":\"noun\"}".to_string()

        )
}

#[derive(Embed)]
struct WordDefinition2 {
    id: String,
    #[embed]
    word: String,
    #[embed(embed_with = "serialize")]
    definition: Definition,
}

#[test]
fn test_custom_and_basic_embed() {
    let fake_definition = WordDefinition2 {
        id: "doc1".to_string(),
        word: "house".to_string(),
        definition: Definition {
            speech: "noun".to_string(),
            word: "a building in which people live; residence for human beings.".to_string(),
            link: "https://www.dictionary.com/browse/house".to_string(),
        },
    };

    println!(
        "WordDefinition: {}, {}",
        fake_definition.id, fake_definition.word
    );

    let texts = to_texts(fake_definition).unwrap();

    assert_eq!(texts.first().unwrap().clone(), "house".to_string());

    assert_eq!(
        texts.last().unwrap().clone(),
        "{\"word\":\"a building in which people live; residence for human beings.\",\"link\":\"https://www.dictionary.com/browse/house\",\"speech\":\"noun\"}".to_string()
    )
}

#[derive(Embed)]
struct WordDefinition3 {
    id: String,
    word: String,
    #[embed]
    definition: String,
}

#[test]
fn test_single_embed() {
    let definition = "a building in which people live; residence for human beings.".to_string();

    let fake_definition = WordDefinition3 {
        id: "doc1".to_string(),
        word: "house".to_string(),
        definition: definition.clone(),
    };
    println!(
        "WordDefinition3: {}, {}",
        fake_definition.id, fake_definition.word
    );

    assert_eq!(
        to_texts(fake_definition).unwrap().first().unwrap().clone(),
        definition
    )
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

    assert_eq!(
        to_texts(company).unwrap(),
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

    assert_eq!(
        to_texts(company).unwrap(),
        vec![
            "Google".to_string(),
            "25".to_string(),
            "30".to_string(),
            "35".to_string(),
            "40".to_string()
        ]
    );
}
