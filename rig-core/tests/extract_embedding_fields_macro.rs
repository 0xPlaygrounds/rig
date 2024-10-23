use rig::embeddings::extract_embedding_fields::ExtractEmbeddingFieldsError;
use rig::{ExtractEmbeddingFields, OneOrMany};
use serde::Serialize;

fn serialize(definition: Definition) -> Result<OneOrMany<String>, ExtractEmbeddingFieldsError> {
    Ok(OneOrMany::one(
        serde_json::to_string(&definition).map_err(ExtractEmbeddingFieldsError::new)?,
    ))
}

#[derive(ExtractEmbeddingFields)]
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

    assert_eq!(
            fake_definition.extract_embedding_fields().unwrap(),
            OneOrMany::one(
                "{\"word\":\"a building in which people live; residence for human beings.\",\"link\":\"https://www.dictionary.com/browse/house\",\"speech\":\"noun\"}".to_string()
            )

        )
}

#[derive(ExtractEmbeddingFields)]
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

    assert_eq!(
        fake_definition.extract_embedding_fields().unwrap().first(),
        "house".to_string()
    );

    assert_eq!(
        fake_definition.extract_embedding_fields().unwrap().rest(),
        vec!["{\"word\":\"a building in which people live; residence for human beings.\",\"link\":\"https://www.dictionary.com/browse/house\",\"speech\":\"noun\"}".to_string()]
    )
}

#[derive(ExtractEmbeddingFields)]
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

    assert_eq!(
        fake_definition.extract_embedding_fields().unwrap(),
        OneOrMany::one(definition)
    )
}

#[derive(ExtractEmbeddingFields)]
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

    let result = company.extract_embedding_fields().unwrap();

    assert_eq!(
        result,
        OneOrMany::many(vec![
            "25".to_string(),
            "30".to_string(),
            "35".to_string(),
            "40".to_string()
        ])
        .unwrap()
    );

    assert_eq!(result.first(), "25".to_string());

    assert_eq!(
        result.rest(),
        vec!["30".to_string(), "35".to_string(), "40".to_string()]
    )
}

#[derive(ExtractEmbeddingFields)]
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
        company.extract_embedding_fields().unwrap(),
        OneOrMany::many(vec![
            "Google".to_string(),
            "25".to_string(),
            "30".to_string(),
            "35".to_string(),
            "40".to_string()
        ])
        .unwrap()
    );
}
