use rig::embeddings::embeddable::{EmbeddableError, OneOrMany};
use rig::Embeddable;
use serde::Serialize;

fn serialize(definition: Definition) -> Result<OneOrMany<String>, EmbeddableError> {
    Ok(OneOrMany::from(
        serde_json::to_string(&definition).map_err(EmbeddableError::SerdeError)?,
    ))
}

#[derive(Embeddable)]
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
            fake_definition.embeddable().unwrap(),
            OneOrMany::from(
                "{\"word\":\"a building in which people live; residence for human beings.\",\"link\":\"https://www.dictionary.com/browse/house\",\"speech\":\"noun\"}".to_string()
            )

        )
}

#[derive(Embeddable)]
struct FakeDefinition2 {
    id: String,
    word: String,
    #[embed]
    definition: String,
}

#[test]
fn test_single_embed() {
    let definition = "a building in which people live; residence for human beings.".to_string();

    let fake_definition = FakeDefinition2 {
        id: "doc1".to_string(),
        word: "house".to_string(),
        definition: definition.clone(),
    };

    println!(
        "FakeDefinition2: {}, {}",
        fake_definition.id, fake_definition.word
    );

    assert_eq!(
        fake_definition.embeddable().unwrap(),
        OneOrMany::from(definition)
    )
}

#[derive(Embeddable)]
struct Company {
    id: String,
    company: String,
    #[embed]
    employee_ages: Vec<i32>,
}

#[test]
fn test_multiple_embed() {
    let company = Company {
        id: "doc1".to_string(),
        company: "Google".to_string(),
        employee_ages: vec![25, 30, 35, 40],
    };

    println!("Company: {}, {}", company.id, company.company);

    assert_eq!(
        company.embeddable().unwrap(),
        OneOrMany::try_from(vec![
            "25".to_string(),
            "30".to_string(),
            "35".to_string(),
            "40".to_string()
        ])
        .unwrap()
    );
}

#[derive(Embeddable)]
struct Company2 {
    id: String,
    #[embed]
    company: String,
    #[embed]
    employee_ages: Vec<i32>,
}

#[test]
fn test_many_embed() {
    let company = Company2 {
        id: "doc1".to_string(),
        company: "Google".to_string(),
        employee_ages: vec![25, 30, 35, 40],
    };

    println!("Company2: {}", company.id);

    assert_eq!(
        company.embeddable().unwrap(),
        OneOrMany::try_from(vec![
            "Google".to_string(),
            "25".to_string(),
            "30".to_string(),
            "35".to_string(),
            "40".to_string()
        ])
        .unwrap()
    );
}
