use rig::{
    embeddings::{embed::EmbedError, TextEmbedder},
    to_texts, Embed,
};
use serde::Serialize;

#[test]
fn test_custom_embed() {
    #[derive(Embed)]
    struct WordDefinition {
        #[allow(dead_code)]
        id: String,
        #[allow(dead_code)]
        word: String,
        #[embed(embed_with = "custom_embedding_function")]
        definition: Definition,
    }
    
    #[derive(Serialize, Clone)]
    struct Definition {
        word: String,
        link: String,
        speech: String,
    }    

    fn custom_embedding_function(embedder: &mut TextEmbedder, definition: Definition) -> Result<(), EmbedError> {
        embedder.embed(serde_json::to_string(&definition).map_err(EmbedError::new)?);
    
        Ok(())
    }
    
    let fake_definition = WordDefinition {
        id: "doc1".to_string(),
        word: "house".to_string(),
        definition: Definition {
            speech: "noun".to_string(),
            word: "a building in which people live; residence for human beings.".to_string(),
            link: "https://www.dictionary.com/browse/house".to_string(),
        },
    };

    assert_eq!(
        to_texts(fake_definition).unwrap().first().unwrap().clone(),
            "{\"word\":\"a building in which people live; residence for human beings.\",\"link\":\"https://www.dictionary.com/browse/house\",\"speech\":\"noun\"}".to_string()
        )
}

#[test]
fn test_custom_and_basic_embed() {
    #[derive(Embed)]
    struct WordDefinition {
        #[allow(dead_code)]
        id: String,
        #[embed]
        word: String,
        #[embed(embed_with = "custom_embedding_function")]
        definition: Definition,
    }

    #[derive(Serialize, Clone)]
    struct Definition {
        word: String,
        link: String,
        speech: String,
    }    

    fn custom_embedding_function(embedder: &mut TextEmbedder, definition: Definition) -> Result<(), EmbedError> {
        embedder.embed(serde_json::to_string(&definition).map_err(EmbedError::new)?);
    
        Ok(())
    }

    let fake_definition = WordDefinition {
        id: "doc1".to_string(),
        word: "house".to_string(),
        definition: Definition {
            speech: "noun".to_string(),
            word: "a building in which people live; residence for human beings.".to_string(),
            link: "https://www.dictionary.com/browse/house".to_string(),
        },
    };

    let texts = to_texts(fake_definition).unwrap();

    assert_eq!(texts.first().unwrap().clone(), "house".to_string());

    assert_eq!(
        texts.last().unwrap().clone(),
        "{\"word\":\"a building in which people live; residence for human beings.\",\"link\":\"https://www.dictionary.com/browse/house\",\"speech\":\"noun\"}".to_string()
    )
}

#[test]
fn test_single_embed() {
    #[derive(Embed)]
    struct WordDefinition {
        #[allow(dead_code)]
        id: String,
        #[allow(dead_code)]
        word: String,
        #[embed]
        definition: String,
    }

    let definition = "a building in which people live; residence for human beings.".to_string();

    let fake_definition = WordDefinition {
        id: "doc1".to_string(),
        word: "house".to_string(),
        definition: definition.clone(),
    };

    assert_eq!(
        to_texts(fake_definition).unwrap().first().unwrap().clone(),
        definition
    )
}

#[test]
fn test_multiple_embed_strings() {
    #[derive(Embed)]
    struct Company {
        #[allow(dead_code)]
        id: String,
        #[allow(dead_code)]
        company: String,
        #[embed]
        employee_ages: Vec<i32>,
    }

    let company = Company {
        id: "doc1".to_string(),
        company: "Google".to_string(),
        employee_ages: vec![25, 30, 35, 40],
    };

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

#[test]
fn test_multiple_embed_tags() {
    #[derive(Embed)]
    struct Company {
        #[allow(dead_code)]
        id: String,
        #[embed]
        company: String,
        #[embed]
        employee_ages: Vec<i32>,
    }
    
    let company = Company {
        id: "doc1".to_string(),
        company: "Google".to_string(),
        employee_ages: vec![25, 30, 35, 40],
    };

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
