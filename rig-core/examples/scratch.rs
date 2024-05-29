use std::collections::HashMap;

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Document {
    text: String,
    #[serde(flatten)]
    additional_prop: HashMap<String, String>,
}

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    let document = Document {
        text: "Hello, world!".to_string(),
        additional_prop: HashMap::from([
            ("key1".to_string(), "value1".to_string()),
            ("key2".to_string(), "value2".to_string()),
        ]),
    };

    println!("{}", serde_json::to_string_pretty(&document)?);

    let document_json = r#"{
        "text": "Hello, world!",
        "key1": "value1",
        "key2": "value2"
    }"#;

    let document: Document = serde_json::from_str(document_json)?;
    println!("{:?}", document);

    Ok(())
}
