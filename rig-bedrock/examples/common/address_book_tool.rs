use rig::{completion::ToolDefinition, tool::Tool};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::HashMap;
use std::{
    error::Error,
    fmt::{Display, Formatter},
};

#[derive(Debug)]
pub struct AddressBookError(String);

impl Display for AddressBookError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "Address Book error {}", self.0)
    }
}

impl Error for AddressBookError {}

#[derive(Serialize, Clone)]
pub struct AddressBook {
    street_name: String,
    city: String,
    state: String,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
#[serde(untagged)]
pub enum AddressBookResult {
    Found(AddressBook),
    NotFound(String),
}

#[derive(Deserialize)]
pub struct AddressBookArgs {
    email: String,
}

#[derive(Deserialize, Serialize)]
pub struct AddressBookTool;
impl Tool for AddressBookTool {
    const NAME: &'static str = "address_book";

    type Error = AddressBookError;
    type Args = AddressBookArgs;
    type Output = AddressBookResult;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: "address_book".to_string(),
            description: "get address by email".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "email": {
                        "type": "string",
                        "description": "email address"
                    },
                }
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let mut address_book: HashMap<String, AddressBook> = HashMap::new();
        address_book.extend(vec![
            (
                "john.doe@example.com".to_string(),
                AddressBook {
                    street_name: "123 Elm St".to_string(),
                    city: "Springfield".to_string(),
                    state: "IL".to_string(),
                },
            ),
            (
                "jane.smith@example.com".to_string(),
                AddressBook {
                    street_name: "456 Oak St".to_string(),
                    city: "Metropolis".to_string(),
                    state: "NY".to_string(),
                },
            ),
            (
                "alice.johnson@example.com".to_string(),
                AddressBook {
                    street_name: "789 Pine St".to_string(),
                    city: "Gotham".to_string(),
                    state: "NJ".to_string(),
                },
            ),
        ]);

        if args.email.starts_with("malice") {
            return Err(AddressBookError("Corrupted database".into()));
        }

        match address_book.get(&args.email) {
            Some(address) => Ok(AddressBookResult::Found(address.clone())),
            None => Ok(AddressBookResult::NotFound("Address not found".into())),
        }
    }
}
