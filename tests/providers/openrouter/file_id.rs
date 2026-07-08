use rig::OneOrMany;
use rig::message::{Document, DocumentSourceKind, Message, UserContent as RigUserContent};
use rig::providers::openai::{FileData as OpenAiFileData, UserContent as OpenAiUserContent};
use rig::providers::openrouter::{Message as OpenRouterMessage, messages_from_rig_message};

#[test]
fn generic_document_file_id_fails_openrouter_message_conversion() {
    let message = Message::User {
        content: OneOrMany::one(RigUserContent::Document(Document {
            data: DocumentSourceKind::file_id("file_abc"),
            media_type: None,
            additional_params: None,
        })),
    };

    let result: Result<Vec<OpenRouterMessage>, _> = messages_from_rig_message(message);

    assert!(result.is_err());
    let error = result.unwrap_err().to_string();
    assert!(
        error.contains("Provider file IDs are not supported for OpenRouter document inputs"),
        "unexpected error: {error}"
    );
}

#[test]
fn openai_file_data_converts_to_openrouter_file_data() {
    let openai_content = OpenAiUserContent::File {
        file: OpenAiFileData {
            file_data: Some("data:application/pdf;base64,AAAA".to_string()),
            file_id: Some("file_abc".to_string()),
            filename: Some("document.pdf".to_string()),
        },
    };

    let message = Message::User {
        content: OneOrMany::one(RigUserContent::from(openai_content)),
    };
    let messages = messages_from_rig_message(message).unwrap();
    let json = serde_json::to_value(messages.first().expect("one message")).unwrap();
    let json = &json["content"][0];

    assert_eq!(json["type"], "file");
    assert_eq!(json["file"]["filename"], "document.pdf");
    assert_eq!(
        json["file"]["file_data"],
        "data:application/pdf;base64,AAAA"
    );
    assert!(
        json["file"].get("file_id").is_none(),
        "OpenRouter payload should not include provider file IDs: {json}"
    );
}

#[test]
fn openai_file_id_only_fails_openrouter_user_content_conversion() {
    let openai_content = OpenAiUserContent::File {
        file: OpenAiFileData {
            file_data: None,
            file_id: Some("file_abc".to_string()),
            filename: Some("document.pdf".to_string()),
        },
    };

    let message = Message::User {
        content: OneOrMany::one(RigUserContent::from(openai_content)),
    };
    let result = messages_from_rig_message(message);

    assert!(result.is_err());
    let error = result.unwrap_err().to_string();
    assert!(
        error.contains("Provider file IDs are not supported"),
        "unexpected error: {error}"
    );
}
