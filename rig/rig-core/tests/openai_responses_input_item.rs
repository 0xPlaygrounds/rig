use rig::OneOrMany;
use rig::completion::{CompletionError, Message as CompletionMessage};
use rig::message::{AssistantContent, Reasoning, ReasoningContent};
use rig::providers::openai::responses_api::{InputItem, Message, UserContent};

#[test]
fn test_input_item_serialization_avoids_duplicate_role() {
    let message = Message::User {
        content: OneOrMany::one(UserContent::InputText {
            text: "hello".to_string(),
        }),
        name: None,
    };
    let item: InputItem = message.into();
    let json = serde_json::to_string(&item).expect("serialize InputItem");
    let role_count = json.matches("\"role\"").count();

    assert_eq!(
        role_count, 1,
        "InputItem should serialize a single role field, got {role_count}: {json}"
    );
}

#[test]
fn assistant_reasoning_without_id_returns_error() {
    let message = CompletionMessage::Assistant {
        id: Some("assistant_message_id".to_string()),
        content: OneOrMany::one(AssistantContent::Reasoning(Reasoning::new("thought"))),
    };

    let items: Result<Vec<InputItem>, CompletionError> = message.try_into();
    assert!(matches!(
        items,
        Err(CompletionError::ProviderError(message))
            if message.contains("OpenAI-generated ID is required")
    ));
}

#[test]
fn assistant_reasoning_encrypted_only_serializes_encrypted_content() {
    let reasoning = Reasoning::encrypted("encrypted_blob").with_id("rs_1".to_string());
    let message = CompletionMessage::Assistant {
        id: Some("assistant_message_id".to_string()),
        content: OneOrMany::one(AssistantContent::Reasoning(reasoning)),
    };

    let items: Vec<InputItem> = message
        .try_into()
        .expect("assistant reasoning should convert to InputItem");
    assert_eq!(items.len(), 1);

    let item_json = serde_json::to_value(&items[0]).expect("serialize InputItem");
    let item_type = item_json
        .get("type")
        .and_then(|value| value.as_str())
        .expect("reasoning item should include type");
    assert_eq!(item_type, "reasoning");
    assert_eq!(
        item_json.get("id").and_then(|value| value.as_str()),
        Some("rs_1")
    );
    assert_eq!(
        item_json
            .get("encrypted_content")
            .and_then(|value| value.as_str()),
        Some("encrypted_blob")
    );
    assert_eq!(
        item_json
            .get("summary")
            .and_then(|value| value.as_array())
            .map(Vec::len),
        Some(0)
    );
}

#[test]
fn assistant_reasoning_mixed_content_serializes_only_text_like_summaries() {
    let mut reasoning =
        Reasoning::new_with_signature("step-1", Some("sig-1".to_string())).with_id("rs_2".into());
    reasoning
        .content
        .push(ReasoningContent::Summary("summary-2".to_string()));
    reasoning
        .content
        .push(ReasoningContent::Encrypted("ciphertext".to_string()));
    reasoning.content.push(ReasoningContent::Redacted {
        data: "redacted".to_string(),
    });

    let message = CompletionMessage::Assistant {
        id: Some("assistant_message_id".to_string()),
        content: OneOrMany::one(AssistantContent::Reasoning(reasoning)),
    };

    let items: Vec<InputItem> = message
        .try_into()
        .expect("assistant reasoning should convert to InputItem");
    assert_eq!(items.len(), 1);

    let item_json = serde_json::to_value(&items[0]).expect("serialize InputItem");
    let summary = item_json
        .get("summary")
        .and_then(|value| value.as_array())
        .expect("reasoning item should include summary array");
    let summary_texts: Vec<&str> = summary
        .iter()
        .filter_map(|entry| entry.get("text").and_then(|text| text.as_str()))
        .collect();

    assert_eq!(summary_texts, vec!["step-1", "summary-2"]);
    assert_eq!(
        item_json
            .get("encrypted_content")
            .and_then(|value| value.as_str()),
        Some("ciphertext")
    );
}
