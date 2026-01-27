use rig::providers::openai::responses_api::{InputItem, Message, UserContent};
use rig::OneOrMany;

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
