use crate::client::{CompletionClient, EmbeddingsClient};
use crate::message;
use crate::message::ImageDetail;
use crate::providers::openai::{
    AssistantContent, Function, ImageUrl, Message, ToolCall, ToolType, UserContent,
};
use serde_path_to_error::deserialize;

#[test]
fn test_deserialize_message() {
    let assistant_message_json = r#"
        {
            "role": "assistant",
            "content": "\n\nHello there, how may I assist you today?"
        }
        "#;

    let assistant_message_json2 = r#"
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "\n\nHello there, how may I assist you today?"
                }
            ],
            "tool_calls": null
        }
        "#;

    let assistant_message_json3 = r#"
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_h89ipqYUjEpCPI6SxspMnoUU",
                    "type": "function",
                    "function": {
                        "name": "subtract",
                        "arguments": "{\"x\": 2, \"y\": 5}"
                    }
                }
            ],
            "content": null,
            "refusal": null
        }
        "#;

    let user_message_json = r#"
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What's in this image?"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
                    }
                },
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data": "...",
                        "format": "mp3"
                    }
                }
            ]
        }
        "#;

    let assistant_message: Message = {
        let jd = &mut serde_json::Deserializer::from_str(assistant_message_json);
        deserialize(jd).unwrap_or_else(|err| {
            panic!(
                "Deserialization error at {} ({}:{}): {}",
                err.path(),
                err.inner().line(),
                err.inner().column(),
                err
            );
        })
    };

    let assistant_message2: Message = {
        let jd = &mut serde_json::Deserializer::from_str(assistant_message_json2);
        deserialize(jd).unwrap_or_else(|err| {
            panic!(
                "Deserialization error at {} ({}:{}): {}",
                err.path(),
                err.inner().line(),
                err.inner().column(),
                err
            );
        })
    };

    let assistant_message3: Message = {
        let jd: &mut serde_json::Deserializer<serde_json::de::StrRead<'_>> =
            &mut serde_json::Deserializer::from_str(assistant_message_json3);
        deserialize(jd).unwrap_or_else(|err| {
            panic!(
                "Deserialization error at {} ({}:{}): {}",
                err.path(),
                err.inner().line(),
                err.inner().column(),
                err
            );
        })
    };

    let user_message: Message = {
        let jd = &mut serde_json::Deserializer::from_str(user_message_json);
        deserialize(jd).unwrap_or_else(|err| {
            panic!(
                "Deserialization error at {} ({}:{}): {}",
                err.path(),
                err.inner().line(),
                err.inner().column(),
                err
            );
        })
    };

    match assistant_message {
        Message::Assistant { content, .. } => {
            assert_eq!(
                content[0],
                AssistantContent::Text {
                    text: "\n\nHello there, how may I assist you today?".to_string()
                }
            );
        }
        _ => panic!("Expected assistant message"),
    }

    match assistant_message2 {
        Message::Assistant {
            content,
            tool_calls,
            ..
        } => {
            assert_eq!(
                content[0],
                AssistantContent::Text {
                    text: "\n\nHello there, how may I assist you today?".to_string()
                }
            );

            assert_eq!(tool_calls, vec![]);
        }
        _ => panic!("Expected assistant message"),
    }

    match assistant_message3 {
        Message::Assistant {
            content,
            tool_calls,
            refusal,
            ..
        } => {
            assert!(content.is_empty());
            assert!(refusal.is_none());
            assert_eq!(
                tool_calls[0],
                ToolCall {
                    id: "call_h89ipqYUjEpCPI6SxspMnoUU".to_string(),
                    r#type: ToolType::Function,
                    function: Function {
                        name: "subtract".to_string(),
                        arguments: serde_json::json!({"x": 2, "y": 5}),
                    },
                }
            );
        }
        _ => panic!("Expected assistant message"),
    }

    match user_message {
        Message::User { content, .. } => {
            let (first, second) = {
                let mut iter = content.into_iter();
                (iter.next().unwrap(), iter.next().unwrap())
            };
            assert_eq!(
                first,
                UserContent::Text {
                    text: "What's in this image?".to_string()
                }
            );
            assert_eq!(second, UserContent::Image { image_url: ImageUrl { url: "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg".to_string(), detail: None } });
        }
        _ => panic!("Expected user message"),
    }
}

#[test]
fn test_message_to_message_conversion() {
    let user_message = message::Message::User {
        content: vec![message::UserContent::text("Hello")],
    };

    let assistant_message = message::Message::Assistant {
        id: None,
        content: vec![message::AssistantContent::text("Hi there!")],
    };

    let converted_user_message: Vec<Message> = user_message.clone().try_into().unwrap();
    let converted_assistant_message: Vec<Message> = assistant_message.clone().try_into().unwrap();

    match converted_user_message[0].clone() {
        Message::User { content, .. } => {
            assert_eq!(
                content.first(),
                Some(&UserContent::Text {
                    text: "Hello".to_string()
                })
            );
        }
        _ => panic!("Expected user message"),
    }

    match converted_assistant_message[0].clone() {
        Message::Assistant { content, .. } => {
            assert_eq!(
                content[0].clone(),
                AssistantContent::Text {
                    text: "Hi there!".to_string()
                }
            );
        }
        _ => panic!("Expected assistant message"),
    }

    let original_user_message: message::Message =
        converted_user_message[0].clone().try_into().unwrap();
    let original_assistant_message: message::Message =
        converted_assistant_message[0].clone().try_into().unwrap();

    assert_eq!(original_user_message, user_message);
    assert_eq!(original_assistant_message, assistant_message);
}

#[test]
fn test_message_from_message_conversion() {
    let user_message = Message::User {
        content: vec![UserContent::Text {
            text: "Hello".to_string(),
        }],
        name: None,
    };

    let assistant_message = Message::Assistant {
        content: vec![AssistantContent::Text {
            text: "Hi there!".to_string(),
        }],
        reasoning: None,
        refusal: None,
        audio: None,
        name: None,
        tool_calls: vec![],
        reasoning_details: vec![],
        images: vec![],
    };

    let converted_user_message: message::Message = user_message.clone().try_into().unwrap();
    let converted_assistant_message: message::Message =
        assistant_message.clone().try_into().unwrap();

    match converted_user_message.clone() {
        message::Message::User { content } => {
            assert_eq!(content.first(), Some(&message::UserContent::text("Hello")));
        }
        _ => panic!("Expected user message"),
    }

    match converted_assistant_message.clone() {
        message::Message::Assistant { content, .. } => {
            assert_eq!(
                content.first(),
                Some(&message::AssistantContent::text("Hi there!"))
            );
        }
        _ => panic!("Expected assistant message"),
    }

    let original_user_message: Vec<Message> = converted_user_message.try_into().unwrap();
    let original_assistant_message: Vec<Message> = converted_assistant_message.try_into().unwrap();

    assert_eq!(original_user_message[0], user_message);
    assert_eq!(original_assistant_message[0], assistant_message);
}

#[test]
fn test_user_message_single_text_serializes_as_string() {
    let user_message = Message::User {
        content: vec![UserContent::Text {
            text: "Hello world".to_string(),
        }],
        name: None,
    };

    let serialized = serde_json::to_value(&user_message).unwrap();

    assert_eq!(serialized["role"], "user");
    assert_eq!(serialized["content"], "Hello world");
}

#[test]
fn test_user_message_multiple_parts_serializes_as_array() {
    let user_message = Message::User {
        content: vec![
            UserContent::Text {
                text: "What's in this image?".to_string(),
            },
            UserContent::Image {
                image_url: ImageUrl {
                    url: "https://example.com/image.jpg".to_string(),
                    detail: Some(ImageDetail::default()),
                },
            },
        ],
        name: None,
    };

    let serialized = serde_json::to_value(&user_message).unwrap();

    assert_eq!(serialized["role"], "user");
    assert!(serialized["content"].is_array());
    assert_eq!(serialized["content"].as_array().unwrap().len(), 2);
}

#[test]
fn test_user_message_single_image_serializes_as_array() {
    let user_message = Message::User {
        content: vec![UserContent::Image {
            image_url: ImageUrl {
                url: "https://example.com/image.jpg".to_string(),
                detail: Some(ImageDetail::default()),
            },
        }],
        name: None,
    };

    let serialized = serde_json::to_value(&user_message).unwrap();

    assert_eq!(serialized["role"], "user");
    // Single non-text content should still serialize as array
    assert!(serialized["content"].is_array());
}
#[test]
fn test_client_initialization() {
    let _client = crate::providers::openai::Client::new_with(
        "dummy-key",
        crate::test_utils::RecordingHttpClient::new(""),
    )
    .expect("Client::new() failed");
    let _client_from_builder = crate::providers::openai::Client::builder()
        .api_key("dummy-key")
        .http_client(crate::test_utils::RecordingHttpClient::new(""))
        .build()
        .expect("Client::builder() failed");
}

#[test]
fn test_legacy_chat_completion_model_type_annotation_still_compiles() {
    let client = crate::providers::openai::Client::new_with(
        "dummy-key",
        crate::test_utils::RecordingHttpClient::new(""),
    )
    .expect("Client::new() failed")
    .completions_api();

    let _model: crate::providers::openai::completion::CompletionModel<
        crate::test_utils::RecordingHttpClient,
    > = client.completion_model("gpt-4o");
}

#[test]
fn test_legacy_embedding_model_type_annotation_still_compiles() {
    let client = crate::providers::openai::Client::new_with(
        "dummy-key",
        crate::test_utils::RecordingHttpClient::new(""),
    )
    .expect("Client::new() failed");

    let _model: crate::providers::openai::EmbeddingModel<crate::test_utils::RecordingHttpClient> =
        client.embedding_model(crate::providers::openai::TEXT_EMBEDDING_3_SMALL);
}

#[test]
fn api_switch_preserves_non_completion_capabilities() {
    use crate::client::ModelListingClient;
    use crate::client::transcription::TranscriptionClient;

    let client = crate::providers::openai::Client::new_with(
        "dummy-key",
        crate::test_utils::RecordingHttpClient::new(""),
    )
    .expect("Client::new() failed")
    .completions_api();

    let _: crate::providers::openai::GenericEmbeddingModel<
        crate::providers::openai::OpenAICompletions,
        crate::test_utils::RecordingHttpClient,
    > = client.embedding_model(crate::providers::openai::TEXT_EMBEDDING_3_SMALL);
    let _: crate::providers::openai::CompletionsTranscriptionModel<_> =
        client.transcription_model(crate::providers::openai::WHISPER_1);

    fn assert_model_listing<T: ModelListingClient>(_: &T) {}
    assert_model_listing(&client);

    #[cfg(feature = "image")]
    {
        use crate::client::image_generation::ImageGenerationClient;
        let _: crate::providers::openai::CompletionsImageGenerationModel<_> =
            client.image_generation_model(crate::providers::openai::DALL_E_3);
    }

    #[cfg(feature = "audio")]
    {
        use crate::client::audio_generation::AudioGenerationClient;
        let _: crate::providers::openai::audio_generation::CompletionsAudioGenerationModel<_> =
            client.audio_generation_model(crate::providers::openai::TTS_1);
    }
}
