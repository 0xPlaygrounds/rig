//! OpenRouter's handling of provider file IDs, and its `file_data` pass,
//! asserted on the bytes the `OPENROUTER` dialect actually sends.
//!
//! OpenRouter takes no provider file id, so the adapter leaves a placeholder
//! for a document addressed by one instead of forwarding it to the gateway,
//! while a document carrying inline `file_data` goes out as OpenRouter's
//! `file` content part with no `file_id` beside it.

use rig::completion::CompletionRequest;
use rig::error::ProviderError;
use rig::message::{
    Document, DocumentMediaType, DocumentSourceKind, Message, UserContent as RigUserContent,
};
use rig::providers::openai::wire::Chat;
use rig::providers::openai::wire::{OPENROUTER, OpenAIConfig};
use rig::wire::{Body, Encoded, Mode, Operation, Wire};
use rig_core::operation::Completion;
use serde_json::Value;

const MODEL: &str = "openai/gpt-4o-mini";

/// The chat body the `OPENROUTER` dialect encodes for one user message,
/// prepared as the driver prepares it.
///
/// The key is a real credential-free config: `encode` never touches a socket,
/// so the bytes are reachable with no cassette and no network.
fn encoded_body(message: Message) -> Result<Value, ProviderError> {
    let wire = Chat::new(OpenAIConfig::with_key(&OPENROUTER, "k"), MODEL);
    let request = Completion::prepare(CompletionRequest::new(message), &wire.describe())?;
    let encoded = wire.encode(request, Mode::Unary)?;
    Ok(sole_body(encoded))
}

/// The one serialized chat body `encoded` carries, as JSON. Separate from
/// [`encoded_body`] because this is a test invariant rather than an encode
/// failure: a multipart body is a bug in the wire.
fn sole_body(encoded: Encoded) -> Value {
    let Body::Bytes(bytes) = encoded.request.body() else {
        panic!("the chat wire sends a serialized body, not a multipart form")
    };
    serde_json::from_slice(bytes).expect("the chat body is JSON")
}

#[test]
fn generic_document_file_id_reaches_openrouter_as_a_placeholder() {
    let message = Message::User {
        content: vec![RigUserContent::Document(Document {
            data: DocumentSourceKind::file_id("file_abc").into(),
            media_type: None,
            additional_params: None,
        })],
    };

    let body = encoded_body(message).expect("the adapter leaves nothing the encoder refuses");

    assert_eq!(
        body["messages"][0]["content"],
        rig_core::completion::history::DOCUMENT_UNSENDABLE
    );
    assert!(!body.to_string().contains("file_abc"), "{body}");
}

/// A base64 PDF goes out as OpenRouter's `file` content part, with no
/// `file_id` beside it.
#[test]
fn file_data_document_encodes_as_an_openrouter_file_part() {
    let message = Message::User {
        content: vec![RigUserContent::Document(Document {
            data: DocumentSourceKind::Base64("AAAA".to_string()).into(),
            media_type: Some(DocumentMediaType::PDF),
            additional_params: None,
        })],
    };

    let body = encoded_body(message).expect("a file_data document should encode");
    let json = &body["messages"][0]["content"][0];

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
