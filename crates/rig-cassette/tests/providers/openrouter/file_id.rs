//! OpenRouter's refusal of provider file IDs, and its `file_data` pass,
//! asserted on the bytes the `OPENROUTER` dialect actually sends.
//!
//! The refusal used to be raised by OpenRouter's own message conversion; the
//! conversion is now the shared OpenAI chat wire's `encode`, and the refusal a
//! quirk of the `OPENROUTER` dialect. The contract is unchanged and is what
//! these cells pin: a document addressed by provider file id is refused
//! locally, with a message naming OpenRouter, instead of being forwarded to
//! the gateway — while a document carrying inline `file_data` goes out as
//! OpenRouter's `file` content part with no `file_id` beside it.

use rig::completion::CompletionRequestBuilder;
use rig::error::ProviderError;
use rig::message::{
    Document, DocumentMediaType, DocumentSourceKind, Message, UserContent as RigUserContent,
};
use rig::providers::openai::wire::{OPENROUTER, OpenAI};
use rig::wire::{Body, Encoded, Mode, Wire};
use serde_json::Value;

const MODEL: &str = "openai/gpt-4o-mini";

/// The chat body the `OPENROUTER` dialect encodes for one user message, or the
/// error it refuses with.
///
/// The key is a real credential-free config: `encode` never touches a socket,
/// so the bytes are reachable with no cassette and no network.
fn encoded_body(message: Message) -> Result<Value, ProviderError> {
    let encoded = OpenAI::with_key(&OPENROUTER, "k").chat(MODEL).encode(
        CompletionRequestBuilder::unbound(message).build(),
        Mode::Unary,
    )?;
    Ok(sole_body(encoded))
}

/// The one serialized chat body `encoded` carries, as JSON. Separate from
/// [`encoded_body`] because these are test invariants rather than encode
/// failures: a second request or a multipart body is a bug in the wire, not an
/// error the refusal cells may accept as their expected `Err`.
fn sole_body(encoded: Encoded) -> Value {
    assert_eq!(
        encoded.requests.len(),
        1,
        "the chat wire sends exactly one request"
    );
    let Body::Bytes(bytes) = encoded.requests[0].body() else {
        panic!("the chat wire sends a serialized body, not a multipart form")
    };
    serde_json::from_slice(bytes).expect("the chat body is JSON")
}

#[test]
fn generic_document_file_id_fails_openrouter_message_conversion() {
    let message = Message::User {
        content: vec![RigUserContent::Document(Document {
            data: DocumentSourceKind::file_id("file_abc"),
            media_type: None,
            additional_params: None,
        })],
    };

    let result = encoded_body(message);

    assert!(result.is_err());
    let error = result.unwrap_err().to_string();
    assert!(
        error.contains("Provider file IDs are not supported for OpenRouter document inputs"),
        "unexpected error: {error}"
    );
}

/// A base64 PDF is the one document shape OpenRouter accepts: it goes out as
/// the `file` content part, with no `file_id` beside it for the dialect to
/// refuse.
#[test]
fn file_data_document_encodes_as_an_openrouter_file_part() {
    let message = Message::User {
        content: vec![RigUserContent::Document(Document {
            data: DocumentSourceKind::Base64("AAAA".to_string()),
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
