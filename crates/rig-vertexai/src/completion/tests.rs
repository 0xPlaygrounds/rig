use super::*;
use google_cloud_gax::error::{
    Error,
    rpc::{Code, Status},
};
use rig_core::error::ErrorKind;

/// A service error from the SDK carries its RPC code: the code is the
/// provider's own, the reply has no HTTP status, and only the codes that
/// say the server was unreachable, overloaded or cut the call short retry.
#[test]
fn rpc_codes_classify_retryability_and_keep_the_code() {
    let cells = [
        (Code::Unavailable, true),
        (Code::ResourceExhausted, true),
        (Code::DeadlineExceeded, true),
        (Code::Aborted, true),
        (Code::InvalidArgument, false),
        (Code::NotFound, false),
        (Code::PermissionDenied, false),
        (Code::Unauthenticated, false),
        (Code::FailedPrecondition, false),
        (Code::Internal, false),
        (Code::Unknown, false),
    ];
    for (code, retryable) in cells {
        let name = code.name();
        let error = Error::service(Status::default().set_code(code).set_message("boom"));
        let err = rpc_error(&error);
        assert_eq!(err.is_retryable(), retryable, "{name}");
        assert_eq!(err.provider_response_status(), None, "{name}");
        assert_eq!(
            err.provider_response_body(),
            Some(error.to_string().as_str()),
            "the provider's text is kept verbatim"
        );
        let report = err.report();
        assert_eq!(report.kind, ErrorKind::ProviderResponse, "{name}");
        assert_eq!(report.retryable, retryable, "{name}");
        assert_eq!(report.code.as_deref(), Some(name), "{name}");
    }
}

/// A response-less failure the SDK classifies as its own (an I/O error on
/// the connection) has no code and no status, and the same call may be
/// retried: the request never reached a decision, as on every HTTP wire.
#[test]
fn an_sdk_transport_failure_has_no_code_and_retries() {
    let error = Error::io(std::io::Error::other("connection reset"));
    let err = rpc_error(&error);
    assert!(err.is_retryable());
    assert_eq!(err.provider_response_status(), None);
    let report = err.report();
    assert_eq!(report.code, None);
    assert_eq!(
        report.provider_response.expect("kept").transient,
        Some(true)
    );
}

/// The history [`a_rebuilt_call_for_gemini_3_carries_the_placeholder_signature`]
/// and [`the_request_model_override_reaches_the_rpc`] send: a call with no
/// provider item, answered.
fn rebuilt_call_history() -> CompletionRequest {
    use rig_core::message::{
        AssistantContent, AssistantMessage, CallId, Message, ToolCall, ToolFunction, ToolName,
        ToolResultContent, UserContent,
    };
    let call = ToolCall::new(
        CallId::from_wire("call_1"),
        ToolFunction::new(
            ToolName::new("lookup").expect("a tool name"),
            serde_json::json!({"q": "rig"}),
        ),
    );
    let mut request = CompletionRequest::new("next");
    request.chat_history = vec![
        Message::user("q"),
        Message::Assistant(AssistantMessage::new(vec![AssistantContent::ToolCall(
            call.clone(),
        )])),
        Message::User {
            content: vec![UserContent::ToolResult(
                call.result(vec![ToolResultContent::text("ok")]),
            )],
        },
    ];
    request
}

/// #2026: Gemini 3 rejects a function call it did not sign. A call rebuilt
/// from its canonical fields carries Google's placeholder signature, as the
/// bytes its URL-safe base64 spells, which the SDK sends in standard base64.
#[test]
fn a_rebuilt_call_for_gemini_3_carries_the_placeholder_signature() {
    let request = GenerateContent::new("gemini-3-flash-preview")
        .encode(rebuilt_call_history(), Mode::Unary)
        .expect("the history encodes");
    let call = request
        .contents
        .iter()
        .flat_map(|content| &content.parts)
        .find(|part| part.function_call().is_some())
        .expect("the call is sent");
    assert_eq!(
        call.thought_signature.to_vec(),
        URL_SAFE
            .decode("skip_thought_signature_validator")
            .expect("URL-safe base64")
    );
    // The SDK does not model the call id; it keeps and sends it.
    let call = serde_json::to_value(call.function_call().expect("a call")).expect("JSON");
    assert_eq!(call["id"], "call_1", "Gemini 3 takes call ids");

    let request = GenerateContent::new(GEMINI_2_5_FLASH)
        .encode(rebuilt_call_history(), Mode::Unary)
        .expect("the history encodes");
    assert!(
        request
            .contents
            .iter()
            .flat_map(|content| &content.parts)
            .all(|part| part.thought_signature.is_empty()),
        "Gemini 2 validates no signatures"
    );
}

/// The model a request names is the model the RPC addresses.
#[test]
fn the_request_model_override_reaches_the_rpc() {
    let request = GenerateContent::new(GEMINI_2_5_FLASH)
        .encode(
            rebuilt_call_history().model("gemini-3-pro-preview"),
            Mode::Unary,
        )
        .expect("the request encodes");
    assert_eq!(request.model, "gemini-3-pro-preview");
}
