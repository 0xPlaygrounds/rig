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

/// Vertex AI requires `FunctionResponse.response`. A result that is only an
/// image names it by `$ref`, the image carrying that `displayName`.
#[test]
fn an_image_only_tool_result_keeps_a_response_naming_the_image() {
    use rig_core::message::{
        AssistantContent, AssistantMessage, CallId, DocumentSourceKind, Image, ImageMediaType,
        Message, ToolCall, ToolFunction, ToolName, ToolResultContent, UserContent,
    };
    use rig_core::wire::Operation;
    let call = ToolCall::new(
        CallId::from_wire("call_1"),
        ToolFunction::new(
            ToolName::new("shot").expect("a tool name"),
            serde_json::json!({}),
        ),
    );
    let image = Image {
        data: DocumentSourceKind::Base64("iVBORw0KGgo=".to_owned()),
        media_type: Some(ImageMediaType::PNG),
        ..Image::default()
    };
    let mut request = CompletionRequest::new("next");
    request.tools = vec![rig_core::completion::ToolDefinition {
        name: ToolName::new("shot").expect("a tool name"),
        description: "a screenshot".to_owned(),
        parameters: serde_json::json!({"type": "object"}),
    }];
    request.chat_history = vec![
        Message::user("shoot"),
        Message::Assistant(AssistantMessage::new(vec![AssistantContent::ToolCall(
            call.clone(),
        )])),
        Message::User {
            content: vec![UserContent::ToolResult(
                call.result(vec![ToolResultContent::Image(image)]),
            )],
        },
    ];
    let wire = GenerateContent::new("gemini-3-flash-preview");
    let request = rig_core::operation::Completion::prepare(request, &wire.describe())
        .expect("the request prepares");
    let request = wire
        .encode(request, Mode::Unary)
        .expect("the history encodes");
    let contents = serde_json::to_value(&request.contents).expect("JSON");
    let response = &contents[2]["parts"][0]["functionResponse"];
    assert_eq!(
        response["response"],
        serde_json::json!({"output": {"$ref": "rig_tool_result_image_0"}}),
        "{contents:#}"
    );
    assert_eq!(
        response["parts"][0]["inlineData"]["displayName"], "rig_tool_result_image_0",
        "{contents:#}"
    );
}
