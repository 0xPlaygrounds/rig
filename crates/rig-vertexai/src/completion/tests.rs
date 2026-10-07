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
/// The encoded `contents` of a history whose one tool result holds `images`
/// copies of one image and nothing else.
fn image_tool_result_contents(images: usize) -> serde_json::Value {
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
                call.result(vec![ToolResultContent::Image(image); images]),
            )],
        },
    ];
    let wire = GenerateContent::new("gemini-3-flash-preview");
    let request = rig_core::operation::Completion::prepare(request, &wire.describe())
        .expect("the request prepares");
    let request = wire
        .encode(request, Mode::Unary)
        .expect("the history encodes");
    serde_json::to_value(&request.contents).expect("JSON")
}

#[test]
fn an_image_only_tool_result_keeps_a_response_naming_the_image() {
    let contents = image_tool_result_contents(1);
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

#[test]
fn a_tool_result_of_several_images_names_each_in_order() {
    let contents = image_tool_result_contents(2);
    let response = &contents[2]["parts"][0]["functionResponse"];
    assert_eq!(
        response["response"],
        serde_json::json!({"output": [
            {"$ref": "rig_tool_result_image_0"},
            {"$ref": "rig_tool_result_image_1"},
        ]}),
        "{contents:#}"
    );
    assert_eq!(
        response["parts"][1]["inlineData"]["displayName"], "rig_tool_result_image_1",
        "{contents:#}"
    );
}

/// Vertex AI takes the GenerateContent cells, but its tiers: the standard
/// one is its default, and the others need headers its SDK cannot set. The
/// cells reach the SDK request, a `generationConfig` merged over the typed
/// fields.
#[test]
fn options_take_the_generate_content_cells_with_vertex_tiers() {
    use rig_core::completion::ReplayTarget as _;
    use rig_core::completion::{CompletionRequest, Effort, ServiceTier, options::Mapping};
    use rig_core::wire::{Mode, Wire as _};

    let wire = GenerateContent::new("gemini-3-flash-preview");
    let request = CompletionRequest::new("hi")
        .max_tokens(16)
        .reasoning(Effort::High)
        .top_p(0.5)
        .seed(7);
    let encoded = wire
        .encode(request, Mode::Unary)
        .expect("the request encodes");
    let config = encoded
        .generation_config
        .expect("the request has a generation config");
    assert_eq!(config.top_p, Some(0.5));
    assert_eq!(config.seed, Some(7));
    assert_eq!(config.max_output_tokens, Some(16));
    assert!(config.thinking_config.is_some());
    assert_eq!(encoded.model, "gemini-3-flash-preview");

    let tier = |tier| {
        let request = CompletionRequest::new("hi").service_tier(tier);
        wire.map_options(&request, request.options.fields())
            .service_tier
    };
    assert!(matches!(tier(ServiceTier::Default), Mapping::Omit(_)));
    assert!(matches!(tier(ServiceTier::Flex), Mapping::Unsupported(_)));
    assert!(matches!(
        tier(ServiceTier::Priority),
        Mapping::Unsupported(_)
    ));
}

/// A raw `generationConfig: null` is absent, as it was before the merge:
/// the typed fields and the mapped thinking still reach the SDK request.
#[test]
fn a_raw_null_generation_config_keeps_the_typed_fields() {
    use rig_core::completion::{CompletionRequest, Effort};
    use rig_core::wire::{Mode, Wire as _};

    let request = CompletionRequest::new("hi")
        .temperature(0.2)
        .max_tokens(64)
        .reasoning(Effort::High)
        .additional_params(serde_json::json!({"generationConfig": null}));
    let encoded = GenerateContent::new("gemini-3-flash-preview")
        .encode(request, Mode::Unary)
        .expect("the request encodes");
    let config = encoded
        .generation_config
        .expect("the request has a generation config");
    assert_eq!(config.temperature, Some(0.2));
    assert_eq!(config.max_output_tokens, Some(64));
    assert!(config.thinking_config.is_some());
}
