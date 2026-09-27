//! Each rig-core error converts with the kind, retry verdict, message and
//! source chain written out here.

use std::env::VarError;

use super::*;
use crate::tool::ToolErrorKind;

/// `error`'s classification, as the fields a caller routes on.
fn assert_converts(
    error: RigError,
    kind: ErrorKind,
    retryable: bool,
    message: &str,
    source_chain: &[&str],
) {
    assert_eq!(error.kind, kind, "{error:?}");
    assert_eq!(error.retryable, retryable, "{error:?}");
    assert_eq!(error.message, message, "{error:?}");
    assert_eq!(error.source_chain, source_chain, "{error:?}");
}

#[test]
fn other_takes_the_display_and_the_sources() {
    let io = std::io::Error::other("disk full");
    assert_converts(
        RigError::other(EnvError::Variable {
            name: "APP_TOKEN",
            source: VarError::NotPresent,
        }),
        ErrorKind::Other,
        false,
        "environment variable `APP_TOKEN` is not set or is invalid",
        &["environment variable not found"],
    );
    assert_converts(
        RigError::other(io),
        ErrorKind::Other,
        false,
        "disk full",
        &[],
    );
}

#[test]
fn an_environment_error_is_configuration() {
    assert_converts(
        EnvError::Variable {
            name: "OPENAI_API_KEY",
            source: VarError::NotPresent,
        }
        .into(),
        ErrorKind::Configuration,
        false,
        "environment variable `OPENAI_API_KEY` is not set or is invalid",
        &["environment variable not found"],
    );
    assert_converts(
        EnvError::Invalid {
            name: "OPENAI_BASE_URL",
            detail: "not a URL".to_owned(),
        }
        .into(),
        ErrorKind::Configuration,
        false,
        "environment variable `OPENAI_BASE_URL` is invalid: not a URL",
        &[],
    );
}

#[test]
fn a_sign_in_failure_is_classified_by_what_failed() {
    assert_converts(
        AuthError::Message("the device code expired".to_owned()).into(),
        ErrorKind::Other,
        false,
        "the device code expired",
        &[],
    );
    assert_converts(
        AuthError::Io(std::io::Error::other("read-only file system")).into(),
        ErrorKind::Other,
        false,
        "read-only file system",
        &[],
    );
    let json = serde_json::from_str::<serde_json::Value>("{").expect_err("truncated JSON");
    assert_converts(
        AuthError::Json(json).into(),
        ErrorKind::Json,
        false,
        "EOF while parsing an object at line 1 column 1",
        &[],
    );
    let refused: RigError = AuthError::Http(http_client::Error::non_success_with_details(
        http::StatusCode::UNAUTHORIZED,
        http::HeaderMap::new(),
        "bad_verification_code".to_owned(),
    ))
    .into();
    assert_converts(
        refused.clone(),
        ErrorKind::ProviderResponse,
        false,
        "ProviderResponseError: status 401 Unauthorized: bad_verification_code",
        &[],
    );
    assert_eq!(refused.http_status, Some(401));
    assert_eq!(
        refused.provider_response_body(),
        Some("bad_verification_code")
    );
}

#[test]
fn a_reference_or_selection_that_does_not_resolve_is_configuration() {
    assert_converts(
        RefError::NoModel {
            reference: "openai".to_owned(),
        }
        .into(),
        ErrorKind::Configuration,
        false,
        "`openai` names no model: expected `vendor[/format]:model`",
        &[],
    );
    assert_converts(
        RefError::EmptyModel.into(),
        ErrorKind::Configuration,
        false,
        "model identifier must not be empty",
        &[],
    );
    let unknown = SelectionError::Unknown {
        vendor: "nope".to_owned(),
    };
    assert_converts(
        RefError::Selection(unknown.clone()).into(),
        ErrorKind::Configuration,
        false,
        "no registered provider is named `nope`",
        &[],
    );
    assert_converts(
        unknown.into(),
        ErrorKind::Configuration,
        false,
        "no registered provider is named `nope`",
        &[],
    );
}

#[test]
fn an_unparseable_id_is_configuration() {
    let error = "0"
        .parse::<crate::id::RunId>()
        .expect_err("zero is not an id");
    assert_converts(
        error.into(),
        ErrorKind::Configuration,
        false,
        "invalid id: expected a non-zero integer, got \"0\"",
        &[],
    );
}

#[test]
fn a_request_that_cannot_be_built_reports_as_a_wire_does() {
    assert_converts(
        MessageError::ConversionError("no audio on this wire".to_owned()).into(),
        ErrorKind::Request,
        false,
        "RequestError: Message conversion error: no audio on this wire",
        &["Message conversion error: no audio on this wire"],
    );
    assert_converts(
        EncodeError::request("no base URL").into(),
        ErrorKind::Request,
        false,
        "RequestError: no base URL",
        &["no base URL"],
    );
    assert_converts(
        TranscriptError::ConsecutiveAssistant { index: 1 }.into(),
        ErrorKind::Request,
        false,
        "consecutive assistant messages at index 1",
        &[],
    );
    assert_converts(
        FilterError::MissingField("year".to_owned()).into(),
        ErrorKind::Request,
        false,
        "Filter error: Missing field 'year'",
        &["Missing field 'year'"],
    );
}

#[test]
fn a_transport_error_reports_as_the_driver_does() {
    let reset: Box<dyn std::error::Error + Send + Sync> =
        Box::new(std::io::Error::other("connection reset"));
    assert_converts(
        http_client::Error::Instance(reset).into(),
        ErrorKind::Http,
        true,
        "HttpError: Http client error: connection reset",
        &[],
    );
    let limited: RigError = http_client::Error::non_success_with_details(
        http::StatusCode::TOO_MANY_REQUESTS,
        http::HeaderMap::new(),
        "slow down".to_owned(),
    )
    .into();
    assert_converts(
        limited.clone(),
        ErrorKind::ProviderResponse,
        true,
        "ProviderResponseError: status 429 Too Many Requests: slow down",
        &[],
    );
    assert_eq!(limited.http_status, Some(429));
}

#[test]
fn a_tool_context_failure_reports_as_the_tool_does() {
    assert_converts(
        ToolContextError::Missing("user").into(),
        ErrorKind::Tool(ToolErrorKind::Other),
        false,
        "required tool context value `user` was not found",
        &["required tool context value `user` was not found"],
    );
}

#[test]
fn an_extraction_or_loading_failure_is_other() {
    assert_converts(
        EmbedError::new(std::io::Error::other("unreadable")).into(),
        ErrorKind::Other,
        false,
        "unreadable",
        &["unreadable"],
    );
    assert_converts(
        FileLoaderError::InvalidGlobPattern("[".to_owned()).into(),
        ErrorKind::Other,
        false,
        "Invalid glob pattern: [",
        &[],
    );
}

#[cfg(feature = "pdf")]
#[test]
fn a_pdf_loader_failure_reports_its_file_error_unchanged() {
    assert_converts(
        crate::loaders::pdf::PdfLoaderError::FileLoaderError(FileLoaderError::InvalidGlobPattern(
            "[".to_owned(),
        ))
        .into(),
        ErrorKind::Other,
        false,
        "Invalid glob pattern: [",
        &[],
    );
    let utf8 = String::from_utf8(vec![0xff]).expect_err("not UTF-8");
    assert_converts(
        crate::loaders::pdf::PdfLoaderError::FromUtf8Error(utf8).into(),
        ErrorKind::Other,
        false,
        "UTF-8 conversion error: invalid utf-8 sequence of 1 bytes from index 0",
        &["invalid utf-8 sequence of 1 bytes from index 0"],
    );
}

#[cfg(feature = "epub")]
#[test]
fn an_epub_loader_failure_reports_its_file_error_unchanged() {
    assert_converts(
        crate::loaders::epub::EpubLoaderError::FileLoaderError(
            FileLoaderError::InvalidGlobPattern("[".to_owned()),
        )
        .into(),
        ErrorKind::Other,
        false,
        "Invalid glob pattern: [",
        &[],
    );
    let processor: Box<dyn std::error::Error> = Box::new(std::io::Error::other("bad chapter"));
    assert_converts(
        crate::loaders::epub::EpubLoaderError::TextProcessorError(processor).into(),
        ErrorKind::Other,
        false,
        "Text processor error: bad chapter",
        &["bad chapter"],
    );
}
