//! The source of a Converse image, document or video: inline bytes, or an
//! object in Amazon S3 named by an `s3://` URL.

use aws_sdk_bedrockruntime::types as aws_bedrock;
use aws_smithy_types::Blob;
use base64::Engine as _;
use base64::alphabet::STANDARD;
use base64::engine::{DecodePaddingMode, GeneralPurpose, GeneralPurposeConfig};
use rig_core::error::ProviderError;
use rig_core::message::DocumentSourceKind;

/// Standard base64, padded or not.
const BASE64: GeneralPurpose = GeneralPurpose::new(
    &STANDARD,
    GeneralPurposeConfig::new().with_decode_padding_mode(DecodePaddingMode::Indifferent),
);

/// Where a media part's content is.
pub(crate) enum Source {
    Bytes(Blob),
    Stored(aws_bedrock::S3Location),
}

/// Whether `data` names an S3 object.
pub(crate) fn is_stored(data: &DocumentSourceKind) -> bool {
    matches!(data, DocumentSourceKind::Url(url) if url.starts_with("s3://"))
}

/// The source of `data`: base64 or raw bytes, or an S3 URL. Any other
/// form is an error.
pub(crate) fn of(data: DocumentSourceKind) -> Result<Source, ProviderError> {
    match data {
        DocumentSourceKind::Base64(data) => BASE64
            .decode(data)
            .map(|bytes| Source::Bytes(Blob::new(bytes)))
            .map_err(ProviderError::request),
        DocumentSourceKind::Raw(bytes) => Ok(Source::Bytes(Blob::new(bytes))),
        DocumentSourceKind::Url(uri) if uri.starts_with("s3://") => {
            aws_bedrock::S3Location::builder()
                .uri(uri)
                .build()
                .map(Source::Stored)
                .map_err(ProviderError::request)
        }
        data => Err(ProviderError::request(format!(
            "Converse takes base64 data or an s3:// URL, not {data}"
        ))),
    }
}
