use aws_sdk_bedrockruntime::types as aws_bedrock;

use rig_core::error::ProviderError;
use rig_core::message::{Image, ImageMediaType, MimeType};

use super::source::{self, Source};

/// The Converse image block for `image`: PNG, JPEG, GIF or WEBP, as base64
/// data or an S3 object.
pub(crate) fn to_aws(image: Image) -> Result<aws_bedrock::ImageBlock, ProviderError> {
    let format = match image.media_type {
        Some(ImageMediaType::JPEG) => aws_bedrock::ImageFormat::Jpeg,
        Some(ImageMediaType::PNG) => aws_bedrock::ImageFormat::Png,
        Some(ImageMediaType::GIF) => aws_bedrock::ImageFormat::Gif,
        Some(ImageMediaType::WEBP) => aws_bedrock::ImageFormat::Webp,
        Some(other) => {
            return Err(ProviderError::Provider(format!(
                "Unsupported format {}",
                other.to_mime_type()
            )));
        }
        None => return Err(ProviderError::request("Converse needs an image's format")),
    };
    let source = match source::of(image.data)? {
        Source::Bytes(blob) => aws_bedrock::ImageSource::Bytes(blob),
        Source::Stored(location) => aws_bedrock::ImageSource::S3Location(location),
    };
    aws_bedrock::ImageBlock::builder()
        .format(format)
        .source(source)
        .build()
        .map_err(|e| ProviderError::Provider(e.to_string()))
}

#[cfg(test)]
mod tests;
