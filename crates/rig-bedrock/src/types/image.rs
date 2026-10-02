use aws_sdk_bedrockruntime::types as aws_bedrock;

use rig_core::error::ProviderError;
use rig_core::message::{DocumentSourceKind, Image, ImageMediaType, MimeType};

use base64::{Engine, prelude::BASE64_STANDARD};

/// The Converse image block for `image`, which must be base64 encoded.
pub(crate) fn to_aws(image: Image) -> Result<aws_bedrock::ImageBlock, ProviderError> {
    let maybe_format: Option<Result<aws_bedrock::ImageFormat, ProviderError>> =
        image.media_type.map(|f| match f {
            ImageMediaType::JPEG => Ok(aws_bedrock::ImageFormat::Jpeg),
            ImageMediaType::PNG => Ok(aws_bedrock::ImageFormat::Png),
            ImageMediaType::GIF => Ok(aws_bedrock::ImageFormat::Gif),
            ImageMediaType::WEBP => Ok(aws_bedrock::ImageFormat::Webp),
            e => Err(ProviderError::Provider(format!(
                "Unsupported format {}",
                e.to_mime_type()
            ))),
        });

    let format = match maybe_format {
        Some(Ok(image_format)) => Ok(Some(image_format)),
        Some(Err(err)) => Err(err),
        None => Ok(None),
    }?;

    let DocumentSourceKind::Base64(data) = image.data else {
        return Err(ProviderError::request(
            "Only base64 encoded strings are allowed for image input on AWS Bedrock",
        ));
    };

    let img_data = BASE64_STANDARD
        .decode(data)
        .map_err(|e| ProviderError::Provider(e.to_string()))?;
    let blob = aws_smithy_types::Blob::new(img_data);
    aws_bedrock::ImageBlock::builder()
        .set_format(format)
        .source(aws_bedrock::ImageSource::Bytes(blob))
        .build()
        .map_err(|e| ProviderError::Provider(e.to_string()))
}

#[cfg(test)]
#[cfg(any())]
mod tests;
