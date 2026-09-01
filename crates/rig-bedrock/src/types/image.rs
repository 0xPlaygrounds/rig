use aws_sdk_bedrockruntime::types as aws_bedrock;

use rig_core::{
    completion::CompletionError,
    message::{DocumentSourceKind, Image, ImageMediaType, MimeType},
};

use base64::{Engine, prelude::BASE64_STANDARD};

use super::converse_output::{ImageBlock, ImageFormat, ImageSource};

#[derive(Clone)]
pub struct RigImage(pub Image);

impl TryFrom<RigImage> for aws_bedrock::ImageBlock {
    type Error = CompletionError;

    fn try_from(image: RigImage) -> Result<Self, Self::Error> {
        let maybe_format: Option<Result<aws_bedrock::ImageFormat, CompletionError>> =
            image.0.media_type.map(|f| match f {
                ImageMediaType::JPEG => Ok(aws_bedrock::ImageFormat::Jpeg),
                ImageMediaType::PNG => Ok(aws_bedrock::ImageFormat::Png),
                ImageMediaType::GIF => Ok(aws_bedrock::ImageFormat::Gif),
                ImageMediaType::WEBP => Ok(aws_bedrock::ImageFormat::Webp),
                e => Err(CompletionError::ProviderError(format!(
                    "Unsupported format {}",
                    e.to_mime_type()
                ))),
            });

        let format = match maybe_format {
            Some(Ok(image_format)) => Ok(Some(image_format)),
            Some(Err(err)) => Err(err),
            None => Ok(None),
        }?;

        let DocumentSourceKind::Base64(data) = image.0.data else {
            return Err(CompletionError::RequestError(
                "Only base64 encoded strings are allowed for image input on AWS Bedrock".into(),
            ));
        };

        let img_data = BASE64_STANDARD
            .decode(data)
            .map_err(|e| CompletionError::ProviderError(e.to_string()))?;
        let blob = aws_smithy_types::Blob::new(img_data);
        let result = aws_bedrock::ImageBlock::builder()
            .set_format(format)
            .source(aws_bedrock::ImageSource::Bytes(blob))
            .build()
            .map_err(|e| CompletionError::ProviderError(e.to_string()))?;
        Ok(result)
    }
}

impl TryFrom<ImageBlock> for RigImage {
    type Error = CompletionError;

    fn try_from(image: ImageBlock) -> Result<Self, Self::Error> {
        let media_type = match image.format {
            ImageFormat::Gif => Ok(ImageMediaType::GIF),
            ImageFormat::Jpeg => Ok(ImageMediaType::JPEG),
            ImageFormat::Png => Ok(ImageMediaType::PNG),
            ImageFormat::Webp => Ok(ImageMediaType::WEBP),
            // The mirror carries the raw wire token for a format the SDK did
            // not recognize, which is what the message quoted before.
            ImageFormat::Unknown(format) => Err(CompletionError::ProviderError(format!(
                "Unsupported format {format}"
            ))),
        }?;

        let data = match image.source {
            Some(ImageSource::Bytes(blob)) => {
                let encoded_img = BASE64_STANDARD.encode(blob.inner);
                Ok(encoded_img)
            }
            _ => Err(CompletionError::ProviderError(
                "Image source is missing".into(),
            )),
        }?;
        Ok(RigImage(Image {
            data: DocumentSourceKind::Base64(data),
            media_type: Some(media_type),
            detail: None,
            additional_params: None,
        }))
    }
}

#[cfg(test)]
mod tests;
