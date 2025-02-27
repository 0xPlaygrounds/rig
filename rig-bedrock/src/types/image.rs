use aws_sdk_bedrockruntime::types as aws_bedrock;

use rig::{
    completion::CompletionError,
    message::{ContentFormat, Image, ImageMediaType, MimeType},
};

use base64::{prelude::BASE64_STANDARD, Engine};

pub struct RigImage(pub Image);

impl TryFrom<RigImage> for aws_bedrock::ImageBlock {
    type Error = CompletionError;

    fn try_from(image: RigImage) -> Result<Self, Self::Error> {
        let format = image
            .0
            .media_type
            .map(|f| match f {
                ImageMediaType::JPEG => Ok(aws_bedrock::ImageFormat::Jpeg),
                ImageMediaType::PNG => Ok(aws_bedrock::ImageFormat::Png),
                ImageMediaType::GIF => Ok(aws_bedrock::ImageFormat::Gif),
                ImageMediaType::WEBP => Ok(aws_bedrock::ImageFormat::Webp),
                e => Err(CompletionError::ProviderError(format!(
                    "Unsupported format {}",
                    e.to_mime_type()
                ))),
            })
            .and_then(|img| img.ok());

        let img_data = BASE64_STANDARD
            .decode(image.0.data)
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

impl TryFrom<aws_bedrock::ImageBlock> for RigImage {
    type Error = CompletionError;

    fn try_from(image: aws_bedrock::ImageBlock) -> Result<Self, Self::Error> {
        let media_type = match image.format {
            aws_bedrock::ImageFormat::Gif => Ok(ImageMediaType::GIF),
            aws_bedrock::ImageFormat::Jpeg => Ok(ImageMediaType::JPEG),
            aws_bedrock::ImageFormat::Png => Ok(ImageMediaType::PNG),
            aws_bedrock::ImageFormat::Webp => Ok(ImageMediaType::WEBP),
            e => Err(CompletionError::ProviderError(format!(
                "Unsupported format {}",
                e
            ))),
        };
        let data = match image.source {
            Some(aws_bedrock::ImageSource::Bytes(blob)) => {
                let encoded_img = BASE64_STANDARD.encode(blob.into_inner());
                Ok(encoded_img)
            }
            _ => Err(CompletionError::ProviderError(
                "Image source is missing".into(),
            )),
        }?;
        Ok(RigImage(Image {
            data,
            format: Some(ContentFormat::Base64),
            media_type: media_type.ok(),
            detail: None,
        }))
    }
}
