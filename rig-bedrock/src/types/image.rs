use aws_sdk_bedrockruntime::types as aws_bedrock;

use rig::{
    completion::CompletionError,
    message::{ContentFormat, Image, ImageMediaType, MimeType},
};

use base64::{prelude::BASE64_STANDARD, Engine};

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
        }?;

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
            media_type: Some(media_type),
            detail: None,
        }))
    }
}

#[cfg(test)]
mod tests {
    use aws_sdk_bedrockruntime::types as aws_bedrock;
    use base64::{prelude::BASE64_STANDARD, Engine};
    use rig::{
        completion::CompletionError,
        message::{ContentFormat, Image, ImageMediaType},
    };

    use crate::types::image::RigImage;

    #[test]
    fn test_image_to_aws_image() {
        let rig_image = RigImage(Image {
            data: BASE64_STANDARD.encode("img_data"),
            format: Some(ContentFormat::Base64),
            media_type: Some(ImageMediaType::JPEG),
            detail: None,
        });
        let aws_image: Result<aws_bedrock::ImageBlock, _> = rig_image.clone().try_into();
        assert_eq!(aws_image.is_ok(), true);
        let aws_image = aws_image.unwrap();
        assert_eq!(aws_image.format, aws_bedrock::ImageFormat::Jpeg);
        let img_data = BASE64_STANDARD.decode(rig_image.0.data).unwrap();
        let aws_image_bytes = aws_image
            .source()
            .unwrap()
            .as_bytes()
            .unwrap()
            .as_ref()
            .to_owned();
        assert_eq!(aws_image_bytes, img_data)
    }

    #[test]
    fn test_unsupported_image_to_aws_image() {
        let rig_image = RigImage(Image {
            data: BASE64_STANDARD.encode("img_data"),
            format: Some(ContentFormat::Base64),
            media_type: Some(ImageMediaType::HEIC),
            detail: None,
        });
        let aws_image: Result<aws_bedrock::ImageBlock, _> = rig_image.clone().try_into();
        assert_eq!(
            aws_image.err().unwrap().to_string(),
            CompletionError::ProviderError("Unsupported format image/heic".into()).to_string()
        )
    }
}
