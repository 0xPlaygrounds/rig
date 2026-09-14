use aws_sdk_bedrockruntime::types as aws_bedrock;
use base64::{Engine, prelude::BASE64_STANDARD};
use rig_core::{
    completion::CompletionError,
    message::{DocumentSourceKind, Image, ImageMediaType},
};

use crate::types::image::RigImage;

#[test]
fn test_image_to_aws_image() {
    let encoded_str = BASE64_STANDARD.encode("img_data");
    let rig_image = RigImage(Image {
        data: DocumentSourceKind::Base64(encoded_str),
        media_type: Some(ImageMediaType::JPEG),
        detail: None,
        additional_params: None,
    });
    let aws_image: Result<aws_bedrock::ImageBlock, _> = rig_image.clone().try_into();
    assert!(aws_image.is_ok());
    let aws_image = aws_image.unwrap();
    assert_eq!(aws_image.format, aws_bedrock::ImageFormat::Jpeg);
    let DocumentSourceKind::Base64(data) = rig_image.0.data else {
        panic!("This shouldn't fail since AWS Bedrock only supports base64 encoded strings!")
    };
    let img_data = BASE64_STANDARD.decode(data).unwrap();
    let aws_image_bytes = aws_image
        .source()
        .unwrap()
        .as_bytes()
        .unwrap()
        .as_ref()
        .to_owned();
    assert_eq!(aws_image_bytes, img_data);
}

#[test]
fn test_unsupported_image_to_aws_image() {
    let encoded_str = BASE64_STANDARD.encode("img_data");
    let rig_image = RigImage(Image {
        data: DocumentSourceKind::Base64(encoded_str),
        media_type: Some(ImageMediaType::HEIC),
        detail: None,
        additional_params: None,
    });
    let aws_image: Result<aws_bedrock::ImageBlock, _> = rig_image.try_into();
    assert_eq!(
        aws_image.err().unwrap().to_string(),
        CompletionError::ProviderError("Unsupported format image/heic".into()).to_string()
    );
}
