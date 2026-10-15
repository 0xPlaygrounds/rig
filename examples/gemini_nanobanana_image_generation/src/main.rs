//! Generates a 16:9 image with Nano Banana 2, optionally drawing from a
//! reference image, and writes every image and text part of the reply.
//!
//! ```sh
//! cargo run -p gemini_nanobanana_image_generation -- [reference.png]
//! ```

use anyhow::{Context, Result};
use rig::image_generation::ImageGenerationRequestBuilder;
use rig::message::{DocumentSourceKind, Image, ImageMediaType};
use rig::providers::gemini::image_generation::{
    GEMINI_3_1_FLASH_IMAGE, ImageOptions, ImagePart, parts_of,
};
use rig::providers::gemini::{Gemini, ImageSize};

#[tokio::main]
async fn main() -> Result<()> {
    let model = Gemini::from_env()?.image_generation(GEMINI_3_1_FLASH_IMAGE);

    let mut request = ImageGenerationRequestBuilder::new(
        "Write one sentence describing a flat icon of a yellow banana on a white background, \
         then generate the icon.",
    )
    .width(16)
    .height(9)
    .additional_params(
        ImageOptions::new()
            .image_size(ImageSize::OneK)
            .with_text()
            .params(),
    );
    if let Some(path) = std::env::args().nth(1) {
        let bytes = std::fs::read(&path).with_context(|| format!("reading {path}"))?;
        let media_type = if path.ends_with(".jpg") || path.ends_with(".jpeg") {
            ImageMediaType::JPEG
        } else {
            ImageMediaType::PNG
        };
        request = request.image(Image {
            data: DocumentSourceKind::Raw(bytes),
            media_type: Some(media_type),
            ..Image::default()
        });
    }

    let response = model.call(request.build()).await?;
    for (index, part) in parts_of(&response.raw)?.into_iter().enumerate() {
        match part {
            ImagePart::Text(text) => println!("{text}"),
            ImagePart::Image { mime_type, data } => {
                let extension = mime_type.strip_prefix("image/").unwrap_or("png");
                let path = std::env::temp_dir().join(format!("rig-nanobanana-{index}.{extension}"));
                std::fs::write(&path, data)?;
                println!("Wrote {}", path.display());
            }
        }
    }

    Ok(())
}
