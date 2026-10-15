//! Gemini 3 image generation and Gemini text-to-speech, recorded.

use rig::audio_generation::AudioGenerationRequestBuilder;
use rig::image_generation::ImageGenerationRequestBuilder;
use rig::message::{DocumentSourceKind, Image, ImageMediaType};
use rig::providers::gemini::ImageSize;
use rig::providers::gemini::audio_generation::{
    AudioFormat, GEMINI_3_8_FLASH_LITE_TTS, GEMINI_3_8_FLASH_TTS, Script, Turn, format_of,
};
use rig::providers::gemini::image_generation::{
    GEMINI_3_1_FLASH_IMAGE, ImageOptions, ImagePart, parts_of,
};

use super::super::support::{recorded_request_bodies, with_gemini_cassette};

/// An 8 by 8 yellow PNG.
const YELLOW_SQUARE: &str = "iVBORw0KGgoAAAANSUhEUgAAAAgAAAAICAIAAABLbSncAAAAEUlEQVR42mP4f40BK2IYWhIAEDp1QbHYEGgAAAAASUVORK5CYII=";

/// A Gemini 3 image model draws from a reference image at the 512 size and
/// writes text beside the image; every part comes back.
#[tokio::test]
async fn gemini_3_image_with_reference_size_and_text() {
    const SCENARIO: &str = "image_generation/gemini_3_reference_size_text";
    with_gemini_cassette(
        "image_generation/gemini_3_reference_size_text",
        |client| async move {
            let request = ImageGenerationRequestBuilder::new(
                "Write one sentence describing a small flat banana icon in the colour of this \
                 square, then draw the icon.",
            )
            .width(1)
            .height(1)
            .image(Image {
                data: DocumentSourceKind::Base64(YELLOW_SQUARE.to_owned()),
                media_type: Some(ImageMediaType::PNG),
                ..Image::default()
            })
            .additional_params(
                ImageOptions::new()
                    .image_size(ImageSize::FiveTwelve)
                    .with_text()
                    .params(),
            )
            .build();
            let response = client
                .image_generation(GEMINI_3_1_FLASH_IMAGE)
                .call(request)
                .await
                .expect("the image generates");
            let parts = parts_of(&response.raw).expect("the reply's parts decode");
            assert!(
                parts.iter().any(
                    |part| matches!(part, ImagePart::Image { data, .. } if *data == response.image)
                ),
                "the response image is one of the reply's images"
            );
            assert!(
                parts
                    .iter()
                    .any(|part| matches!(part, ImagePart::Text(text) if !text.trim().is_empty())),
                "the reply carries text beside the image"
            );
        },
    )
    .await;
    let body = recorded_request_bodies(SCENARIO);
    let body = body.first().expect("one request");
    assert_eq!(
        body["generationConfig"]["imageConfig"],
        serde_json::json!({"aspectRatio": "1:1", "imageSize": "512"})
    );
    assert_eq!(
        body["contents"][0]["parts"][1]["inlineData"]["data"], YELLOW_SQUARE,
        "the reference image follows the prompt"
    );
}

/// One voice, the API's default format: a WAV file.
#[tokio::test]
async fn speech_in_one_voice_is_wav() {
    with_gemini_cassette("audio_generation/one_voice_wav", |client| async move {
        let response = client
            .audio_generation(GEMINI_3_8_FLASH_LITE_TTS)
            .call(AudioGenerationRequestBuilder::new("Have a wonderful day!", "Kore").build())
            .await
            .expect("the speech generates");
        assert!(response.audio.starts_with(b"RIFF"), "a WAV file");
        let format = format_of(&response).expect("the reply states its format");
        assert_eq!(format.mime_type.as_deref(), Some("audio/wav"));
    })
    .await;
}

/// Two speakers, each turn styled, as headerless PCM: the reply states the
/// sample rate and channels a caller joins chunks by.
#[tokio::test]
async fn a_two_speaker_script_is_pcm_with_its_rate() {
    with_gemini_cassette("audio_generation/two_speakers_pcm", |client| async move {
        let request = Script::speakers([("Joe", "Puck"), ("Jane", "Kore")])
            .turn(Turn::new("Ready to start?").speaker("Joe").style("excited"))
            .turn(Turn::new("Let's go.").speaker("Jane").style("calm"))
            .format(AudioFormat::pcm(24_000))
            .build();
        let response = client
            .audio_generation(GEMINI_3_8_FLASH_TTS)
            .call(request)
            .await
            .expect("the dialogue generates");
        assert!(
            !response.audio.is_empty() && response.audio.len() % 2 == 0,
            "16-bit samples"
        );
        assert!(!response.audio.starts_with(b"RIFF"), "no header");
        let format = format_of(&response).expect("the reply states its format");
        assert_eq!(format.sample_rate, Some(24_000));
        assert_eq!(format.channels, Some(1));
    })
    .await;
}
