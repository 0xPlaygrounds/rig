//! MiniMax's image, speech and voice-cloning wires. These are unit tests
//! because no MiniMax recording exists: each reply is built here in the shape
//! MiniMax's API reference documents.

use serde_json::json;

use crate::driver::Model;
use crate::error::ErrorKind;
use crate::providers::openai::wire::{MINIMAX, OpenAIConfig};
use crate::test_utils::{RecordingHttpClient, json_body};
use crate::wire::{Mode, Wire};
#[cfg(feature = "audio")]
use crate::{
    http_client::multipart::PartContent,
    providers::minimax::voice_clone::{CloneAudio, VoiceClone, VoiceCloneRequest, VoiceClones},
    providers::openai::wire::OPENAI,
    wire::Body,
};

/// The image body asks for base64 at the width and height the caller set,
/// leaves the size out for an `aspect_ratio` or a model that reads none, and
/// refuses a size MiniMax rejects, such as the builder's 256 by 256 default. The reply's first
/// base64 image is the response, its trace id the response id.
#[cfg(feature = "image")]
#[tokio::test]
async fn an_image_request_asks_for_base64_and_decodes_the_first_image() {
    use crate::image_generation::ImageGenerationRequestBuilder;
    use crate::providers::minimax::IMAGE_01;

    let wire = || OpenAIConfig::with_key(&MINIMAX, "key").image_generation(IMAGE_01);
    let request = || {
        ImageGenerationRequestBuilder::new("a lighthouse")
            .width(1024)
            .height(768)
            .build()
    };
    let encoded = OpenAIConfig::with_key(&crate::providers::openai::wire::MINIMAX_CHINA, "key")
        .image_generation(IMAGE_01)
        .encode(request(), Mode::Unary)
        .expect("the request encodes");
    assert_eq!(
        encoded.request.uri().to_string(),
        "https://api.minimax.cn/v1/image_generation"
    );
    assert_eq!(
        json_body(&encoded.request),
        json!({
            "model": "image-01",
            "prompt": "a lighthouse",
            "width": 1024,
            "height": 768,
            "response_format": "base64",
        })
    );

    let ratio = ImageGenerationRequestBuilder::new("a lighthouse")
        .additional_params(json!({"aspect_ratio": "16:9"}))
        .build();
    let encoded = wire()
        .encode(ratio, Mode::Unary)
        .expect("an aspect ratio stands for the size");
    assert_eq!(
        json_body(&encoded.request),
        json!({
            "model": "image-01",
            "prompt": "a lighthouse",
            "response_format": "base64",
            "aspect_ratio": "16:9",
        })
    );

    let refusal = wire()
        .encode(
            ImageGenerationRequestBuilder::new("a lighthouse").build(),
            Mode::Unary,
        )
        .expect_err("256 by 256 is below MiniMax's minimum");
    assert!(refusal.to_string().contains("256x256"), "{refusal}");
    // A size in the parameters overrides the builder's, so it is the one checked.
    wire()
        .encode(
            ImageGenerationRequestBuilder::new("a lighthouse")
                .additional_params(json!({"width": 1024, "height": 1024}))
                .build(),
            Mode::Unary,
        )
        .expect("the parameters' size is in range");
    // `image-01-live` reads no size, so none is sent or checked.
    let live = OpenAIConfig::with_key(&MINIMAX, "key")
        .image_generation(crate::providers::minimax::IMAGE_01_LIVE)
        .encode(
            ImageGenerationRequestBuilder::new("a lighthouse").build(),
            Mode::Unary,
        )
        .expect("a live request needs no size");
    assert!(json_body(&live.request).get("width").is_none());

    let reply = json!({
        "id": "03ff3cd0820949eb8a410056b5f21d38",
        "data": {"image_base64": ["aGk=", "aGk="]},
        "metadata": {"success_count": 2, "failed_count": 0},
        "base_resp": {"status_code": 0, "status_msg": "success"},
    });
    let response = Model::new(wire(), RecordingHttpClient::new(reply.to_string()))
        .call(request())
        .await
        .expect("the reply decodes");
    assert_eq!(response.image, b"hi");
    assert_eq!(response.provider, "minimax");
    assert_eq!(
        response.response_id.as_deref(),
        Some("03ff3cd0820949eb8a410056b5f21d38")
    );
}

/// MiniMax reports a failure under HTTP 200 with a nonzero `base_resp`
/// status, which must fail the call with the body rather than read as a
/// reply with no image.
#[cfg(feature = "image")]
#[tokio::test]
async fn a_failed_status_under_http_200_is_a_provider_error() {
    use crate::image_generation::ImageGenerationRequestBuilder;

    let body = json!({
        "id": "trace",
        "data": null,
        "base_resp": {"status_code": 1026, "status_msg": "input new_sensitive"},
    })
    .to_string();
    let error = Model::new(
        OpenAIConfig::with_key(&MINIMAX, "key").image_generation("image-01"),
        RecordingHttpClient::new(body),
    )
    .call(
        ImageGenerationRequestBuilder::new("x")
            .width(1024)
            .height(1024)
            .build(),
    )
    .await
    .expect_err("a failed status fails the call");
    assert_eq!(error.kind(), ErrorKind::ProviderResponse);
    assert!(error.to_string().contains("1026"), "{error}");
}

/// The speech body nests the voice and speed under `voice_setting`, a
/// caller's `voice_setting` adds to it, and the hex audio decodes to bytes.
#[cfg(feature = "audio")]
#[tokio::test]
async fn a_speech_request_nests_the_voice_and_decodes_hex_audio() {
    use crate::audio_generation::AudioGenerationRequestBuilder;
    use crate::providers::minimax::SPEECH_2_8_HD;

    let request = || {
        AudioGenerationRequestBuilder::new("Hello", "English_Graceful_Lady")
            .speed(1.25)
            .additional_params(json!({
                "voice_setting": {"vol": 2},
                "audio_setting": {"format": "wav"},
            }))
            .build()
    };
    let encoded = OpenAIConfig::with_key(&MINIMAX, "key")
        .audio_generation(SPEECH_2_8_HD)
        .encode(request(), Mode::Unary)
        .expect("the request encodes");
    assert_eq!(
        encoded.request.uri().to_string(),
        "https://api.minimax.io/v1/t2a_v2"
    );
    assert_eq!(
        json_body(&encoded.request),
        json!({
            "model": "speech-2.8-hd",
            "text": "Hello",
            "stream": false,
            "output_format": "hex",
            "voice_setting": {"voice_id": "English_Graceful_Lady", "speed": 1.25, "vol": 2},
            "audio_setting": {"format": "wav"},
        })
    );
    // Without a caller `voice_setting`, the derived one goes out as is.
    for params in [None, Some(json!({"language_boost": "auto"}))] {
        let encoded = OpenAIConfig::with_key(&MINIMAX, "key")
            .audio_generation(SPEECH_2_8_HD)
            .encode(
                AudioGenerationRequestBuilder::new("Hello", "English_Graceful_Lady")
                    .additional_params(params)
                    .build(),
                Mode::Unary,
            )
            .expect("the request encodes");
        assert_eq!(
            json_body(&encoded.request)["voice_setting"],
            json!({"voice_id": "English_Graceful_Lady", "speed": 1.0})
        );
    }

    let reply = json!({
        "data": {"audio": "52494646ff00", "status": 2},
        "extra_info": {"audio_format": "wav", "usage_characters": 5},
        "trace_id": "01b8bf9bb7433cc75c18eee6cfa8fe21",
        "base_resp": {"status_code": 0, "status_msg": "success"},
    });
    let response = Model::new(
        OpenAIConfig::with_key(&MINIMAX, "key").audio_generation(SPEECH_2_8_HD),
        RecordingHttpClient::new(reply.to_string()),
    )
    .call(request())
    .await
    .expect("the reply decodes");
    assert_eq!(response.audio, b"RIFF\xff\x00");
    assert_eq!(
        response.response_id.as_deref(),
        Some("01b8bf9bb7433cc75c18eee6cfa8fe21")
    );

    // `output_format: url` answers with a link, which is not audio.
    let url_reply = json!({"data": {"audio": "https://cdn.example/a.mp3", "status": 2}});
    let error = Model::new(
        OpenAIConfig::with_key(&MINIMAX, "key").audio_generation(SPEECH_2_8_HD),
        RecordingHttpClient::new(url_reply.to_string()),
    )
    .call(request())
    .await
    .expect_err("a link is not hex audio");
    assert!(error.to_string().contains("hex"), "{error}");
}

/// The upload is a multipart form with the purpose beside the file, and
/// refuses a format MiniMax does not take; the clone is a JSON body with the
/// prompt, preview and extra fields.
#[cfg(feature = "audio")]
#[test]
fn voice_cloning_requests_are_the_documented_shapes() {
    let wire = VoiceClones::new(OpenAIConfig::with_key(&MINIMAX, "key"));

    let upload = wire
        .encode(
            VoiceCloneRequest::Upload(CloneAudio::prompt("clip.wav", b"RIFF".to_vec())),
            Mode::Unary,
        )
        .expect("the upload encodes");
    assert_eq!(
        upload.request.uri().to_string(),
        "https://api.minimax.io/v1/files/upload"
    );
    let Body::Multipart(form) = upload.request.body() else {
        panic!("the upload is multipart");
    };
    let parts: Vec<_> = form
        .parts()
        .iter()
        .map(|part| {
            let content = match part.content() {
                PartContent::Text(text) => text.clone(),
                PartContent::Binary(bytes) => String::from_utf8_lossy(bytes).into_owned(),
            };
            (
                part.name().to_owned(),
                part.get_filename().map(str::to_owned),
                content,
            )
        })
        .collect();
    assert_eq!(
        parts,
        vec![
            ("purpose".to_owned(), None, "prompt_audio".to_owned()),
            (
                "file".to_owned(),
                Some("clip.wav".to_owned()),
                "RIFF".to_owned()
            ),
        ]
    );

    let refusal = wire
        .encode(
            VoiceCloneRequest::Upload(CloneAudio::sample("sample.ogg", b"OggS".to_vec())),
            Mode::Unary,
        )
        .expect_err("MiniMax takes no ogg");
    assert!(refusal.to_string().contains("sample.ogg"), "{refusal}");

    let clone = VoiceClone::new(123_456_789_012_345_680, "MyVoice001")
        .prompt(987_654_321, "This voice sounds natural.")
        .preview("Hello there.", "speech-2.8-hd")
        .additional_params(json!({"need_noise_reduction": true}));
    let encoded = wire
        .encode(VoiceCloneRequest::Clone(clone), Mode::Unary)
        .expect("the clone encodes");
    assert_eq!(
        encoded.request.uri().to_string(),
        "https://api.minimax.io/v1/voice_clone"
    );
    assert_eq!(
        json_body(&encoded.request),
        json!({
            "file_id": 123_456_789_012_345_680_i64,
            "voice_id": "MyVoice001",
            "clone_prompt": {"prompt_audio": 987_654_321, "prompt_text": "This voice sounds natural."},
            "text": "Hello there.",
            "model": "speech-2.8-hd",
            "need_noise_reduction": true,
        })
    );

    // Another provider's client has no such endpoint.
    let refusal = VoiceClones::new(OpenAIConfig::with_key(&OPENAI, "key"))
        .encode(
            VoiceCloneRequest::Clone(VoiceClone::new(1, "MyVoice001")),
            Mode::Unary,
        )
        .expect_err("OpenAI serves no voice cloning");
    assert!(refusal.to_string().contains("MiniMax"), "{refusal}");
}

/// An upload reply yields its file id exactly, past `f64` precision, and a
/// clone reply its preview link and the voice id the request created.
#[cfg(feature = "audio")]
#[tokio::test]
async fn voice_cloning_replies_decode() {
    let wire = || VoiceClones::new(OpenAIConfig::with_key(&MINIMAX, "key"));

    let upload_reply = json!({
        "file": {
            "file_id": 123_456_789_012_345_680_i64,
            "bytes": 5_896_337,
            "created_at": 1_700_469_398,
            "filename": "sample.mp3",
            "purpose": "voice_clone",
        },
        "base_resp": {"status_code": 0, "status_msg": "success"},
    });
    let file = Model::new(wire(), RecordingHttpClient::new(upload_reply.to_string()))
        .upload(CloneAudio::sample("sample.mp3", b"ID3".to_vec()))
        .await
        .expect("the upload reply decodes");
    assert_eq!(file.file_id, 123_456_789_012_345_680);

    let clone_reply = json!({
        "input_sensitive": false,
        "demo_audio": "https://cdn.example/preview.mp3",
        "extra_info": {"usage_characters": 12},
        "base_resp": {"status_code": 0, "status_msg": "success"},
    });
    let voice = Model::new(wire(), RecordingHttpClient::new(clone_reply.to_string()))
        .clone_voice(
            VoiceClone::new(file.file_id, "MyVoice001")
                .additional_params(json!({"voice_id": "MyVoice002"})),
        )
        .await
        .expect("the clone reply decodes");
    // The id the request body carried, which `additional_params` replaced.
    assert_eq!(voice.voice_id, "MyVoice002");
    assert_eq!(
        voice.demo_audio.as_deref(),
        Some("https://cdn.example/preview.mp3")
    );

    let refused =
        json!({"base_resp": {"status_code": 2038, "status_msg": "no cloning permission"}});
    let error = Model::new(wire(), RecordingHttpClient::new(refused.to_string()))
        .clone_voice(VoiceClone::new(file.file_id, "MyVoice001"))
        .await
        .expect_err("a failed status fails the clone");
    assert_eq!(error.kind(), ErrorKind::ProviderResponse);
}
