use super::*;
use rig_core::completion::Place;
use rig_core::message::{
    Audio, AudioMediaType, Document, DocumentMediaType, DocumentSourceKind as Source, Image,
    ImageMediaType, Video, VideoMediaType,
};

const PROFILE: &str =
    "arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/a1b2c3d4";

/// The family comes from the provider a model id names, or from the
/// caller for an id that names none.
#[test]
fn the_family_is_the_provider_the_id_names() {
    assert_eq!(Family::of(ANTHROPIC_CLAUDE_SONNET_4_6), Family::Claude);
    assert_eq!(
        Family::of("anthropic.claude-3-5-haiku-20241022-v1:0"),
        Family::Claude
    );
    assert_eq!(
        Family::of("arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-opus-4-7-v1:0"),
        Family::Claude
    );
    assert_eq!(Family::of(AMAZON_NOVA_LITE), Family::Nova);
    assert_eq!(Family::of("us.amazon.nova-2-lite-v1:0"), Family::Nova);
    assert_eq!(Family::of("amazon.titan-text-express-v1"), Family::Other);
    assert_eq!(Family::of(PROFILE), Family::Other);
    // A model that only mentions Claude in its name is not Claude.
    assert_eq!(
        Family::of("acme.anthropic-claude-clone-v1:0"),
        Family::Other
    );
    let wire = Converse::new(PROFILE).with_family(Family::Claude);
    assert_eq!(wire.family(PROFILE), Family::Claude);
    assert_eq!(wire.family(AMAZON_NOVA_LITE), Family::Nova);
    assert!(wire.accepts(PROFILE).user_images);
}

/// Converse carries inline images, documents in the formats it lists, and
/// S3 objects and video to Nova, which reads them; never URLs, file ids,
/// string documents (its text source is rejected) or audio.
#[test]
fn converse_encodes_only_what_it_carries() {
    let png = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==";
    let image = |data| Image {
        data,
        media_type: Some(ImageMediaType::PNG),
        ..Image::default()
    };
    let document = |data, media_type| Document {
        data,
        media_type: Some(media_type),
        additional_params: None,
    };
    let video = |data| Video {
        data,
        media_type: Some(VideoMediaType::MP4),
        additional_params: None,
    };
    let encodes = |model: &str, media| Converse::new(model).encodes(model, media);
    let s3 = || Source::url("s3://bucket/key");
    let inline = image(Source::base64(png));
    assert!(encodes(
        ANTHROPIC_CLAUDE_SONNET_4_6,
        Media::Image(&inline, Place::User)
    ));
    let stored = image(s3());
    assert!(encodes(
        AMAZON_NOVA_LITE,
        Media::Image(&stored, Place::ToolResult)
    ));
    assert!(!encodes(
        ANTHROPIC_CLAUDE_SONNET_4_6,
        Media::Image(&stored, Place::User)
    ));
    let url = image(Source::url("https://example.com/a.png"));
    assert!(!encodes(AMAZON_NOVA_LITE, Media::Image(&url, Place::User)));
    let heic = Image {
        media_type: Some(ImageMediaType::HEIC),
        ..inline.clone()
    };
    assert!(!encodes(AMAZON_NOVA_LITE, Media::Image(&heic, Place::User)));
    let script = document(
        Source::base64("bGV0IGEgPSAxOw=="),
        DocumentMediaType::Javascript,
    );
    assert!(encodes(
        ANTHROPIC_CLAUDE_SONNET_4_6,
        Media::Document(&script)
    ));
    let text = document(Source::string("notes"), DocumentMediaType::TXT);
    assert!(!encodes(AMAZON_NOVA_LITE, Media::Document(&text)));
    let mp4 = video(Source::base64("AAAAIGZ0eXBpc29tAAACAA"));
    assert!(encodes(AMAZON_NOVA_LITE, Media::Video(&mp4)));
    let stored = video(s3());
    assert!(encodes(AMAZON_NOVA_LITE, Media::Video(&stored)));
    assert!(!encodes(AMAZON_NOVA_MICRO, Media::Video(&mp4)));
    assert!(!encodes(ANTHROPIC_CLAUDE_SONNET_4_6, Media::Video(&mp4)));
    let audio = Audio {
        data: Source::base64("SUQzBAAAAAAAI1RTU0UAAAAP"),
        media_type: Some(AudioMediaType::MP3),
    };
    assert!(!encodes(AMAZON_NOVA_LITE, Media::Audio(&audio)));
}

/// A hosted tool's use and result pair by the id they share; a client call
/// holds its id at the slot replay spells.
#[test]
fn hosted_items_pair_and_calls_have_an_id_slot() {
    let wire = Converse::new(AMAZON_NOVA_LITE);
    let used = json!({ "toolUse": { "toolUseId": "srv_1", "type": "server_tool_use" } });
    let result = json!({ "toolResult": { "toolUseId": "srv_1", "content": [] } });
    assert_eq!(
        wire.hosted_pair(&used),
        Some((Pairing::Use, "srv_1".to_owned()))
    );
    assert_eq!(
        wire.hosted_pair(&result),
        Some((Pairing::Result, "srv_1".to_owned()))
    );
    assert_eq!(wire.hosted_pair(&json!({ "image": {} })), None);
    assert_eq!(wire.call_id_slot(), Some("/toolUse/toolUseId"));
}
