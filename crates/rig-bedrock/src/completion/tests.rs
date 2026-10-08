use super::*;
use rig_core::completion::Place;
use rig_core::message::{
    Audio, AudioMediaType, Document, DocumentMediaType, DocumentSourceKind as Source, Image,
    ImageMediaType, Video, VideoMediaType,
};
use serde_json::json;

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
        Source::base64("bGV0IGEgPSAxOw==").into(),
        DocumentMediaType::Javascript,
    );
    assert!(encodes(
        ANTHROPIC_CLAUDE_SONNET_4_6,
        Media::Document(&script)
    ));
    let text = document(
        rig_core::message::DocumentData::Text("notes".into()),
        DocumentMediaType::TXT,
    );
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

/// What a wire on the default facts reads for `model`.
fn spec(model: &str) -> Option<&'static ModelSpec> {
    static FACTS: std::sync::OnceLock<ModelFacts> = std::sync::OnceLock::new();
    super::spec_in(FACTS.get_or_init(ModelFacts::default), model)
}

/// Bedrock finds a model's catalog entry by its id or the last part of its
/// ARN, and a Claude row carries the Anthropic model's facts.
#[test]
fn bedrock_models_take_their_facts_from_the_catalog() {
    let opus = spec("us.anthropic.claude-opus-5-5").expect("listed");
    assert!(opus.compat.binds_context && opus.reasoning.can_disable() == Some(false));
    let arn =
        "arn:aws:bedrock:us-east-1:123456789012:inference-profile/us.anthropic.claude-opus-5-5";
    assert_eq!(
        spec(arn).map(|spec| spec.id.as_str()),
        Some("us.anthropic.claude-opus-5-5")
    );
    assert!(
        spec(AMAZON_NOVA_LITE).is_some(),
        "another model's Bedrock row"
    );
    let haiku = spec(ANTHROPIC_CLAUDE_HAIKU_4_5).expect("listed");
    assert!(!haiku.compat.adaptive_thinking && haiku.reasoning.budget().is_some());
    assert!(
        spec("arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/a1b2c3")
            .is_none()
    );
}

/// A Claude id the catalog does not list under Bedrock (another region's
/// profile, a `-v1:N` revision, a dated snapshot) takes its base model's
/// Bedrock row, which carries the Anthropic facts, so its thinking and
/// context binding still apply.
#[test]
fn an_unlisted_bedrock_claude_id_takes_its_base_models_row() {
    for model in [
        "in.anthropic.claude-opus-5-5",
        "us.anthropic.claude-opus-5-5-20260101-v1:0",
        "anthropic.claude-opus-5-5-v1:0",
    ] {
        let spec = spec(model).unwrap_or_else(|| panic!("{model}: a Claude entry"));
        assert!(
            spec.compat.binds_context && spec.reasoning.can_disable() == Some(false),
            "{model}"
        );
    }
    let sonnet = spec("jp.anthropic.claude-sonnet-5-5").expect("a Claude entry");
    assert_eq!(sonnet.id, "anthropic.claude-sonnet-5-5");
    assert_eq!(sonnet.compat.thinking_off.as_deref(), Some("between_tools"));
    assert!(sonnet.compat.binds_context);
    assert!(spec("jp.amazon.nova-unlisted-v1:0").is_none());
    assert!(spec("anthropic.claude-unlisted-v1:0").is_none());
}

/// Only a lower-case region prefix is a profile, and only `-v` followed by
/// digits is a revision: anything else is an id the catalog does not list.
#[test]
fn a_malformed_profile_or_revision_is_not_read_as_its_base() {
    for model in [
        ".anthropic.claude-opus-5-5",
        "US.anthropic.claude-opus-5-5",
        "anthropic.claude-opus-5-5-vnext",
        "anthropic.claude-opus-5-5-v1:x",
        "anthropic.claude-opus-5-5-v",
    ] {
        assert!(spec(model).is_none(), "{model}");
    }
}

/// A region profile the catalog does not list reads images as its base
/// model's Bedrock row says. A model listed under neither reads images
/// unless its id names a text-only family, so an unlisted model of another
/// family is sent images.
#[test]
fn an_unlisted_region_profile_reads_images_as_its_base_model() {
    for model in [
        "eu.meta.llama3-3-70b-instruct-v1:0",
        "us.deepseek.v3-v1:0",
        "arn:aws:bedrock:eu-west-1:123456789012:inference-profile/eu.meta.llama3-3-70b-instruct-v1:0",
        "us.writer.palmyra-unlisted-v1:0",
    ] {
        assert!(
            spec(model).is_none_or(|base| !base.input.image),
            "{model} reads text only"
        );
        let accepts = Converse::new(model).accepts(model);
        assert!(!accepts.user_images, "{model}");
        assert!(!accepts.tool_result_images, "{model}");
    }
    assert_eq!(
        spec("eu.meta.llama3-3-70b-instruct-v1:0").map(|base| base.id.as_str()),
        Some("meta.llama3-3-70b-instruct-v1:0")
    );
    assert!(spec("us.writer.palmyra-unlisted-v1:0").is_none());
    assert!(
        Converse::new("eu.amazon.nova-lite-v1:0")
            .accepts("eu.amazon.nova-lite-v1:0")
            .user_images
    );
    assert!(
        Converse::new("eu.acme.unlisted-v1:0")
            .accepts("eu.acme.unlisted-v1:0")
            .user_images
    );
}
