//! A2A message parts as Rig user content.

use a2a::{Part, PartContent};
use rig_core::message::{
    DocumentMediaType, ImageMediaType, MediaType, Message, MimeType, UserContent,
};

/// A message part that cannot be passed to a model.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub(crate) enum UnsupportedPart {
    /// The message has no parts.
    #[error("the message has no parts")]
    Empty,
    /// A file part's media type has no Rig content kind.
    #[error("the media type `{0}` is not supported")]
    MediaType(String),
    /// A file URL part without a media type.
    #[error("a file URL part needs a mediaType")]
    UntypedUrl,
    /// Raw bytes without a media type that are neither a known image nor text.
    #[error("raw bytes without a mediaType must be an image or UTF-8 text")]
    UntypedBytes,
}

/// The user message for an A2A message: text and structured data as text,
/// files as image, audio, video or document content by their media type.
pub(crate) fn user_message(message: &a2a::Message) -> Result<Message, UnsupportedPart> {
    if message.parts.is_empty() {
        return Err(UnsupportedPart::Empty);
    }
    let content = message
        .parts
        .iter()
        .map(user_content)
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Message::User { content })
}

fn user_content(part: &Part) -> Result<UserContent, UnsupportedPart> {
    let media_type = part.media_type.as_deref().filter(|media| !media.is_empty());
    match &part.content {
        PartContent::Text(text) => Ok(UserContent::text(text)),
        PartContent::Data(value) => Ok(UserContent::text(value.to_string())),
        PartContent::Raw(bytes) => raw_content(bytes, media_type),
        PartContent::Url(url) => {
            let media_type = media_type.ok_or(UnsupportedPart::UntypedUrl)?;
            match MediaType::from_mime_type(media_type) {
                Some(MediaType::Image(kind)) => Ok(UserContent::image_url(url, Some(kind), None)),
                Some(MediaType::Audio(kind)) => Ok(UserContent::audio_url(url, Some(kind))),
                Some(MediaType::Video(kind)) => Ok(UserContent::video_url(url, Some(kind))),
                Some(MediaType::Document(kind)) => Ok(UserContent::document_url(url, Some(kind))),
                None => Err(UnsupportedPart::MediaType(media_type.to_owned())),
            }
        }
    }
}

fn raw_content(bytes: &[u8], media_type: Option<&str>) -> Result<UserContent, UnsupportedPart> {
    let text = std::str::from_utf8(bytes).ok();
    let Some(media_type) = media_type else {
        return match (ImageMediaType::sniff(bytes), text) {
            (Some(kind), _) => Ok(UserContent::image_raw(bytes, Some(kind), None)),
            (None, Some(text)) => Ok(UserContent::text(text)),
            (None, None) => Err(UnsupportedPart::UntypedBytes),
        };
    };
    match (MediaType::from_mime_type(media_type), text) {
        (Some(MediaType::Image(kind)), _) => Ok(UserContent::image_raw(bytes, Some(kind), None)),
        (Some(MediaType::Audio(kind)), _) => Ok(UserContent::audio_raw(bytes, Some(kind))),
        (Some(MediaType::Video(kind)), _) => Ok(UserContent::video_raw(bytes, Some(kind))),
        (Some(MediaType::Document(DocumentMediaType::PDF)), _) => Ok(UserContent::document_raw(
            bytes,
            Some(DocumentMediaType::PDF),
        )),
        (Some(MediaType::Document(kind)), Some(text)) => {
            Ok(UserContent::document_text(text, Some(kind)))
        }
        (Some(MediaType::Document(kind)), None) => Ok(UserContent::document_raw(bytes, Some(kind))),
        (None, Some(text)) if media_type == "application/json" => Ok(UserContent::text(text)),
        (None, _) => Err(UnsupportedPart::MediaType(media_type.to_owned())),
    }
}
