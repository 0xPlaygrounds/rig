use super::encoding::{self, AnyEncoding};
use crate::nothing::{Nothing, is_nothing};
use serde::{Deserialize, Serialize};
use std::marker::PhantomData;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Text {
    pub text: String,
}

impl<S> From<S> for Text
where
    S: Into<String>,
{
    fn from(value: S) -> Self {
        Self { text: value.into() }
    }
}

// Media-specific types
#[repr(u8)]
#[derive(
    Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default,
)]
pub enum Quality {
    Low,
    High,
    #[default]
    Auto,
}

#[derive(Debug, Clone, thiserror::Error)]
#[error("Unexpected mime type '{0}'")]
pub struct ParseMimeError(String);

pub trait MimeType: Sized {
    fn to_mime(&self) -> &'static str;
    fn from_mime(input: impl AsRef<str>) -> Result<Self, ParseMimeError>;
}

#[repr(u8)]
#[derive(
    Default, Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash,
)]
#[serde(rename_all = "lowercase")]
pub enum ImageMediaType {
    JPEG,
    #[default]
    PNG,
    GIF,
    WEBP,
    HEIC,
    HEIF,
    SVG,
}

impl MimeType for ImageMediaType {
    fn from_mime(mime_type: impl AsRef<str>) -> Result<Self, ParseMimeError> {
        match mime_type.as_ref() {
            "image/jpeg" => Ok(ImageMediaType::JPEG),
            "image/png" => Ok(ImageMediaType::PNG),
            "image/gif" => Ok(ImageMediaType::GIF),
            "image/webp" => Ok(ImageMediaType::WEBP),
            "image/heic" => Ok(ImageMediaType::HEIC),
            "image/heif" => Ok(ImageMediaType::HEIF),
            "image/svg+xml" => Ok(ImageMediaType::SVG),
            otherwise => Err(ParseMimeError(otherwise.to_string())),
        }
    }

    fn to_mime(&self) -> &'static str {
        match self {
            ImageMediaType::JPEG => "image/jpeg",
            ImageMediaType::PNG => "image/png",
            ImageMediaType::GIF => "image/gif",
            ImageMediaType::WEBP => "image/webp",
            ImageMediaType::HEIC => "image/heic",
            ImageMediaType::HEIF => "image/heif",
            ImageMediaType::SVG => "image/svg+xml",
        }
    }
}

#[repr(u8)]
#[derive(
    Default, Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash,
)]
#[serde(rename_all = "lowercase")]
pub enum AudioMediaType {
    #[default]
    WAV,
    MP3,
    AIFF,
    AAC,
    OGG,
    FLAC,
}

impl MimeType for AudioMediaType {
    fn from_mime(mime_type: impl AsRef<str>) -> Result<Self, ParseMimeError> {
        match mime_type.as_ref() {
            "audio/wav" => Ok(AudioMediaType::WAV),
            "audio/mp3" => Ok(AudioMediaType::MP3),
            "audio/aiff" => Ok(AudioMediaType::AIFF),
            "audio/aac" => Ok(AudioMediaType::AAC),
            "audio/ogg" => Ok(AudioMediaType::OGG),
            "audio/flac" => Ok(AudioMediaType::FLAC),
            otherwise => Err(ParseMimeError(otherwise.to_string())),
        }
    }

    fn to_mime(&self) -> &'static str {
        match self {
            AudioMediaType::WAV => "audio/wav",
            AudioMediaType::MP3 => "audio/mp3",
            AudioMediaType::AIFF => "audio/aiff",
            AudioMediaType::AAC => "audio/aac",
            AudioMediaType::OGG => "audio/ogg",
            AudioMediaType::FLAC => "audio/flac",
        }
    }
}

#[repr(u8)]
#[derive(
    Default, Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash,
)]
#[serde(rename_all = "lowercase")]
pub enum VideoMediaType {
    AVI,
    #[default]
    MP4,
    MPEG,
}

impl MimeType for VideoMediaType {
    fn from_mime(mime_type: impl AsRef<str>) -> Result<Self, ParseMimeError>
    where
        Self: Sized,
    {
        match mime_type.as_ref() {
            "video/avi" => Ok(VideoMediaType::AVI),
            "video/mp4" => Ok(VideoMediaType::MP4),
            "video/mpeg" => Ok(VideoMediaType::MPEG),
            otherwise => Err(ParseMimeError(otherwise.to_string())),
        }
    }

    fn to_mime(&self) -> &'static str {
        match self {
            VideoMediaType::AVI => "video/avi",
            VideoMediaType::MP4 => "video/mp4",
            VideoMediaType::MPEG => "video/mpeg",
        }
    }
}

/// Describes the document media type of the content. Not every provider supports every media type.
/// Includes also programming languages as document types for providers who support code running.
/// Convertible to and from MIME type strings.
#[derive(Default, Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum DocumentMediaType {
    PDF,
    #[default]
    TXT,
    RTF,
    HTML,
    CSS,
    MARKDOWN,
    CSV,
    XML,
    Javascript,
    Python,
}

impl DocumentMediaType {
    pub fn is_code(&self) -> bool {
        matches!(self, Self::Javascript | Self::Python)
    }
}

impl MimeType for DocumentMediaType {
    fn from_mime(mime_type: impl AsRef<str>) -> Result<Self, ParseMimeError> {
        match mime_type.as_ref() {
            "application/pdf" => Ok(DocumentMediaType::PDF),
            "text/plain" => Ok(DocumentMediaType::TXT),
            "text/rtf" => Ok(DocumentMediaType::RTF),
            "text/html" => Ok(DocumentMediaType::HTML),
            "text/css" => Ok(DocumentMediaType::CSS),
            "text/md" | "text/markdown" => Ok(DocumentMediaType::MARKDOWN),
            "text/csv" => Ok(DocumentMediaType::CSV),
            "text/xml" => Ok(DocumentMediaType::XML),
            "application/x-javascript" | "text/x-javascript" => Ok(DocumentMediaType::Javascript),
            "application/x-python" | "text/x-python" => Ok(DocumentMediaType::Python),
            otherwise => Err(ParseMimeError(otherwise.to_string())),
        }
    }

    fn to_mime(&self) -> &'static str {
        match self {
            DocumentMediaType::PDF => "application/pdf",
            DocumentMediaType::TXT => "text/plain",
            DocumentMediaType::RTF => "text/rtf",
            DocumentMediaType::HTML => "text/html",
            DocumentMediaType::CSS => "text/css",
            DocumentMediaType::MARKDOWN => "text/markdown",
            DocumentMediaType::CSV => "text/csv",
            DocumentMediaType::XML => "text/xml",
            DocumentMediaType::Javascript => "application/x-javascript",
            DocumentMediaType::Python => "application/x-python",
        }
    }
}

#[derive(Debug, Clone)]
pub struct Image;

#[derive(Debug, Clone)]
pub struct Audio;

#[derive(Debug, Clone)]
pub struct Video;

#[derive(Debug, Clone)]
pub struct Document;

trait GetMime {
    type Output;
}

type Mime<T> = <T as GetMime>::Output;

impl GetMime for Image {
    type Output = ImageMediaType;
}

impl GetMime for Audio {
    type Output = AudioMediaType;
}

impl GetMime for Video {
    type Output = VideoMediaType;
}

impl GetMime for Document {
    type Output = DocumentMediaType;
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Media<MediaKind, Encoding, MediaType = Nothing, Quality = Nothing>
where
    Encoding: 'static,
    MediaType: 'static,
    Quality: 'static,
{
    __media_kind: PhantomData<MediaKind>,
    #[serde(skip_serializing_if = "is_nothing", default)]
    pub data: Encoding,
    #[serde(skip_serializing_if = "is_nothing", default)]
    pub media_type: MediaType,
    #[serde(skip_serializing_if = "is_nothing", default)]
    pub quality: Quality,
}

impl Media<Image, Nothing, Nothing, Nothing> {
    pub fn image() -> Self {
        Self {
            __media_kind: PhantomData,
            data: Nothing,
            media_type: Nothing,
            quality: Nothing,
        }
    }
}

impl Media<Audio, Nothing, Nothing, Nothing> {
    pub fn audio() -> Self {
        Self {
            __media_kind: PhantomData,
            data: Nothing,
            media_type: Nothing,
            quality: Nothing,
        }
    }
}

impl Media<Video, Nothing, Nothing, Nothing> {
    pub fn video() -> Self {
        Self {
            __media_kind: PhantomData,
            data: Nothing,
            media_type: Nothing,
            quality: Nothing,
        }
    }
}

// Builder methods for setting encoding
impl<K, MT, Q> Media<K, Nothing, MT, Q> {
    pub fn base64(self, data: impl Into<String>) -> Media<K, encoding::Base64, MT, Q> {
        Media {
            __media_kind: PhantomData,
            data: encoding::Base64(data.into()),
            media_type: self.media_type,
            quality: self.quality,
        }
    }

    pub fn uri(self, uri: url::Url) -> Media<K, encoding::Url, MT, Q> {
        Media {
            __media_kind: PhantomData,
            data: encoding::Url(uri),
            media_type: self.media_type,
            quality: self.quality,
        }
    }

    pub fn raw(self, data: impl Into<Vec<u8>>) -> Media<K, encoding::Raw, MT, Q> {
        Media {
            __media_kind: PhantomData,
            data: encoding::Raw(data.into()),
            media_type: self.media_type,
            quality: self.quality,
        }
    }
}

// Builder methods for setting media type
impl<E, Q> Media<Image, E, Nothing, Q> {
    pub fn media_type(self, media_type: ImageMediaType) -> Media<Image, E, ImageMediaType, Q> {
        Media {
            __media_kind: PhantomData,
            data: self.data,
            media_type,
            quality: self.quality,
        }
    }
}

impl<E, Q> Media<Audio, E, Nothing, Q> {
    pub fn media_type(self, media_type: AudioMediaType) -> Media<Audio, E, AudioMediaType, Q> {
        Media {
            __media_kind: PhantomData,
            data: self.data,
            media_type,
            quality: self.quality,
        }
    }
}

impl<E, Q> Media<Video, E, Nothing, Q> {
    pub fn media_type(self, media_type: VideoMediaType) -> Media<Video, E, VideoMediaType, Q> {
        Media {
            __media_kind: PhantomData,
            data: self.data,
            media_type,
            quality: self.quality,
        }
    }
}

// Builder methods for setting quality (only for Image)
impl<E, MT> Media<Image, E, MT, Nothing> {
    pub fn quality(self, quality: Quality) -> Media<Image, E, MT, Quality> {
        Media {
            __media_kind: PhantomData,
            data: self.data,
            media_type: self.media_type,
            quality,
        }
    }
}

// Standard From implementations for Nothing -> concrete types (defaults)
impl From<Nothing> for AnyEncoding {
    fn from(_: Nothing) -> Self {
        AnyEncoding::Raw(encoding::Raw::default())
    }
}

impl From<Nothing> for ImageMediaType {
    fn from(_: Nothing) -> Self {
        ImageMediaType::JPEG
    }
}

impl From<Nothing> for AudioMediaType {
    fn from(_: Nothing) -> Self {
        AudioMediaType::MP3
    }
}

impl From<Nothing> for VideoMediaType {
    fn from(_: Nothing) -> Self {
        VideoMediaType::MP4
    }
}

impl From<Nothing> for Quality {
    fn from(_: Nothing) -> Self {
        Quality::Auto
    }
}

// Tool-specific types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolFunction {
    pub name: String,
    pub arguments: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub name: String,
    pub function: ToolFunction,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult<Data = Nothing, CallId = Nothing> {
    pub data: Data,
    pub call_id: CallId,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Reasoning<Data = Nothing, Signature = Nothing, Id = Nothing> {
    pub data: Data,
    pub signature: Signature,
    pub id: Id,
}

// Conversions for Text
impl From<Text> for UserPart {
    fn from(text: Text) -> Self {
        Self::Text(text)
    }
}

// USER CONTENT

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum UserPart {
    Text(Text),
    ToolResult(ToolResult<String, String>),
    Image(Media<Image, encoding::AnyEncoding, ImageMediaType, Quality>),
    Audio(Media<Audio, encoding::AnyEncoding, AudioMediaType>),
    Video(Media<Video, encoding::AnyEncoding, VideoMediaType>),
    Document(Media<Document, encoding::AnyEncoding, DocumentMediaType>),
}

impl From<String> for UserPart {
    fn from(value: String) -> Self {
        Self::Text(Text::from(value))
    }
}

impl<E, MT, Q> From<Media<Image, E, MT, Q>> for UserPart
where
    E: Into<AnyEncoding>,
    MT: Into<ImageMediaType>,
    Q: Into<Quality>,
{
    fn from(media: Media<Image, E, MT, Q>) -> Self {
        Self::Image(Media {
            __media_kind: PhantomData,
            data: media.data.into(),
            media_type: media.media_type.into(),
            quality: media.quality.into(),
        })
    }
}

impl<E, MT> From<Media<Audio, E, MT>> for UserPart
where
    E: Into<AnyEncoding>,
    MT: Into<AudioMediaType>,
{
    fn from(media: Media<Audio, E, MT>) -> Self {
        Self::Audio(Media {
            __media_kind: PhantomData,
            data: media.data.into(),
            media_type: media.media_type.into(),
            quality: Nothing,
        })
    }
}

impl<E, MT> From<Media<Video, E, MT>> for UserPart
where
    E: Into<AnyEncoding>,
    MT: Into<VideoMediaType>,
{
    fn from(media: Media<Video, E, MT>) -> Self {
        Self::Video(Media {
            __media_kind: PhantomData,
            data: media.data.into(),
            media_type: media.media_type.into(),
            quality: Nothing,
        })
    }
}

impl<E, MT> From<Media<Document, E, MT>> for UserPart
where
    E: Into<AnyEncoding>,
    MT: Into<DocumentMediaType>,
{
    fn from(media: Media<Document, E, MT>) -> Self {
        Self::Document(Media {
            __media_kind: PhantomData,
            data: media.data.into(),
            media_type: media.media_type.into(),
            quality: Nothing,
        })
    }
}

impl<D, C> From<ToolResult<D, C>> for UserPart
where
    D: Into<String>,
    C: Into<String>,
{
    fn from(tool_result: ToolResult<D, C>) -> Self {
        Self::ToolResult(ToolResult {
            data: tool_result.data.into(),
            call_id: tool_result.call_id.into(),
        })
    }
}

// ASSISTANT CONTENT

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AssistantPart {
    Text(Text),
    ToolCall(ToolCall),
    Reasoning(Reasoning<String, String, String>),
    Image(Media<Image, encoding::AnyEncoding, ImageMediaType, Quality>),
}

impl<S> From<S> for AssistantPart
where
    S: Into<String>,
{
    fn from(value: S) -> Self {
        Self::Text(value.into().into())
    }
}

impl<D, S, I> From<Reasoning<D, S, I>> for AssistantPart
where
    D: Into<String>,
    S: Into<String>,
    I: Into<String>,
{
    fn from(
        Reasoning {
            data,
            signature,
            id,
        }: Reasoning<D, S, I>,
    ) -> AssistantPart {
        AssistantPart::Reasoning(Reasoning {
            data: data.into(),
            signature: signature.into(),
            id: id.into(),
        })
    }
}

impl From<ToolCall> for AssistantPart {
    fn from(tool_call: ToolCall) -> Self {
        Self::ToolCall(tool_call)
    }
}
