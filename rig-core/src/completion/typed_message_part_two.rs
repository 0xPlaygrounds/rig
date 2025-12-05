use crate::{
    OneOrMany,
    completion::typed_message_part_two::encoding::AnyEncoding,
    nothing::Nothing,
    type_level::list::{Cons, List, Nil, NonEmpty},
};
use std::marker::PhantomData;

pub mod role {
    pub trait Label {
        const LABEL: &str;
    }

    pub struct User;
    pub struct Assistant;
}

pub mod content {
    use super::encoding;
    use crate::{completion::typed_message_part_two::encoding::AnyEncoding, nothing::Nothing};
    use serde::{Deserialize, Serialize};
    use std::marker::PhantomData;

    #[derive(Debug, Clone)]
    pub struct Text {
        pub text: String,
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

    #[repr(u8)]
    #[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
    #[serde(rename_all = "lowercase")]
    pub enum ImageMediaType {
        JPEG,
        PNG,
        GIF,
        WEBP,
        HEIC,
        HEIF,
        SVG,
    }

    #[repr(u8)]
    #[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
    #[serde(rename_all = "lowercase")]
    pub enum AudioMediaType {
        WAV,
        MP3,
        AIFF,
        AAC,
        OGG,
        FLAC,
    }

    #[repr(u8)]
    #[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
    #[serde(rename_all = "lowercase")]
    pub enum VideoMediaType {
        AVI,
        MP4,
        MPEG,
    }

    #[derive(Debug, Clone)]
    pub struct Image;

    #[derive(Debug, Clone)]
    pub struct Audio;

    #[derive(Debug, Clone)]
    pub struct Video;

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

    #[derive(Debug, Clone)]
    pub struct Media<MediaKind, Encoding, MediaType = Nothing, Quality = Nothing> {
        __media_kind: PhantomData<MediaKind>,
        pub data: Encoding,
        pub media_type: MediaType,
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

        pub fn uri(self, uri: http::Uri) -> Media<K, encoding::Uri, MT, Q> {
            Media {
                __media_kind: PhantomData,
                data: encoding::Uri(uri),
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

    // Conversion implementations for Media -> Part
    impl<E, MT, Q> From<Media<Image, E, MT, Q>> for Part
    where
        E: Into<AnyEncoding>,
        MT: Into<ImageMediaType>,
        Q: Into<Quality>,
    {
        fn from(media: Media<Image, E, MT, Q>) -> Part {
            Part::Image(Media {
                __media_kind: PhantomData,
                data: media.data.into(),
                media_type: media.media_type.into(),
                quality: media.quality.into(),
            })
        }
    }

    impl<E, MT> From<Media<Audio, E, MT>> for Part
    where
        E: Into<AnyEncoding>,
        MT: Into<AudioMediaType>,
    {
        fn from(media: Media<Audio, E, MT>) -> Part {
            Part::Audio(Media {
                __media_kind: PhantomData,
                data: media.data.into(),
                media_type: media.media_type.into(),
                quality: Nothing,
            })
        }
    }

    impl<E, MT> From<Media<Video, E, MT>> for Part
    where
        E: Into<AnyEncoding>,
        MT: Into<VideoMediaType>,
    {
        fn from(media: Media<Video, E, MT>) -> Part {
            Part::Video(Media {
                __media_kind: PhantomData,
                data: media.data.into(),
                media_type: media.media_type.into(),
                quality: Nothing,
            })
        }
    }

    #[derive(Debug, Clone)]
    pub struct Document<Data = Nothing> {
        pub data: Data,
    }

    // Tool-specific types
    #[derive(Debug, Clone)]
    pub struct ToolFunction {
        pub name: String,
        pub arguments: serde_json::Value,
    }

    #[derive(Clone, Debug)]
    pub struct ToolCall {
        pub name: String,
        pub function: ToolFunction,
    }

    #[derive(Clone, Debug)]
    pub struct ToolResult<Data = Nothing, CallId = Nothing> {
        pub data: Data,
        pub call_id: CallId,
    }

    #[derive(Clone, Debug)]
    pub struct Reasoning<Data = Nothing, Signature = Nothing, Id = Nothing> {
        pub data: Data,
        pub signature: Signature,
        pub id: Id,
    }

    // Additional From implementations for Nothing -> String types
    impl From<Nothing> for String {
        fn from(_: Nothing) -> Self {
            String::new()
        }
    }

    impl From<Nothing> for Option<String> {
        fn from(_: Nothing) -> Self {
            None
        }
    }

    // Conversions for Text
    impl From<Text> for Part {
        fn from(text: Text) -> Part {
            Part::Text(text)
        }
    }

    // Conversions for ToolCall
    impl From<ToolCall> for Part {
        fn from(tool_call: ToolCall) -> Part {
            Part::ToolCall(tool_call)
        }
    }

    // Conversions for ToolResult
    impl<D, C> From<ToolResult<D, C>> for Part
    where
        D: Into<String>,
        C: Into<String>,
    {
        fn from(tool_result: ToolResult<D, C>) -> Part {
            Part::ToolResult(ToolResult {
                data: tool_result.data.into(),
                call_id: tool_result.call_id.into(),
            })
        }
    }

    // Conversions for Reasoning
    impl<D, S, I> From<Reasoning<D, S, I>> for Part
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
        ) -> Part {
            Part::Reasoning(Reasoning {
                data: data.into(),
                signature: signature.into(),
                id: id.into(),
            })
        }
    }

    #[derive(Debug, Clone)]
    pub enum Part {
        Text(Text),
        Image(Media<Image, encoding::AnyEncoding, ImageMediaType, Quality>),
        Audio(Media<Audio, encoding::AnyEncoding, AudioMediaType>),
        Video(Media<Video, encoding::AnyEncoding, VideoMediaType>),
        Document,
        ToolCall(ToolCall),
        // Really this should only be able to be like text or an image
        // or maybe something we can extend as providers potentially add tool call capabilities
        ToolResult(ToolResult<String, String>),
        Reasoning(Reasoning<String, String, String>),
    }
}

// Common encoding types for all media
pub mod encoding {
    #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Default)]
    pub struct Base64(pub String);

    impl<S> From<S> for Base64
    where
        S: Into<String>,
    {
        fn from(value: S) -> Self {
            Base64(value.into())
        }
    }

    #[derive(Debug, Clone, PartialEq, Eq, Default)]
    pub struct Uri(pub http::Uri);

    impl From<http::Uri> for Uri {
        fn from(value: http::Uri) -> Self {
            Uri(value)
        }
    }

    #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Default)]
    pub struct Raw(pub Vec<u8>);

    impl From<Vec<u8>> for Raw {
        fn from(value: Vec<u8>) -> Self {
            Raw(value)
        }
    }

    #[derive(Debug, Clone)]
    pub enum AnyEncoding {
        Base64(Base64),
        Uri(Uri),
        Raw(Raw),
    }

    impl From<Base64> for AnyEncoding {
        fn from(value: Base64) -> Self {
            Self::Base64(value)
        }
    }

    impl From<Raw> for AnyEncoding {
        fn from(value: Raw) -> Self {
            Self::Raw(value)
        }
    }

    impl From<Uri> for AnyEncoding {
        fn from(value: Uri) -> Self {
            Self::Uri(value)
        }
    }
}

// Implementation patterns for each content type

// Text builders
impl Default for content::Text {
    fn default() -> Self {
        Self {
            text: String::from(""),
        }
    }
}

impl content::Text {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn text(self, text: impl Into<String>) -> content::Text {
        content::Text { text: text.into() }
    }
}

// Reasoning builders
impl Default for content::Reasoning {
    fn default() -> Self {
        Self {
            data: Nothing,
            signature: Nothing,
            id: Nothing,
        }
    }
}

impl content::Reasoning {
    pub fn new() -> Self {
        Self::default()
    }
}

impl<S> content::Reasoning<Nothing, S> {
    pub fn text(self, text: impl Into<String>) -> content::Reasoning<String, S> {
        content::Reasoning {
            data: text.into(),
            signature: self.signature,
            id: self.id,
        }
    }
}

impl<D> content::Reasoning<D, Nothing> {
    pub fn signature(self, sig: String) -> content::Reasoning<D, String> {
        content::Reasoning {
            data: self.data,
            signature: sig,
            id: self.id,
        }
    }
}

/// Capability trait: model supports content kind K for role R
pub trait Supports<Kind, Role> {}

#[repr(transparent)]
#[derive(Debug, Clone)]
pub struct Message<C: NonEmpty> {
    __content: PhantomData<C>,
    parts: OneOrMany<content::Part>,
}

#[repr(transparent)]
#[derive(Debug, Clone)]
pub struct MessageBuilder<Content: List = Nil> {
    __content: PhantomData<fn() -> Content>,
    parts: Vec<content::Part>,
}

impl MessageBuilder<Nil> {
    pub fn new() -> Self {
        Self {
            __content: PhantomData,
            parts: Default::default(),
        }
    }
}

impl<C> MessageBuilder<C>
where
    C: NonEmpty,
{
    pub fn build(self) -> Message<C> {
        Message {
            __content: PhantomData,
            // SAFETY: This is safe because the `NonEmpty` bound guarantees something has been
            // cons'd onto the type-level list tracking the content of this message
            parts: unsafe { OneOrMany::unsafe_from_vec(self.parts) },
        }
    }
}

impl<C> MessageBuilder<C>
where
    C: List,
{
    pub fn text(mut self, text: impl Into<String>) -> MessageBuilder<Cons<content::Text, C>> {
        self.parts
            .push(content::Part::Text(content::Text { text: text.into() }));

        MessageBuilder {
            __content: PhantomData,
            parts: self.parts,
        }
    }
    pub fn audio<E, MT>(
        mut self,
        audio: impl Into<content::Media<content::Audio, E, MT>>,
    ) -> MessageBuilder<Cons<content::Media<content::Audio, E, MT>, C>>
    where
        E: Into<AnyEncoding>,
        MT: Into<content::AudioMediaType>,
    {
        let part: content::Part = audio.into().into();

        self.parts.push(part);

        MessageBuilder {
            __content: PhantomData,
            parts: self.parts,
        }
    }

    pub fn image<E, MT, Q>(
        mut self,
        image: content::Media<content::Image, E, MT, Q>,
    ) -> MessageBuilder<Cons<content::Media<content::Image, E, MT, Q>, C>>
    where
        E: Into<AnyEncoding>,
        MT: Into<content::ImageMediaType>,
        Q: Into<content::Quality>,
    {
        let part: content::Part = image.into();

        self.parts.push(part);

        MessageBuilder {
            __content: PhantomData,
            parts: self.parts,
        }
    }
}

mod supports {
    use std::marker::PhantomData;

    use crate::type_level::list::{Cons, List, Nil};

    pub struct SupportsContent<Role>(PhantomData<Role>);

    pub trait Satisfies<Rel, Content> {}
    impl<Model, Content, Role> Satisfies<SupportsContent<Role>, Content> for Model where
        Model: super::Supports<Content, Role>
    {
    }

    pub trait SatisfiesAll<Rel, L: List> {}

    impl<M, R> SatisfiesAll<R, Nil> for M {}

    impl<M, R, Head, Tail> SatisfiesAll<R, Cons<Head, Tail>> for M
    where
        M: Satisfies<R, Head>,
        M: SatisfiesAll<R, Tail>,
        Tail: List,
    {
    }
}

mod example {
    use crate::completion::typed_message_part_two::supports::SupportsContent;

    use super::*;
    use serde::{Deserialize, Serialize};
    use std::convert::Infallible;
    use supports::SatisfiesAll;

    // Our model
    #[derive(Clone, Debug, Serialize, Deserialize)]
    pub struct Claude;

    type ClaudeImage =
        content::Media<content::Image, encoding::Base64, content::ImageMediaType, content::Quality>;

    impl<R> Supports<content::Text, R> for Claude {}
    impl Supports<ClaudeImage, role::User> for Claude {}

    impl Claude {
        pub fn new() -> Self {
            Self
        }

        pub fn send<C: NonEmpty>(self, _message: Message<C>) -> Result<(), Infallible>
        where
            Self: SatisfiesAll<SupportsContent<role::User>, C>,
        {
            Ok(())
        }
    }

    #[derive(Clone, Debug, Serialize, Deserialize)]
    pub struct OpenAI;

    impl<R> Supports<content::Text, R> for OpenAI {}

    impl OpenAI {
        pub fn new() -> Self {
            Self
        }

        fn send<C: NonEmpty>(&self, _message: Message<C>) -> Result<(), Infallible>
        where
            Self: SatisfiesAll<SupportsContent<role::User>, C>,
        {
            Ok(())
        }
    }

    #[cfg(test)]
    mod tests {
        use super::content::{ImageMediaType, Media, Quality};

        use super::*;

        #[test]
        fn by_construction() {
            let claude = Claude::new();
            let openai = OpenAI::new();

            let message = MessageBuilder::new()
                .text("Hello World")
                .image(
                    Media::image()
                        .base64("some_base64_data")
                        .media_type(ImageMediaType::JPEG)
                        .quality(Quality::Low),
                )
                .build();

            claude.send(message).unwrap();
            // Doesn't compile
            //openai.send(message).unwrap();
        }
    }
}
