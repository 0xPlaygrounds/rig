use super::content;
use super::encoding::{self, AnyEncoding};
use crate::{
    OneOrMany,
    nothing::Nothing,
    type_level::list::{Cons, List, Nil, NonEmpty},
};
use std::marker::PhantomData;

// ROLES
pub mod role {
    #[derive(Debug, Clone, Copy)]
    pub struct User;

    #[derive(Debug, Clone, Copy)]
    pub struct Assistant;
}

pub trait Label {
    const LABEL: &str;
}

pub trait GetRoleContent {
    type Output: std::fmt::Debug + Clone;
}

pub type Content<T> = <T as GetRoleContent>::Output;

impl GetRoleContent for role::User {
    type Output = content::UserPart;
}

impl GetRoleContent for role::Assistant {
    type Output = content::AssistantPart;
}

// Text builders

impl Default for content::Text {
    fn default() -> Self {
        Self {
            text: String::from(""),
        }
    }
}

impl content::Text {
    pub fn empty() -> Self {
        Self::default()
    }

    pub fn text(self, text: impl Into<String>) -> content::Text {
        content::Text { text: text.into() }
    }

    pub fn concat(mut self, rhs: impl Into<Self>) -> Self {
        self.text.push_str(rhs.into().text.as_str());

        self
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
pub struct Message<Role, ContentTypes>
where
    Role: GetRoleContent,
    ContentTypes: NonEmpty,
{
    __role: PhantomData<Role>,
    __content: PhantomData<ContentTypes>,
    parts: OneOrMany<Content<Role>>,
}

#[repr(transparent)]
#[derive(Debug, Clone)]
pub struct MessageBuilder<Role, ContentTypes = Nil>
where
    Role: GetRoleContent,
    ContentTypes: List,
{
    __role: PhantomData<Role>,
    __content: PhantomData<fn() -> ContentTypes>,
    parts: Vec<Content<Role>>,
}

impl<R> Default for MessageBuilder<R, Nil>
where
    R: GetRoleContent,
{
    fn default() -> Self {
        Self {
            __role: PhantomData,
            __content: PhantomData,
            parts: Default::default(),
        }
    }
}

impl<R> MessageBuilder<R, Nil>
where
    R: GetRoleContent,
{
    pub fn new() -> Self {
        Self {
            __role: PhantomData,
            __content: PhantomData,
            parts: Default::default(),
        }
    }
}

impl<R, C> MessageBuilder<R, C>
where
    R: GetRoleContent,
    C: NonEmpty,
{
    pub fn build(self) -> Message<R, C> {
        Message {
            __role: PhantomData,
            __content: PhantomData,
            // SAFETY: This is safe because the `NonEmpty` bound guarantees something has been
            // cons'd onto the type-level list tracking the content of this message
            parts: unsafe { OneOrMany::unsafe_from_vec(self.parts) },
        }
    }
}

// MEDIA
impl<R, C> MessageBuilder<R, C>
where
    R: GetRoleContent,
    Content<R>: From<String>,
    C: List,
{
    pub fn text(mut self, text: impl Into<String>) -> MessageBuilder<R, Cons<content::Text, C>> {
        self.parts.push(R::Output::from(text.into()));

        MessageBuilder {
            parts: self.parts,
            __role: PhantomData,
            __content: PhantomData,
        }
    }

    pub fn audio<E, MT>(
        mut self,
        audio: impl Into<content::Media<content::Audio, E, MT>>,
    ) -> MessageBuilder<R, Cons<content::Media<content::Audio, E, MT>, C>>
    where
        E: Into<AnyEncoding>,
        MT: Into<content::AudioMediaType>,
        Content<R>: From<content::Media<content::Audio, E, MT>>,
    {
        let part: Content<R> = audio.into().into();

        self.parts.push(part);

        MessageBuilder {
            __content: PhantomData,
            __role: PhantomData,
            parts: self.parts,
        }
    }

    pub fn image<E, MT, Q>(
        mut self,
        image: impl Into<content::Media<content::Image, E, MT, Q>>,
    ) -> MessageBuilder<R, Cons<content::Media<content::Image, E, MT, Q>, C>>
    where
        E: Into<AnyEncoding>,
        MT: Into<content::ImageMediaType>,
        Q: Into<content::Quality>,
        Content<R>: From<content::Media<content::Image, E, MT, Q>>,
    {
        let part: Content<R> = image.into().into();

        self.parts.push(part);

        MessageBuilder {
            __content: PhantomData,
            __role: PhantomData,
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

        pub fn send<C: NonEmpty>(&self, _message: Message<role::User, C>) -> Result<(), Infallible>
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

        fn send<C: NonEmpty>(&self, _message: Message<role::User, C>) -> Result<(), Infallible>
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
                .image(
                    Media::image()
                        .base64("some_base64_data")
                        .media_type(ImageMediaType::JPEG)
                        .quality(Quality::Low),
                )
                .text("Hello World")
                .build();

            claude.send(message).unwrap();
            // Doesn't typecheck because `OpenAI` doesn't impl `Supports<Image, Role>`
            // openai.send(message).unwrap();

            let just_text = MessageBuilder::new().text("Hello GPT").build();

            openai.send(just_text.clone()).unwrap();
            claude.send(just_text).unwrap();
        }
    }
}
