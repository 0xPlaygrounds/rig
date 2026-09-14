//! Lossless conversion between message DTOs and ordered, typed child entities.

use bevy_ecs::prelude::*;
use rig_core::message::{self, AssistantContent, ToolResultContent, UserContent};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use super::binary::{BinaryAssets, BinaryError, PartSource};
use crate::agent::{MessageParts, Order, Role, Utterance};

/// A content entity, owned by an utterance or a tool-result part.
#[derive(Component, Debug, Clone, Copy, Serialize, Deserialize)]
#[cfg_attr(feature = "reflect", derive(bevy_reflect::Reflect), reflect(Component))]
pub struct ContentPart;

/// A request-only edit on an ordered link entity owned by a fresh turn.
/// Text replacement preserves annotations; removal omits the complete target.
/// Persistent history is unchanged. Links are consumed when the request is folded.
#[derive(Component, Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "reflect", derive(bevy_reflect::Reflect), reflect(Component))]
pub enum RequestPartEdit {
    /// Replace the text of a TextPart, preserving its other fields.
    Text(String),
    /// Omit this part (including children of a tool result) from this request.
    Remove,
}

/// Target of a RequestPartEdit link; scene persistence remaps this relationship.
#[derive(Component, Debug, Clone, Copy, PartialEq, Eq)]
#[relationship(relationship_target = EditedBy)]
#[cfg_attr(feature = "reflect", derive(bevy_reflect::Reflect), reflect(Component))]
pub struct EditTarget(pub Entity);

/// Request edit links naming this content part.
#[derive(Component, Debug, Default)]
#[relationship_target(relationship = EditTarget)]
#[cfg_attr(feature = "reflect", derive(bevy_reflect::Reflect), reflect(Component))]
pub struct EditedBy(Vec<Entity>);

/// Provider-assigned assistant message identifier, including explicit absence.
#[derive(Component, Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "reflect", derive(bevy_reflect::Reflect), reflect(Component))]
pub struct MessageId(pub Option<String>);

/// A text part, including provider annotations.
#[derive(Component, Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "reflect", derive(bevy_reflect::Reflect), reflect(Component))]
pub struct TextPart(
    #[cfg_attr(feature = "reflect", reflect(remote = super::reflect::TextPartReflect))]
    pub  message::Text,
);

/// A tool call with correlation IDs, arguments, signature and metadata.
#[derive(Component, Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "reflect", derive(bevy_reflect::Reflect), reflect(Component))]
pub struct ToolCallPart(
    #[cfg_attr(feature = "reflect", reflect(remote = super::reflect::ToolCallPartReflect))]
    pub  message::ToolCall,
);

/// Ordered reasoning content with IDs, signatures and opaque provider data.
#[derive(Component, Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "reflect", derive(bevy_reflect::Reflect), reflect(Component))]
pub struct ReasoningPart(
    #[cfg_attr(feature = "reflect", reflect(remote = super::reflect::ReasoningPartReflect))]
    pub  message::Reasoning,
);

/// A structured JSON item under a tool-result part; never implicitly parsed from text.
#[derive(Component, Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "reflect", derive(bevy_reflect::Reflect), reflect(Component))]
pub struct JsonPart(
    #[cfg_attr(feature = "reflect", reflect(remote = super::reflect::JsonPartReflect))]
    pub  serde_json::Value,
);

/// A image content part, retaining per-use metadata beside its shared source.
#[derive(Component, Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "reflect", derive(bevy_reflect::Reflect), reflect(Component))]
pub struct ImagePart {
    /// Inline source metadata or a reference to shared binary bytes.
    pub source: PartSource,
    /// This occurrence's media type.
    #[cfg_attr(feature = "reflect", reflect(remote = super::reflect::ImageMediaReflect))]
    pub media_type: Option<message::ImageMediaType>,
    /// This occurrence's provider-specific metadata.
    #[cfg_attr(feature = "reflect", reflect(remote = super::reflect::PartParamsReflect))]
    pub additional_params: Option<message::AdditionalParams>,
    /// Provider rendering preference for this occurrence.
    #[cfg_attr(feature = "reflect", reflect(remote = super::reflect::ImageDetailReflect))]
    pub detail: Option<message::ImageDetail>,
}

/// A audio content part, retaining per-use metadata beside its shared source.
#[derive(Component, Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "reflect", derive(bevy_reflect::Reflect), reflect(Component))]
pub struct AudioPart {
    /// Inline source metadata or a reference to shared binary bytes.
    pub source: PartSource,
    /// This occurrence's media type.
    #[cfg_attr(feature = "reflect", reflect(remote = super::reflect::AudioMediaReflect))]
    pub media_type: Option<message::AudioMediaType>,
    /// This occurrence's provider-specific metadata.
    #[cfg_attr(feature = "reflect", reflect(remote = super::reflect::PartParamsReflect))]
    pub additional_params: Option<message::AdditionalParams>,
}

/// A video content part, retaining per-use metadata beside its shared source.
#[derive(Component, Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "reflect", derive(bevy_reflect::Reflect), reflect(Component))]
pub struct VideoPart {
    /// Inline source metadata or a reference to shared binary bytes.
    pub source: PartSource,
    /// This occurrence's media type.
    #[cfg_attr(feature = "reflect", reflect(remote = super::reflect::VideoMediaReflect))]
    pub media_type: Option<message::VideoMediaType>,
    /// This occurrence's provider-specific metadata.
    #[cfg_attr(feature = "reflect", reflect(remote = super::reflect::PartParamsReflect))]
    pub additional_params: Option<message::AdditionalParams>,
}

/// A document content part, retaining per-use metadata beside its shared source.
#[derive(Component, Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "reflect", derive(bevy_reflect::Reflect), reflect(Component))]
pub struct DocumentPart {
    /// Inline source metadata or a reference to shared binary bytes.
    pub source: PartSource,
    /// This occurrence's media type.
    #[cfg_attr(feature = "reflect", reflect(remote = super::reflect::DocumentMediaReflect))]
    pub media_type: Option<message::DocumentMediaType>,
    /// This occurrence's provider-specific metadata.
    #[cfg_attr(feature = "reflect", reflect(remote = super::reflect::PartParamsReflect))]
    pub additional_params: Option<message::AdditionalParams>,
}

/// Tool-result identity; its ordered children are TextPart, ImagePart or JsonPart.
#[derive(Component, Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "reflect", derive(bevy_reflect::Reflect), reflect(Component))]
pub struct ToolResultPart {
    /// The call answered by this result.
    #[cfg_attr(feature = "reflect", reflect(remote = crate::agent::reflect::ToolCallIdReflect))]
    pub call: message::ToolCallId,
    /// Original provider call identifiers.
    #[cfg_attr(feature = "reflect", reflect(remote = crate::agent::reflect::ProviderCallIdReflect))]
    pub provider: Option<message::ProviderCallId>,
    /// Executed tool name, including hook repairs.
    pub name: String,
}

/// Why a graph cannot be converted to a transport message.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, thiserror::Error)]
pub enum ContentError {
    /// Binary conversion failed.
    #[error(transparent)]
    Binary(#[from] BinaryError),
    /// An entity or required component is missing.
    #[error("content graph is missing an entity or required component")]
    Missing,
    /// A part has conflicting types, an invalid parent, or the wrong role.
    #[error("content part has an invalid type or role")]
    Shape,
    /// Two siblings share an order, which cannot round-trip unambiguously.
    #[error("content siblings have duplicate orders")]
    DuplicateOrder,
}

// Transient conversion plan. This is never a component or persisted alongside
// the graph; planning before spawning avoids partially written utterances.
pub(crate) enum PartValue {
    Text(TextPart),
    Image(ImagePart),
    Audio(AudioPart),
    Video(VideoPart),
    Document(DocumentPart),
    Call(ToolCallPart),
    Reasoning(ReasoningPart),
    Result(ToolResultPart, Vec<PartValue>),
    Json(JsonPart),
}

fn image(assets: &mut BinaryAssets, value: message::Image) -> Result<ImagePart, ContentError> {
    Ok(ImagePart {
        source: assets.intern(value.data)?,
        media_type: value.media_type,
        detail: value.detail,
        additional_params: value.additional_params,
    })
}

fn user(assets: &mut BinaryAssets, value: UserContent) -> Result<PartValue, ContentError> {
    Ok(match value {
        UserContent::Text(value) => PartValue::Text(TextPart(value)),
        UserContent::Image(value) => PartValue::Image(image(assets, value)?),
        UserContent::Audio(value) => PartValue::Audio(AudioPart {
            source: assets.intern(value.data)?,
            media_type: value.media_type,
            additional_params: value.additional_params,
        }),
        UserContent::Video(value) => PartValue::Video(VideoPart {
            source: assets.intern(value.data)?,
            media_type: value.media_type,
            additional_params: value.additional_params,
        }),
        UserContent::Document(value) => PartValue::Document(DocumentPart {
            source: assets.intern(value.data)?,
            media_type: value.media_type,
            additional_params: value.additional_params,
        }),
        UserContent::ToolResult(value) => {
            let children = value
                .content
                .into_iter()
                .map(|item| {
                    Ok(match item {
                        ToolResultContent::Text(text) => PartValue::Text(TextPart(text)),
                        ToolResultContent::Image(value) => PartValue::Image(image(assets, value)?),
                        ToolResultContent::Json { value } => PartValue::Json(JsonPart(value)),
                    })
                })
                .collect::<Result<Vec<_>, ContentError>>()?;
            PartValue::Result(
                ToolResultPart {
                    call: value.call,
                    provider: value.provider,
                    name: value.name,
                },
                children,
            )
        }
    })
}

fn prepare(
    assets: &mut BinaryAssets,
    parts: MessageParts,
) -> Result<(Role, Option<MessageId>, Vec<PartValue>), ContentError> {
    Ok(match parts {
        MessageParts::User { content } => (
            Role::User,
            None,
            content
                .into_iter()
                .map(|part| user(assets, part))
                .collect::<Result<_, _>>()?,
        ),
        MessageParts::Assistant { id, content } => (
            Role::Assistant,
            Some(MessageId(id)),
            content
                .into_iter()
                .map(|part| {
                    Ok(match part {
                        AssistantContent::Text(value) => PartValue::Text(TextPart(value)),
                        AssistantContent::Image(value) => PartValue::Image(image(assets, value)?),
                        AssistantContent::ToolCall(value) => PartValue::Call(ToolCallPart(value)),
                        AssistantContent::Reasoning(value) => {
                            PartValue::Reasoning(ReasoningPart(value))
                        }
                    })
                })
                .collect::<Result<_, ContentError>>()?,
        ),
    })
}

fn spawn_parts(world: &mut World, parent: Entity, parts: Vec<PartValue>) {
    for (order, part) in parts.into_iter().enumerate() {
        let mut entity = world.spawn((ContentPart, Order(order as u64), ChildOf(parent)));
        match part {
            PartValue::Text(value) => {
                entity.insert(value);
            }
            PartValue::Image(value) => {
                entity.insert(value);
            }
            PartValue::Audio(value) => {
                entity.insert(value);
            }
            PartValue::Video(value) => {
                entity.insert(value);
            }
            PartValue::Document(value) => {
                entity.insert(value);
            }
            PartValue::Call(value) => {
                entity.insert(value);
            }
            PartValue::Reasoning(value) => {
                entity.insert(value);
            }
            PartValue::Json(value) => {
                entity.insert(value);
            }
            PartValue::Result(value, children) => {
                let id = entity.insert(value).id();
                spawn_parts(world, id, children);
            }
        }
    }
}

/// Replace an utterance's content graph. This is a persistent edit; request-only
/// steering must patch the reconstructed request instead. A rejected source
/// leaves the existing graph unchanged; unreferenced interned assets can be collected.
/// Sibling Order values are local, contiguous indices and do not consume the
/// run/utterance OrderCounter. Repeated identical parts remain distinct entities.
pub fn write_message(
    world: &mut World,
    utterance: Entity,
    parts: MessageParts,
) -> Result<(), ContentError> {
    if world.get::<Utterance>(utterance).is_none() {
        return Err(ContentError::Missing);
    }
    world.init_resource::<BinaryAssets>();
    let (role, id, parts) = prepare(&mut world.resource_mut::<BinaryAssets>(), parts)?;
    let old: Vec<_> = world
        .get::<Children>(utterance)
        .map(|children| children.iter().collect())
        .unwrap_or_default();
    for child in old {
        world.despawn(child);
    }
    let mut entity = world.entity_mut(utterance);
    entity.insert(role).remove::<MessageId>();
    if let Some(id) = id {
        entity.insert(id);
    }
    spawn_parts(world, utterance, parts);
    Ok(())
}

fn ordered<'a>(
    get: &impl Fn(Entity) -> Option<EntityRef<'a>>,
    parent: Entity,
) -> Result<Vec<Entity>, ContentError> {
    let parent = get(parent).ok_or(ContentError::Missing)?;
    let children = parent.get::<Children>();
    let mut ordered = Vec::new();
    for child in children.into_iter().flat_map(|c| c.iter()) {
        let entity = get(child).ok_or(ContentError::Missing)?;
        if entity.get::<ContentPart>().is_none() {
            return Err(ContentError::Shape);
        }
        let order = entity.get::<Order>().ok_or(ContentError::Missing)?;
        ordered.push((*order, child));
    }
    ordered.sort_by_key(|(order, _)| *order);
    if ordered
        .windows(2)
        .any(|pair| pair.first().map(|x| x.0) == pair.get(1).map(|x| x.0))
    {
        return Err(ContentError::DuplicateOrder);
    }
    Ok(ordered.into_iter().map(|(_, entity)| entity).collect())
}

fn read_image(assets: &BinaryAssets, value: &ImagePart) -> Result<message::Image, ContentError> {
    Ok(message::Image {
        data: assets.resolve(&value.source)?,
        media_type: value.media_type.clone(),
        detail: value.detail.clone(),
        additional_params: value.additional_params.clone(),
    })
}

fn read_part<'a>(
    get: &impl Fn(Entity) -> Option<EntityRef<'a>>,
    entity: Entity,
    nested: bool,
) -> Result<PartValue, ContentError> {
    let entity_ref = get(entity).ok_or(ContentError::Missing)?;
    let mut values = Vec::new();
    macro_rules! take {
        ($ty:ty, $variant:ident) => {
            if let Some(value) = entity_ref.get::<$ty>() {
                values.push(PartValue::$variant(value.clone()));
            }
        };
    }
    take!(TextPart, Text);
    take!(ImagePart, Image);
    take!(AudioPart, Audio);
    take!(VideoPart, Video);
    take!(DocumentPart, Document);
    take!(ToolCallPart, Call);
    take!(ReasoningPart, Reasoning);
    take!(JsonPart, Json);
    if let Some(value) = entity_ref.get::<ToolResultPart>() {
        if nested {
            return Err(ContentError::Shape);
        }
        let children = ordered(get, entity)?
            .into_iter()
            .map(|child| read_part(get, child, true))
            .collect::<Result<_, _>>()?;
        values.push(PartValue::Result(value.clone(), children));
    } else if entity_ref
        .get::<Children>()
        .is_some_and(|children| !children.is_empty())
    {
        return Err(ContentError::Shape);
    }
    if values.len() != 1 {
        return Err(ContentError::Shape);
    }
    values.pop().ok_or(ContentError::Missing)
}

fn read_edited_part<'a>(
    get: &impl Fn(Entity) -> Option<EntityRef<'a>>,
    entity: Entity,
    nested: bool,
    edits: &BTreeMap<Entity, RequestPartEdit>,
) -> Result<Option<PartValue>, ContentError> {
    let mut value = read_part(get, entity, nested)?;
    if edits.is_empty() {
        return Ok(Some(value));
    }
    match edits.get(&entity) {
        Some(RequestPartEdit::Remove) => return Ok(None),
        Some(RequestPartEdit::Text(text)) => match &mut value {
            PartValue::Text(part) => part.0.text.clone_from(text),
            _ => return Err(ContentError::Shape),
        },
        None => {}
    }
    if let PartValue::Result(_, children) = &mut value {
        *children = ordered(get, entity)?
            .into_iter()
            .map(|child| read_edited_part(get, child, true, edits))
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .flatten()
            .collect();
    }
    Ok(Some(value))
}

fn to_user(assets: &BinaryAssets, value: PartValue) -> Result<UserContent, ContentError> {
    Ok(match value {
        PartValue::Text(value) => UserContent::Text(value.0),
        PartValue::Image(value) => UserContent::Image(read_image(assets, &value)?),
        PartValue::Audio(value) => UserContent::Audio(message::Audio {
            data: assets.resolve(&value.source)?,
            media_type: value.media_type,
            additional_params: value.additional_params,
        }),
        PartValue::Video(value) => UserContent::Video(message::Video {
            data: assets.resolve(&value.source)?,
            media_type: value.media_type,
            additional_params: value.additional_params,
        }),
        PartValue::Document(value) => UserContent::Document(message::Document {
            data: assets.resolve(&value.source)?,
            media_type: value.media_type,
            additional_params: value.additional_params,
        }),
        PartValue::Result(value, children) => UserContent::ToolResult(message::ToolResult {
            call: value.call,
            provider: value.provider,
            name: value.name,
            content: children
                .into_iter()
                .map(|part| {
                    Ok(match part {
                        PartValue::Text(value) => ToolResultContent::Text(value.0),
                        PartValue::Image(value) => {
                            ToolResultContent::Image(read_image(assets, &value)?)
                        }
                        PartValue::Json(value) => ToolResultContent::Json { value: value.0 },
                        _ => return Err(ContentError::Shape),
                    })
                })
                .collect::<Result<_, ContentError>>()?,
        }),
        _ => return Err(ContentError::Shape),
    })
}

/// Reconstruct a message from its role, ID and typed children. Invalid types,
/// missing components, duplicate sibling orders and unresolved assets are errors.
pub fn read_message(world: &World, utterance: Entity) -> Result<MessageParts, ContentError> {
    let empty = BinaryAssets::default();
    let assets = world.get_resource::<BinaryAssets>().unwrap_or(&empty);
    read_message_from(
        &|entity| world.get_entity(entity).ok(),
        assets,
        utterance,
        &BTreeMap::new(),
    )
}

/// Read-only system access to the content graph. It borrows graph components;
/// DTOs are built only when `message` is called at a conversion boundary.
#[derive(bevy_ecs::system::SystemParam)]
pub struct ContentGraph<'w, 's> {
    entities: Query<'w, 's, EntityRef<'static>, Without<bevy_ecs::resource::IsResource>>,
    assets: Option<Res<'w, BinaryAssets>>,
}

impl ContentGraph<'_, '_> {
    /// Read one message with transient part edits. The caller validates ownership.
    pub fn message_with(
        &self,
        utterance: Entity,
        edits: &BTreeMap<Entity, RequestPartEdit>,
    ) -> Result<MessageParts, ContentError> {
        if edits.is_empty() {
            return self.message(utterance);
        }
        // Validate the unedited message too: removal cannot conceal corrupt data.
        self.message(utterance)?;
        let empty = BinaryAssets::default();
        read_message_from(
            &|entity| self.entities.get(entity).ok(),
            self.assets.as_deref().unwrap_or(&empty),
            utterance,
            edits,
        )
    }

    /// Find the utterance owning a direct or nested tool-result content part.
    pub fn target_utterance(&self, target: Entity) -> Result<Entity, ContentError> {
        let part = self
            .entities
            .get(target)
            .map_err(|_| ContentError::Missing)?;
        if !part.contains::<ContentPart>() {
            return Err(ContentError::Shape);
        }
        let parent = part.get::<ChildOf>().ok_or(ContentError::Missing)?.parent();
        let owner = self
            .entities
            .get(parent)
            .map_err(|_| ContentError::Missing)?;
        if owner.contains::<Utterance>() {
            return Ok(parent);
        }
        if !owner.contains::<ToolResultPart>() {
            return Err(ContentError::Shape);
        }
        let utterance = owner
            .get::<ChildOf>()
            .ok_or(ContentError::Missing)?
            .parent();
        if !self
            .entities
            .get(utterance)
            .map_err(|_| ContentError::Missing)?
            .contains::<Utterance>()
        {
            return Err(ContentError::Shape);
        }
        Ok(utterance)
    }

    /// Reconstruct one utterance, validating ordering, role and binary references.
    pub fn message(&self, utterance: Entity) -> Result<MessageParts, ContentError> {
        let empty = BinaryAssets::default();
        let assets = self.assets.as_deref().unwrap_or(&empty);
        read_message_from(
            &|entity| self.entities.get(entity).ok(),
            assets,
            utterance,
            &BTreeMap::new(),
        )
    }
}

fn read_message_from<'a>(
    get: &impl Fn(Entity) -> Option<EntityRef<'a>>,
    assets: &BinaryAssets,
    utterance: Entity,
    edits: &BTreeMap<Entity, RequestPartEdit>,
) -> Result<MessageParts, ContentError> {
    let entity = get(utterance).ok_or(ContentError::Missing)?;
    if entity.get::<Utterance>().is_none() {
        return Err(ContentError::Missing);
    }
    let role = entity.get::<Role>().ok_or(ContentError::Missing)?;
    let values = ordered(get, utterance)?
        .into_iter()
        .map(|entity| read_edited_part(get, entity, false, edits))
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
    match role {
        Role::User => {
            if entity.get::<MessageId>().is_some() {
                return Err(ContentError::Shape);
            }
            Ok(MessageParts::User {
                content: values
                    .into_iter()
                    .map(|value| to_user(assets, value))
                    .collect::<Result<_, _>>()?,
            })
        }
        Role::Assistant => Ok(MessageParts::Assistant {
            id: entity
                .get::<MessageId>()
                .ok_or(ContentError::Missing)?
                .0
                .clone(),
            content: values
                .into_iter()
                .map(|value| {
                    Ok(match value {
                        PartValue::Text(value) => AssistantContent::Text(value.0),
                        PartValue::Image(value) => {
                            AssistantContent::Image(read_image(assets, &value)?)
                        }
                        PartValue::Call(value) => AssistantContent::ToolCall(value.0),
                        PartValue::Reasoning(value) => AssistantContent::Reasoning(value.0),
                        _ => return Err(ContentError::Shape),
                    })
                })
                .collect::<Result<_, ContentError>>()?,
        }),
    }
}

/// Collect payloads unreachable from every content entity in the world and the
/// host's explicit pins. Scanning all owners preserves assets shared by runs,
/// forks and conversation subtrees. A missing root refuses collection unchanged.
pub fn collect_binary_assets(
    world: &mut World,
    pins: impl IntoIterator<Item = super::binary::BinaryId>,
) -> Result<(), BinaryError> {
    let mut roots: Vec<_> = pins.into_iter().collect();
    let mut query = world.query::<(
        Option<&ImagePart>,
        Option<&AudioPart>,
        Option<&VideoPart>,
        Option<&DocumentPart>,
    )>();
    for (image, audio, video, document) in query.iter(world) {
        for source in [
            image.map(|p| &p.source),
            audio.map(|p| &p.source),
            video.map(|p| &p.source),
            document.map(|p| &p.source),
        ]
        .into_iter()
        .flatten()
        {
            if let PartSource::Binary { id, .. } = source {
                roots.push(*id);
            }
        }
    }
    match world.get_resource_mut::<BinaryAssets>() {
        Some(mut assets) => assets.retain(roots),
        None if roots.is_empty() => Ok(()),
        None => Err(BinaryError::Missing),
    }
}

/// Prepare content before queuing any graph mutations. Deferred application
/// creates all children before the next system boundary can assemble the run.
pub(crate) fn spawn_deferred(
    commands: &mut Commands,
    assets: &mut BinaryAssets,
    parent: Entity,
    message: MessageParts,
    order: Order,
) -> Result<Entity, ContentError> {
    let (role, id, parts) = prepare(assets, message)?;
    let mut utterance = commands.spawn((Utterance, role, order, ChildOf(parent)));
    if let Some(id) = id {
        utterance.insert(id);
    }
    let entity = utterance.id();
    commands.queue(move |world: &mut World| spawn_parts(world, entity, parts));
    Ok(entity)
}

pub(crate) fn replace_deferred(
    commands: &mut Commands,
    assets: &mut BinaryAssets,
    utterance: Entity,
    message: MessageParts,
) -> Result<(), ContentError> {
    let (role, id, parts) = prepare(assets, message)?;
    commands.queue(move |world: &mut World| {
        if world.get::<Utterance>(utterance).is_none() {
            return;
        }
        let old: Vec<_> = world
            .get::<Children>(utterance)
            .map(|children| children.iter().collect())
            .unwrap_or_default();
        for child in old {
            world.despawn(child);
        }
        let mut entity = world.entity_mut(utterance);
        entity.insert(role).remove::<MessageId>();
        if let Some(id) = id {
            entity.insert(id);
        }
        spawn_parts(world, utterance, parts);
    });
    Ok(())
}
