//! Lossless conversion between message DTOs and ordered, typed child entities.
//!
//! ```
//! use rig_ecs::agent::{MessageParts, Utterance, content::parts::{read_message, write_message}};
//! let mut world = bevy_ecs::world::World::new();
//! let utterance = world.spawn(Utterance).id();
//! write_message(&mut world, utterance, MessageParts::User { content: vec![] })?;
//! let message = read_message(&world, utterance)?;
//! # Ok::<(), rig_ecs::agent::content::parts::ContentError>(())
//! ```

use bevy_ecs::prelude::*;
use bevy_reflect::Reflect;
use rig_core::message::{self, AssistantContent, ToolResultContent, UserContent};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use super::binary::{BinaryAssets, BinaryError, PartSource};
use crate::agent::{MessageParts, Role, Utterance};

/// The payload of a content entity, owned by an utterance or a tool result.
/// Tool-result items remain ordered child entities, not fields of this component.
#[derive(Component, Debug, Clone, PartialEq, Serialize, Deserialize, Reflect)]
#[reflect(Component)]
pub enum ContentPart {
    /// Text, including provider annotations.
    Text(#[reflect(remote = super::reflect::TextPartReflect)] message::Text),
    /// An image with per-use metadata and a shared binary source.
    Image(ImagePart),
    /// Audio with per-use metadata and a shared binary source.
    Audio(AudioPart),
    /// Video with per-use metadata and a shared binary source.
    Video(VideoPart),
    /// A document with per-use metadata and a shared binary source.
    Document(DocumentPart),
    /// A tool call with correlation IDs, arguments, signature and metadata.
    ToolCall(#[reflect(remote = super::reflect::ToolCallPartReflect)] message::ToolCall),
    /// Ordered reasoning with IDs, signatures and opaque provider data.
    Reasoning(#[reflect(remote = super::reflect::ReasoningPartReflect)] message::Reasoning),
    /// A tool result whose children must be Text, Image or Json parts.
    ToolResult {
        /// The call answered by this result.
        #[reflect(remote = crate::agent::reflect::ToolCallIdReflect)]
        call: message::ToolCallId,
        /// Original provider call identifiers.
        #[reflect(remote = crate::agent::reflect::ProviderCallIdReflect)]
        provider: Option<message::ProviderCallId>,
        /// Executed tool name, including hook repairs.
        name: String,
    },
    /// Structured JSON under a tool result; never implicitly parsed from text.
    Json(#[reflect(remote = super::reflect::JsonPartReflect)] serde_json::Value),
}

/// A request-only edit on an ordered link entity owned by a fresh turn.
/// Text replacement preserves annotations; removal omits the complete target.
/// Persistent history is unchanged. Links are consumed when the request is folded.
#[derive(Component, Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Reflect)]
#[reflect(Component)]
pub enum RequestPartEdit {
    /// Replace the text of a [`ContentPart::Text`], preserving its other fields.
    Text(String),
    /// Omit this part (including children of a tool result) from this request.
    Remove,
}

/// Target of a RequestPartEdit link; scene persistence remaps this relationship.
#[derive(Component, Debug, Clone, Copy, PartialEq, Eq, Reflect)]
#[relationship(relationship_target = EditedBy)]
#[reflect(Component)]
pub struct EditTarget(pub Entity);

/// Request edit links naming this content part.
#[derive(Component, Debug, Default, Reflect)]
#[relationship_target(relationship = EditTarget)]
#[reflect(Component)]
pub struct EditedBy(Vec<Entity>);

/// Provider-assigned assistant message identifier, including explicit absence.
#[derive(Component, Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Reflect)]
#[reflect(Component)]
pub struct MessageId(pub Option<String>);

/// Image metadata beside its shared source, held by [`ContentPart::Image`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Reflect)]
pub struct ImagePart {
    /// Inline source metadata or a reference to shared binary bytes.
    pub source: PartSource,
    /// This occurrence's media type.
    #[reflect(remote = super::reflect::ImageMediaReflect)]
    pub media_type: Option<message::ImageMediaType>,
    /// This occurrence's provider-specific metadata.
    #[reflect(remote = super::reflect::PartParamsReflect)]
    pub additional_params: Option<message::AdditionalParams>,
    /// Provider rendering preference for this occurrence.
    #[reflect(remote = super::reflect::ImageDetailReflect)]
    pub detail: Option<message::ImageDetail>,
}

/// Audio metadata beside its shared source, held by [`ContentPart::Audio`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Reflect)]
pub struct AudioPart {
    /// Inline source metadata or a reference to shared binary bytes.
    pub source: PartSource,
    /// This occurrence's media type.
    #[reflect(remote = super::reflect::AudioMediaReflect)]
    pub media_type: Option<message::AudioMediaType>,
    /// This occurrence's provider-specific metadata.
    #[reflect(remote = super::reflect::PartParamsReflect)]
    pub additional_params: Option<message::AdditionalParams>,
}

/// Video metadata beside its shared source, held by [`ContentPart::Video`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Reflect)]
pub struct VideoPart {
    /// Inline source metadata or a reference to shared binary bytes.
    pub source: PartSource,
    /// This occurrence's media type.
    #[reflect(remote = super::reflect::VideoMediaReflect)]
    pub media_type: Option<message::VideoMediaType>,
    /// This occurrence's provider-specific metadata.
    #[reflect(remote = super::reflect::PartParamsReflect)]
    pub additional_params: Option<message::AdditionalParams>,
}

/// Document metadata beside its shared source, held by [`ContentPart::Document`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Reflect)]
pub struct DocumentPart {
    /// Inline source metadata or a reference to shared binary bytes.
    pub source: PartSource,
    /// This occurrence's media type.
    #[reflect(remote = super::reflect::DocumentMediaReflect)]
    pub media_type: Option<message::DocumentMediaType>,
    /// This occurrence's provider-specific metadata.
    #[reflect(remote = super::reflect::PartParamsReflect)]
    pub additional_params: Option<message::AdditionalParams>,
}

/// Tool execution status attached to a [`ContentPart::ToolResult`].
/// Not rendered into transport messages or populated by DTO imports.
#[derive(Component, Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Reflect)]
#[reflect(Component)]
#[serde(rename_all = "snake_case")]
pub enum ToolResultStatus {
    /// The tool ran and answered.
    Ok,
    /// The tool failed (`status: error`), or the bus reported an error
    /// the run goes on from (any report but a denial, a cancel, or one
    /// the run fails on).
    Error,
    /// The tool declined the call (`status: refused`).
    Refused,
    /// Nothing ran: the call was skipped or its result is synthetic feedback.
    Skipped,
    /// A `Denied` outcome (a layer's `deny`, a `Gate` system's denial).
    Denied,
    /// The handler answered with an outcome of another family.
    WrongFamily,
}

/// The default marker of [`ToolResultLimit`]: `{omitted}` is the omitted
/// byte count.
pub const TOOL_RESULT_LIMIT_MARKER: &str = "\n[… {omitted} bytes omitted …]\n";

/// Request-only size limit for tool-result text, resolved from the run then agent.
/// Text exceeding `max_bytes` retains at most that many head and tail bytes,
/// split at UTF-8 boundaries around `marker`; `{omitted}` becomes the omitted
/// byte count. JSON and images remain intact. Applied after request part edits;
/// persisted history and replay identity are unchanged. Absence means no limit.
#[derive(Component, Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Reflect)]
#[reflect(Component)]
pub struct ToolResultLimit {
    /// The most bytes of one text item the request carries besides the marker.
    pub max_bytes: usize,
    /// The marker between the head and the tail; `{omitted}` is the omitted byte count.
    pub marker: String,
}

impl ToolResultLimit {
    /// A limit with the default marker.
    #[must_use]
    pub fn new(max_bytes: usize) -> Self {
        Self {
            max_bytes,
            marker: TOOL_RESULT_LIMIT_MARKER.to_owned(),
        }
    }
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
    /// A part has invalid children, an invalid parent, or the wrong role.
    #[error("content part has an invalid type or role")]
    Shape,
}

// Planning before spawning avoids partially written utterances. Only tool
// results have children, and those children cannot themselves own content.
type PreparedParts = Vec<(ContentPart, Vec<ContentPart>)>;

fn image(assets: &mut BinaryAssets, value: message::Image) -> Result<ImagePart, ContentError> {
    Ok(ImagePart {
        source: assets.intern(value.data)?,
        media_type: value.media_type,
        detail: value.detail,
        additional_params: value.additional_params,
    })
}

fn user(
    assets: &mut BinaryAssets,
    value: UserContent,
) -> Result<(ContentPart, Vec<ContentPart>), ContentError> {
    let mut children = Vec::new();
    let part = match value {
        UserContent::Text(value) => ContentPart::Text(value),
        UserContent::Image(value) => ContentPart::Image(image(assets, value)?),
        UserContent::Audio(value) => ContentPart::Audio(AudioPart {
            source: assets.intern(value.data)?,
            media_type: value.media_type,
            additional_params: value.additional_params,
        }),
        UserContent::Video(value) => ContentPart::Video(VideoPart {
            source: assets.intern(value.data)?,
            media_type: value.media_type,
            additional_params: value.additional_params,
        }),
        UserContent::Document(value) => ContentPart::Document(DocumentPart {
            source: assets.intern(value.data)?,
            media_type: value.media_type,
            additional_params: value.additional_params,
        }),
        UserContent::ToolResult(value) => {
            children = value
                .content
                .into_iter()
                .map(|item| {
                    Ok(match item {
                        ToolResultContent::Text(text) => ContentPart::Text(text),
                        ToolResultContent::Image(value) => {
                            ContentPart::Image(image(assets, value)?)
                        }
                        ToolResultContent::Json { value } => ContentPart::Json(value),
                    })
                })
                .collect::<Result<_, ContentError>>()?;
            ContentPart::ToolResult {
                call: value.call,
                provider: value.provider,
                name: value.name,
            }
        }
    };
    Ok((part, children))
}

fn prepare(
    assets: &mut BinaryAssets,
    parts: MessageParts,
) -> Result<(Role, Option<MessageId>, PreparedParts), ContentError> {
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
                    let part = match part {
                        AssistantContent::Text(value) => ContentPart::Text(value),
                        AssistantContent::Image(value) => ContentPart::Image(image(assets, value)?),
                        AssistantContent::ToolCall(value) => ContentPart::ToolCall(value),
                        AssistantContent::Reasoning(value) => ContentPart::Reasoning(value),
                    };
                    Ok((part, Vec::new()))
                })
                .collect::<Result<_, ContentError>>()?,
        ),
    })
}

fn spawn_parts(world: &mut World, parent: Entity, parts: PreparedParts) {
    for (part, children) in parts {
        // Relationship hooks must finish before payload observers inspect siblings.
        let entity = world.spawn(ChildOf(parent)).insert(part).id();
        for child in children {
            world.spawn(ChildOf(entity)).insert(child);
        }
    }
}

/// Replace an utterance's content graph. This is a persistent edit; request-only
/// steering must use request part edits instead. Missing utterances or rejected
/// binary sources return errors without changing the graph; unreferenced interned
/// assets can be collected.
/// Sibling order is the utterance's `Children` order, the parts as given.
/// Repeated identical parts remain distinct entities.
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
    replace_parts(world, utterance, role, id, parts);
    Ok(())
}

fn replace_parts(
    world: &mut World,
    utterance: Entity,
    role: Role,
    id: Option<MessageId>,
    parts: PreparedParts,
) {
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
        ordered.push(child);
    }
    Ok(ordered)
}

fn read_image(assets: &BinaryAssets, value: &ImagePart) -> Result<message::Image, ContentError> {
    Ok(message::Image {
        data: assets.resolve(&value.source)?,
        media_type: value.media_type.clone(),
        detail: value.detail.clone(),
        additional_params: value.additional_params.clone(),
    })
}

fn read_edited_part<'a>(
    get: &impl Fn(Entity) -> Option<EntityRef<'a>>,
    entity: Entity,
    nested: bool,
    edits: &BTreeMap<Entity, RequestPartEdit>,
) -> Result<Option<ContentPart>, ContentError> {
    let entity_ref = get(entity).ok_or(ContentError::Missing)?;
    let mut value = entity_ref
        .get::<ContentPart>()
        .ok_or(ContentError::Shape)?
        .clone();
    if matches!(value, ContentPart::ToolResult { .. }) {
        if nested {
            return Err(ContentError::Shape);
        }
    } else if entity_ref
        .get::<Children>()
        .is_some_and(|children| !children.is_empty())
    {
        return Err(ContentError::Shape);
    }
    match edits.get(&entity) {
        Some(RequestPartEdit::Remove) => return Ok(None),
        Some(RequestPartEdit::Text(text)) => match &mut value {
            ContentPart::Text(part) => part.text.clone_from(text),
            _ => return Err(ContentError::Shape),
        },
        None => {}
    }
    Ok(Some(value))
}

fn to_user<'a>(
    get: &impl Fn(Entity) -> Option<EntityRef<'a>>,
    assets: &BinaryAssets,
    entity: Entity,
    value: ContentPart,
    edits: &BTreeMap<Entity, RequestPartEdit>,
) -> Result<UserContent, ContentError> {
    Ok(match value {
        ContentPart::Text(value) => UserContent::Text(value),
        ContentPart::Image(value) => UserContent::Image(read_image(assets, &value)?),
        ContentPart::Audio(value) => UserContent::Audio(message::Audio {
            data: assets.resolve(&value.source)?,
            media_type: value.media_type,
            additional_params: value.additional_params,
        }),
        ContentPart::Video(value) => UserContent::Video(message::Video {
            data: assets.resolve(&value.source)?,
            media_type: value.media_type,
            additional_params: value.additional_params,
        }),
        ContentPart::Document(value) => UserContent::Document(message::Document {
            data: assets.resolve(&value.source)?,
            media_type: value.media_type,
            additional_params: value.additional_params,
        }),
        ContentPart::ToolResult {
            call,
            provider,
            name,
        } => UserContent::ToolResult(message::ToolResult {
            call,
            provider,
            name,
            content: ordered(get, entity)?
                .into_iter()
                .map(|child| read_edited_part(get, child, true, edits))
                .collect::<Result<Vec<_>, _>>()?
                .into_iter()
                .flatten()
                .map(|part| {
                    Ok(match part {
                        ContentPart::Text(value) => ToolResultContent::Text(value),
                        ContentPart::Image(value) => {
                            ToolResultContent::Image(read_image(assets, &value)?)
                        }
                        ContentPart::Json(value) => ToolResultContent::Json { value },
                        _ => return Err(ContentError::Shape),
                    })
                })
                .collect::<Result<_, ContentError>>()?,
        }),
        _ => return Err(ContentError::Shape),
    })
}

/// Reconstruct a message from its role, ID and typed children in sibling
/// order. Invalid types, missing components and unresolved assets are errors.
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
    /// Read one message with transient part edits. The caller must validate edit
    /// ownership. Returns an error for invalid graph data, binary sources, or edits.
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
    /// Returns an error for missing entities or invalid parent and part types.
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
        if !matches!(
            owner.get::<ContentPart>(),
            Some(ContentPart::ToolResult { .. })
        ) {
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

    /// Reconstruct one utterance in sibling order. Returns an error for missing
    /// entities or components, invalid roles or part types, or unresolved binaries.
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
        .map(|entity| {
            read_edited_part(get, entity, false, edits).map(|part| part.map(|part| (entity, part)))
        })
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
                    .map(|(entity, value)| to_user(get, assets, entity, value, edits))
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
                .map(|(_, value)| {
                    Ok(match value {
                        ContentPart::Text(value) => AssistantContent::Text(value),
                        ContentPart::Image(value) => {
                            AssistantContent::Image(read_image(assets, &value)?)
                        }
                        ContentPart::ToolCall(value) => AssistantContent::ToolCall(value),
                        ContentPart::Reasoning(value) => AssistantContent::Reasoning(value),
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
    for part in world.query::<&ContentPart>().iter(world) {
        let source = match part {
            ContentPart::Image(part) => &part.source,
            ContentPart::Audio(part) => &part.source,
            ContentPart::Video(part) => &part.source,
            ContentPart::Document(part) => &part.source,
            _ => continue,
        };
        if let PartSource::Binary { id, .. } = source {
            roots.push(*id);
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
) -> Result<Entity, ContentError> {
    spawn_deferred_with(commands, assets, parent, message, Vec::new())
}

/// `spawn_deferred`, stamping `statuses` on the utterance's tool-result
/// parts in sibling order (the batch's results, CONTRACT §8.1).
pub(crate) fn spawn_deferred_with(
    commands: &mut Commands,
    assets: &mut BinaryAssets,
    parent: Entity,
    message: MessageParts,
    statuses: Vec<ToolResultStatus>,
) -> Result<Entity, ContentError> {
    let (role, id, parts) = prepare(assets, message)?;
    let mut utterance = commands.spawn((Utterance, role, ChildOf(parent)));
    if let Some(id) = id {
        utterance.insert(id);
    }
    let entity = utterance.id();
    commands.queue(move |world: &mut World| {
        spawn_parts(world, entity, parts);
        stamp_statuses(world, entity, &statuses);
    });
    Ok(entity)
}

fn stamp_statuses(world: &mut World, utterance: Entity, statuses: &[ToolResultStatus]) {
    if statuses.is_empty() {
        return;
    }
    let results: Vec<Entity> = world
        .get::<Children>(utterance)
        .into_iter()
        .flat_map(|children| children.iter())
        .filter(|child| {
            matches!(
                world.get::<ContentPart>(*child),
                Some(ContentPart::ToolResult { .. })
            )
        })
        .collect();
    for (entity, status) in results.into_iter().zip(statuses) {
        world.entity_mut(entity).insert(*status);
    }
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
        replace_parts(world, utterance, role, id, parts);
    });
    Ok(())
}
