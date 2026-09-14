//! Cached utterance views (CONTRACT §1): the DTO an utterance renders to,
//! kept on the utterance between assemblies and dropped by change
//! detection. The request is unchanged by the cache: a view is exactly
//! what `ContentGraph::message` renders, and everything per-request — the
//! turn's `RequestPartEdit`s, the `ToolResultLimit` in force — is applied
//! over it by `assemble`, as it is over a fresh render.
//!
//! Why the verbatim DTO and not a keyed encoding: the adapters encode each
//! message in the context of the whole request (system hoisting, a
//! whole-history tool-call id table, positional cache breakpoints, a merged
//! tool array), so the largest fragment that is reusable across turns is
//! the canonical message. And why verbatim rather than edit-and-limit
//! applied: edits are the turn's, consumed at the fold; the limit is the
//! run's, changeable between turns; a view keyed on either would be
//! invalidated every turn they are present, for no gain — the limit cuts
//! bytes, not entity walks.

use std::collections::HashSet;

use bevy_ecs::{prelude::*, system::SystemParam};

use super::binary::BinaryAssets;
use super::parts::{
    AudioPart, ContentPart, DocumentPart, ImagePart, JsonPart, MessageId, ReasoningPart, TextPart,
    ToolCallPart, ToolResultPart, VideoPart,
};
use crate::agent::{MessageParts, Order, Role, Utterance};

/// The version of the graph-to-DTO rendering a view was made by. A view
/// made by another version is a miss. Bump it when `read_message` changes
/// what it renders for the same graph.
pub const ENCODER_VERSION: u32 = 1;

/// The rendered DTO of an utterance, verbatim — no request edit, no
/// limit — as `assemble` last read it. Runtime state, never scene data
/// (§13): a loaded utterance has none until its first assembly. Dropped by
/// [`MessageCache::stale`] when the utterance or its part subtree changed.
#[derive(Component, Debug, Clone)]
pub struct CachedMessage {
    version: u32,
    assets: u64,
    parts: MessageParts,
}

impl CachedMessage {
    /// A view rendered by this encoder over the asset store at `assets`.
    #[must_use]
    pub fn new(parts: MessageParts, assets: u64) -> Self {
        Self {
            version: ENCODER_VERSION,
            assets,
            parts,
        }
    }

    /// The view, when this encoder rendered it and no asset has been
    /// collected since: a view holds resolved payloads, and a collection
    /// is the one way a resolvable source stops resolving.
    #[must_use]
    pub fn view(&self, assets: u64) -> Option<&MessageParts> {
        (self.version == ENCODER_VERSION && self.assets == assets).then_some(&self.parts)
    }
}

/// What `assemble` did, as counters: a diagnostic resource, zero cost
/// beyond the increments, never reset by the crate. `renders` is the
/// number of full part-subtree renders (`ContentGraph::message` or
/// `message_with`); with the cache warm it is the number of utterances
/// that changed since the last assembly, not the length of the history.
#[derive(Resource, Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct AssemblyStats {
    /// Requests folded.
    pub assemblies: u64,
    /// Full part-subtree renders.
    pub renders: u64,
    /// Utterances read from their cached view.
    pub hits: u64,
    /// Cached views dropped by change detection.
    pub evictions: u64,
}

/// A part entity whose own components changed since the reading system
/// last ran — including its `Order`, its parent, and (a tool result's)
/// children.
type PartChanged = Or<(
    Changed<ContentPart>,
    Changed<Order>,
    Changed<ChildOf>,
    Changed<Children>,
    Changed<TextPart>,
    Changed<ImagePart>,
    Changed<AudioPart>,
    Changed<VideoPart>,
    Changed<DocumentPart>,
    Changed<ToolCallPart>,
    Changed<ReasoningPart>,
    Changed<JsonPart>,
    Changed<ToolResultPart>,
)>;

/// An utterance whose own rendered components changed, or whose children
/// were added, removed or reparented.
type UtteranceChanged = Or<(
    Changed<Utterance>,
    Changed<Role>,
    Changed<MessageId>,
    Changed<Children>,
)>;

/// The removals change detection does not see: a component taken off a
/// live entity. A despawned part is seen through its parent's `Children`.
#[derive(SystemParam)]
struct Removed<'w, 's> {
    content: RemovedComponents<'w, 's, ContentPart>,
    order: RemovedComponents<'w, 's, Order>,
    child_of: RemovedComponents<'w, 's, ChildOf>,
    children: RemovedComponents<'w, 's, Children>,
    text: RemovedComponents<'w, 's, TextPart>,
    image: RemovedComponents<'w, 's, ImagePart>,
    audio: RemovedComponents<'w, 's, AudioPart>,
    video: RemovedComponents<'w, 's, VideoPart>,
    document: RemovedComponents<'w, 's, DocumentPart>,
    call: RemovedComponents<'w, 's, ToolCallPart>,
    reasoning: RemovedComponents<'w, 's, ReasoningPart>,
    json: RemovedComponents<'w, 's, JsonPart>,
    result: RemovedComponents<'w, 's, ToolResultPart>,
    role: RemovedComponents<'w, 's, Role>,
    message_id: RemovedComponents<'w, 's, MessageId>,
}

impl Removed<'_, '_> {
    fn drain(&mut self) -> Vec<Entity> {
        let mut removed = Vec::new();
        removed.extend(self.content.read());
        removed.extend(self.order.read());
        removed.extend(self.child_of.read());
        removed.extend(self.children.read());
        removed.extend(self.text.read());
        removed.extend(self.image.read());
        removed.extend(self.audio.read());
        removed.extend(self.video.read());
        removed.extend(self.document.read());
        removed.extend(self.call.read());
        removed.extend(self.reasoning.read());
        removed.extend(self.json.read());
        removed.extend(self.result.read());
        removed.extend(self.role.read());
        removed.extend(self.message_id.read());
        removed
    }
}

/// The cache beside its counters: what `assemble` takes, as one parameter.
#[derive(SystemParam)]
pub struct Cached<'w, 's> {
    /// The views and their invalidation.
    pub cache: MessageCache<'w, 's>,
    /// The counters.
    pub stats: ResMut<'w, AssemblyStats>,
}

/// Read access to the cached views, with the change detection that
/// invalidates them. The detection is the reading system's: `stale` sees
/// every change since that system last ran, so nothing scheduled between
/// a refresh and the read can leave a view behind — which is why the
/// refresh is not a system of its own before `RigSet::Assemble`.
#[derive(SystemParam)]
pub struct MessageCache<'w, 's> {
    views: Query<'w, 's, &'static CachedMessage>,
    changed_parts: Query<'w, 's, Entity, (With<ContentPart>, PartChanged)>,
    changed_utterances: Query<'w, 's, Entity, (With<Utterance>, UtteranceChanged)>,
    parents: Query<'w, 's, &'static ChildOf>,
    utterances: Query<'w, 's, (), With<Utterance>>,
    removed: Removed<'w, 's>,
    assets: Option<Res<'w, BinaryAssets>>,
}

impl MessageCache<'_, '_> {
    /// The utterances whose views are stale: their own components, a part
    /// of their subtree (at any depth), a sibling order, a parent link or a
    /// child set changed since the reading system last ran; or a component
    /// was removed from one of those. A despawned part reaches here through
    /// its parent's changed `Children`. Read once per run of the system.
    pub fn stale(&mut self) -> HashSet<Entity> {
        let mut stale: HashSet<Entity> = self.changed_utterances.iter().collect();
        let changed: Vec<Entity> = self.changed_parts.iter().collect();
        let removed = self.removed.drain();
        for entity in changed.into_iter().chain(removed) {
            if let Some(owner) = self.owner(entity) {
                stale.insert(owner);
            }
        }
        stale
    }

    /// The utterance owning `entity`: itself, its parent, or — for an item
    /// of a tool result — its grandparent.
    fn owner(&self, mut entity: Entity) -> Option<Entity> {
        for _ in 0..3 {
            if self.utterances.contains(entity) {
                return Some(entity);
            }
            entity = self.parents.get(entity).ok()?.parent();
        }
        None
    }

    /// The asset store's collection generation a view rendered now is
    /// keyed by.
    #[must_use]
    pub fn assets_generation(&self) -> u64 {
        self.assets.as_deref().map_or(0, BinaryAssets::generation)
    }

    /// Whether `utterance` holds a view, stale or not.
    #[must_use]
    pub fn holds(&self, utterance: Entity) -> bool {
        self.views.contains(utterance)
    }

    /// The valid cached view of `utterance`: present, not in `stale`,
    /// rendered by this encoder over the current asset store.
    #[must_use]
    pub fn view(&self, utterance: Entity, stale: &HashSet<Entity>) -> Option<&MessageParts> {
        if stale.contains(&utterance) {
            return None;
        }
        self.views
            .get(utterance)
            .ok()
            .and_then(|view| view.view(self.assets_generation()))
    }
}
