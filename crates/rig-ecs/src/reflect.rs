//! Registration of checkpointed components and opaque rig-core reflection wrappers.
//!
//! ```
//! let mut world = bevy_ecs::world::World::new();
//! rig_ecs::reflect::install_reflect(&mut world);
//! ```

use bevy_ecs::{prelude::*, reflect::AppTypeRegistry};

pub use crate::{agent::reflect::*, bus::reflect::*};

/// Declare opaque [`reflect_remote`](bevy_reflect::reflect_remote) wrappers
/// for the rig-core types this module's components hold: each reflects as a
/// whole through its serde form, which is the wire form the log already has.
/// `Debug`, `Clone`, `Serialize` and `Deserialize` come with every wrapper;
/// anything listed after the colon is both derived and reflected.
macro_rules! opaque_reflect {
    ($(
        $(#[doc = $doc:literal])*
        $kind:ident $name:ident($($remote:tt)*) $(: $($extra:ident),*)? ;
    )*) => {$(
        $(#[doc = $doc])*
        #[bevy_reflect::reflect_remote($($remote)*)]
        #[derive(Debug, Clone, $($($extra,)*)? serde::Serialize, serde::Deserialize)]
        #[reflect(opaque, Debug, $($($extra,)*)? Serialize, Deserialize)]
        pub $kind $name {}
    )*};
}

pub(crate) use opaque_reflect;

macro_rules! register_all {
    ($registry:expr, [$($ty:ty),* $(,)?]) => {
        $( $registry.register::<$ty>(); )*
    };
}

/// Register reflected bus and agent components and their remote wrappers in the
/// world's [`AppTypeRegistry`], creating it if absent.
pub fn install_reflect(world: &mut World) {
    use crate::{agent, bus, systems};
    world.init_resource::<AppTypeRegistry>();
    let registry = world.resource::<AppTypeRegistry>().clone();
    let mut registry = registry.write();
    register_all!(
        registry,
        [
            bevy_ecs::hierarchy::ChildOf,
            bevy_ecs::hierarchy::Children,
            bus::PendingEffect,
            bus::Seq,
            bus::SeqCounter,
            bus::IdCounter,
            bus::Reserved,
            bus::Issued,
            bus::Held,
            bus::InFlight,
            bus::Streamed,
            bus::EffectOutcome,
            bus::ToolInputs,
            bus::ToolOutputs,
            bus::Bound,
            HandlerKeyReflect,
            EffectKindReflect,
            EffectIdReflect,
            HandlerDescriptorReflect,
            ToolContextReflect,
            OutcomeReflect,
            StreamedOutcomeReflect,
            StreamEventsReflect,
            StreamErrorsReflect,
            agent::checkpoint::TurnAssistant,
            agent::checkpoint::AssistantForTurns,
            agent::checkpoint::TurnResults,
            agent::checkpoint::ResultsForTurns,
            agent::OutputKind,
            agent::Unhandled,
            agent::ModelOf,
            agent::RememberedBy,
            agent::RetrievedBy,
            agent::RetrievalKind,
            agent::RoutedTo,
            agent::Grants,
            agent::ContextOf,
            agent::AttachedTo,
            agent::content::parts::EditTarget,
            agent::content::parts::EditedBy,
            agent::content::reflect::TextPartReflect,
            agent::content::reflect::ToolCallPartReflect,
            agent::content::reflect::ReasoningPartReflect,
            agent::content::reflect::JsonPartReflect,
            agent::content::reflect::ImageMediaReflect,
            agent::content::reflect::AudioMediaReflect,
            agent::content::reflect::VideoMediaReflect,
            agent::content::reflect::DocumentMediaReflect,
            agent::content::reflect::PartParamsReflect,
            agent::content::reflect::ImageDetailReflect,
            agent::MessageParts,
            agent::Runs,
            agent::RunCounter,
            agent::ToolCallSlot,
            agent::Failure,
            agent::AdvertisedOn,
            systems::Fresh,
            systems::Folded,
            systems::Materialised,
            JsonReflect,
            OptionalJsonReflect,
            ToolChoiceReflect,
            UsageReflect,
            ToolCallIdReflect,
            ProviderCallIdReflect,
            AssistantContentsReflect,
            MessageReflect,
        ]
    );
    register_all!(
        registry,
        [
            agent::Owner,
            agent::Preamble,
            agent::Temperature,
            agent::MaxTokens,
            agent::AdditionalParams,
            agent::ToolChoiceSpec,
            agent::Output,
            agent::OutputToolConfig,
            agent::MaxTurns,
            agent::InvalidCalls,
            agent::DefaultMaxTurns,
            agent::ToolAccess,
            agent::DocumentId,
            agent::DocumentText,
            agent::DocumentProps,
            agent::Utterance,
            agent::Role,
            agent::content::parts::MessageId,
            agent::content::parts::RequestPartEdit,
            agent::content::parts::ContentPart,
            agent::content::parts::ToolResultStatus,
            agent::content::parts::ToolResultLimit,
            agent::Run,
            agent::RunSeq,
            agent::StreamRequested,
            agent::Cursor,
            agent::Ready,
            agent::Prompt,
            agent::RunPhase,
            agent::Settled,
            agent::Failed,
            agent::RunResult,
            agent::Usage,
            agent::OutputRetries,
            agent::InvalidRetries,
            agent::OutputToolName,
            agent::ProviderRetries,
            agent::ProviderRetried,
            agent::ProviderRetrying,
            bus::Scope,
            agent::Turn,
            agent::Outputs,
            agent::Reprompt,
            agent::checkpoint::ToolTurnCommit,
            agent::checkpoint::ToolTurnHolds,
            agent::InvalidCall,
            agent::Resolution,
            agent::ToolPolicy,
            agent::ToolContextSpec,
            agent::Batch,
            agent::Cancelled,
            agent::Retry,
            agent::RequestPatch,
            agent::Conversation,
            agent::Remembered,
            agent::Remembering,
            agent::MemoryAppendScheduled,
            agent::PolicyVersion,
            agent::Retrieval,
            agent::Retrievable,
            agent::Retrieving,
            agent::UsesModel,
            agent::RunOf,
            agent::Grant,
            agent::Route,
            agent::Remembers,
            agent::Retrieves,
            agent::Context,
            agent::Attachment,
            agent::Advert,
            bus::ServedBy,
            bus::Serves,
            bus::HoldOwners,
            systems::BatchHeld,
        ]
    );
}
