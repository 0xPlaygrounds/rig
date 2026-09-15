//! The run as a graph: the entity vocabulary of the agent runtime.
//!
//! Every setting is one component; every link is a Bevy relationship (a
//! many-to-many link is a link entity, `ChildOf` its owner, with a
//! relationship to what it links); every payload is serde and holds no
//! `Entity`. The request the model sees is derived from this graph by one
//! fold ([`crate::policy::fold_request`]) at [`crate::systems::RigSet::Assemble`];
//! nothing here holds a `CompletionRequest` or a `Vec<Message>`.
//!
//! | design (§3.1) | here |
//! |---|---|
//! | Agent | an entity with [`Owner`], [`Preamble`], [`Temperature`], [`MaxTokens`], [`AdditionalParams`], [`ToolChoiceSpec`], [`Output`], [`MaxTurns`], [`InvalidCalls`]; [`UsesModel`] → the model's handler entity; [`Grant`] link entities → tool handler entities; [`Context`] link entities → document entities; [`Route`] link entities → other models a run may be steered to |
//! | Steering (§9) | [`Cancelled`] on a run; [`Retry`] and [`RequestPatch`] on a turn; [`Resolution`] on an invalid call; `UsesModel` on a run |
//! | Model, Tool | the bus module's handler entities (`Bound`) |
//! | Document | [`DocumentId`], [`DocumentText`], [`DocumentProps`]; attached to a turn by an [`Attachment`] link entity |
//! | Utterance | [`Utterance`] + [`Role`] + ordered [`content::parts::ContentPart`] child entities; `ChildOf` the run, in sibling (`Children`) order |
//! | Run | [`Run`] + [`RunOf`] → agent; [`RunSeq`]; a [`RunPhase`] or an ending ([`Settled`], [`Failed`]); [`Cursor`]; [`RunResult`]; [`Usage`]; retries; [`OutputToolName`]; the run's own overrides of the agent's settings ([`ToolPolicy`], [`ToolContextSpec`] among them) |
//! | Turn | [`Turn`], `ChildOf` the run; [`Advert`] link entities → the tools it advertised; [`Attachment`] link entities → the documents it carried; [`Outputs`]; [`Reprompt`]; [`Batch`] while its tool calls are out |
//! | Effect | the bus module's, `ChildOf` the turn: the completion, then one per tool call ([`ToolCallSlot`] names which) |
//! | Invalid call | [`InvalidCall`] + [`Resolution`], `ChildOf` the turn |

pub mod checkpoint;
pub mod content;
pub mod reflect;

use bevy_ecs::prelude::*;
use bevy_reflect::{Reflect, ReflectDeserialize, ReflectSerialize};
use rig_core::{
    completion::{
        Usage as WireUsage,
        message::{AssistantContent, Message, ProviderCallId, ToolCallId, ToolChoice, UserContent},
    },
    error::ErrorReport,
    tool::ToolContext,
};
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Agent: one component per setting.

/// The agent's name: the owner of every key it mints (`<owner>/model:..`),
/// the scope of every run's records.
#[derive(Component, Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Reflect)]
#[reflect(Component)]
pub struct Owner(pub String);

/// The system prompt: `None` is "no system message" (a run without a
/// preamble), `Some("")` an empty one that is still sent.
#[derive(Component, Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, Reflect)]
#[reflect(Component)]
pub struct Preamble(pub Option<String>);

/// Sampling temperature, if the request names one.
#[derive(Component, Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize, Reflect)]
#[reflect(Component)]
pub struct Temperature(pub Option<f64>);

/// The answer's token budget, if the request names one.
#[derive(
    Component, Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, Reflect,
)]
#[reflect(Component)]
pub struct MaxTokens(pub Option<u64>);

/// Provider-specific parameters the request carries verbatim.
#[derive(Component, Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, Reflect)]
#[reflect(Component)]
pub struct AdditionalParams(
    #[reflect(remote = crate::agent::reflect::OptionalJsonReflect)] pub Option<serde_json::Value>,
);

/// The program's tool choice: what the request's `tool_choice` starts
/// from before the output mode has its say.
#[derive(Component, Debug, Clone, Default, PartialEq, Serialize, Deserialize, Reflect)]
#[reflect(Component)]
pub struct ToolChoiceSpec(
    #[reflect(remote = crate::agent::reflect::ToolChoiceReflect)] pub Option<ToolChoice>,
);

/// How the run's answer is asked for and read.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, Reflect)]
#[serde(rename_all = "snake_case")]
pub enum OutputKind {
    /// Resolve at request time: `Tool` when there is a schema and at least
    /// one tool and the tool choice permits it, else `Native`.
    #[default]
    Auto,
    /// The provider's native structured output (`output_schema`), or plain
    /// text when there is no schema.
    Native,
    /// A synthetic output tool the model must call with the answer.
    Tool,
    /// The schema in the preamble; the answer is the text, unvalidated.
    Prompted,
}

/// The output mode and its schema, if any.
#[derive(Component, Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, Reflect)]
#[reflect(Component)]
pub struct Output {
    /// The mode.
    pub mode: OutputKind,
    /// The JSON schema of the answer, raw.
    #[reflect(remote = crate::agent::reflect::OptionalJsonReflect)]
    pub schema: Option<serde_json::Value>,
}

/// Output-tool configuration, resolved from the run before the agent.
///
/// An explicit name reserves Tool mode when a schema is present, just like
/// a name already committed on the run. A conflicting real tool fails before
/// provider dispatch. Without a name, the normal collision-safe name is used.
#[derive(Component, Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Reflect)]
#[reflect(Component)]
pub struct OutputToolConfig {
    /// Reserved name, or the automatically chosen name when absent.
    pub name: Option<String>,
    /// Tool description, or the standard final-answer description when absent.
    pub description: Option<String>,
    /// Append the standard output-tool instructions to the preamble.
    pub augment_preamble: bool,
}

impl Default for OutputToolConfig {
    fn default() -> Self {
        Self {
            name: None,
            description: None,
            augment_preamble: true,
        }
    }
}

/// The model-call budget of a run.
#[derive(Component, Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Reflect)]
#[reflect(Component)]
pub struct MaxTurns(pub usize);

/// Retries a run allows a completion whose provider failure is retryable
/// (a 5xx, a rate limit, a transient block; the report's `retryable`
/// decides, CONTRACT §5): the same request is issued again over the same
/// history. No tool is re-run, no history is rewritten, the lost turn
/// leaves no assistant utterance and is not counted against `MaxTurns`.
/// On the agent or the run; without one, [`DEFAULT_PROVIDER_RETRIES`].
/// Time is the host's: a backoff is a hold on the re-issued effect from a
/// `Gate` system, released when due.
#[derive(Component, Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Reflect)]
#[reflect(Component)]
pub struct ProviderRetries(pub usize);

/// The provider-retry budget of a run that declares none.
pub const DEFAULT_PROVIDER_RETRIES: usize = 3;

/// How long a provider retry waits before its completion is re-issued:
/// `base × 2^(attempt-1)`, at most `max`, on the world's clock (bevy_time's
/// `Time`: a host that pauses `Time<Virtual>` holds every backoff). On the
/// agent or the run; without one, a retry is re-issued at once.
#[derive(Component, Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Reflect)]
#[reflect(Component)]
pub struct Backoff {
    /// The first retry's delay.
    pub base: std::time::Duration,
    /// The longest delay any retry waits.
    pub max: std::time::Duration,
}

/// What to do with a tool call the program does not advertise when no
/// system resolved it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, Reflect)]
#[serde(rename_all = "snake_case")]
pub enum Unhandled {
    /// The run fails at the record.
    #[default]
    Fail,
    /// The call is dropped; the run goes on.
    Ignore,
}

/// The invalid-call policy: how many retries before `unhandled` applies.
#[derive(
    Component, Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, Reflect,
)]
#[reflect(Component)]
pub struct InvalidCalls {
    /// Retries the program allows an invalid call.
    pub retries: usize,
    /// What happens when they are spent.
    pub unhandled: Unhandled,
}

/// The application's versioned declaration of its policy systems, ordering,
/// and configuration not represented by the library's policy components.
/// Set on the agent or override on the run before stamping its identity.
/// Change this value when that policy changes. It is a declaration, not an
/// automatic fingerprint of executable code or ambient credentials.
#[derive(Component, Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Reflect)]
#[reflect(Component)]
pub struct PolicyVersion(pub String);

/// How many of a turn's tool calls may be in flight at once: 1 (the
/// default) runs them one after another in call order; N keeps N going.
/// The run's, else the agent's.
#[derive(Component, Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Reflect)]
#[reflect(Component)]
pub struct ToolPolicy {
    /// Calls in flight at once; 0 reads as 1.
    pub concurrency: usize,
}

impl Default for ToolPolicy {
    fn default() -> Self {
        Self { concurrency: 1 }
    }
}

/// The context every tool call of the run runs with (format 5: beside the
/// effect, never in it): its `for_dispatch` snapshot becomes the call's
/// `bus::ToolInputs`. The run's, else the agent's, else empty.
#[derive(Component, Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, Reflect)]
#[reflect(Component)]
pub struct ToolContextSpec(
    #[reflect(remote = crate::agent::reflect::ToolContextReflect)] pub ToolContext,
);

/// The default `max_turns` the agent was built with, part of its identity
/// (a run-level override is not).
#[derive(
    Component, Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, Reflect,
)]
#[reflect(Component)]
pub struct DefaultMaxTurns(pub Option<usize>);

/// The agent uses this model: a relationship to the model's handler
/// entity (the bus module's `Bound`). A run may carry its own to override.
#[derive(Component, Debug, Clone, Copy, PartialEq, Eq, Reflect)]
#[relationship(relationship_target = ModelOf)]
#[reflect(Component)]
pub struct UsesModel(pub Entity);

/// The agents and runs using this model.
#[derive(Component, Debug, Default, Reflect)]
#[relationship_target(relationship = UsesModel)]
#[reflect(Component)]
pub struct ModelOf(Vec<Entity>);

/// The agent remembers: a relationship to the memory handler entity. A run
/// spawned with no history loads the conversation before its first turn
/// and appends what it said when it settles (CONTRACT §11).
#[derive(Component, Debug, Clone, Copy, PartialEq, Eq, Reflect)]
#[relationship(relationship_target = RememberedBy)]
#[reflect(Component)]
pub struct Remembers(pub Entity);

/// The agents remembering through this handler.
#[derive(Component, Debug, Default, Reflect)]
#[relationship_target(relationship = Remembers)]
#[reflect(Component)]
pub struct RememberedBy(Vec<Entity>);

/// The conversation a run loads and appends under (the run's, else the
/// agent's).
#[derive(Component, Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Reflect)]
#[reflect(Component)]
pub struct Conversation(pub String);

/// An utterance that came from memory: not appended again.
#[derive(
    Component, Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, Reflect,
)]
#[reflect(Component)]
pub struct Remembered;

/// The run loaded its conversation and will append to it when it settles.
#[derive(
    Component, Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, Reflect,
)]
#[reflect(Component)]
pub struct Remembering;

/// Finalization has created this run's memory append intent. Persisted so
/// rehydrating `Settled` cannot schedule a second operation. The child
/// effect owns the request, saved dispatch id, and eventual outcome; this
/// marker does not claim that the external write succeeded exactly once.
#[derive(
    Component, Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, Reflect,
)]
#[reflect(Component)]
pub struct MemoryAppendScheduled;

/// A retrieval the agent makes before every turn: a link entity, `ChildOf`
/// the agent, naming the index handler entity, with [`Retrieval`] saying
/// what for (CONTRACT §12).
#[derive(Component, Debug, Clone, Copy, PartialEq, Eq, Reflect)]
#[relationship(relationship_target = RetrievedBy)]
#[reflect(Component)]
pub struct Retrieves(pub Entity);

/// The retrieval links naming this index.
#[derive(Component, Debug, Default, Reflect)]
#[relationship_target(relationship = Retrieves)]
#[reflect(Component)]
pub struct RetrievedBy(Vec<Entity>);

/// What a [`Retrieves`] link retrieves.
#[derive(Component, Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Reflect)]
#[reflect(Component)]
pub struct Retrieval {
    /// How many results are asked for.
    pub samples: u64,
    /// Documents to attach, or tools to advertise.
    pub what: RetrievalKind,
}

/// The two retrievals.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Reflect)]
#[serde(rename_all = "snake_case")]
pub enum RetrievalKind {
    /// Scored documents, attached to the turn after its static ones.
    Documents,
    /// Tool ids, advertised first among the turn's tools.
    Tools,
}

/// A grant advertised only when a tool retrieval names it.
#[derive(
    Component, Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, Reflect,
)]
#[reflect(Component)]
pub struct Retrievable;

/// A fresh turn whose retrievals are out: folded once they land.
#[derive(
    Component, Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, Reflect,
)]
#[reflect(Component)]
pub struct Retrieving;

/// A route: a link entity, `ChildOf` the agent, naming another model the
/// agent may be steered to (a system inserting [`UsesModel`] on the run);
/// the required row names it.
#[derive(Component, Debug, Clone, Copy, PartialEq, Eq, Reflect)]
#[relationship(relationship_target = RoutedTo)]
#[reflect(Component)]
pub struct Route(pub Entity);

/// The routes naming this model.
#[derive(Component, Debug, Default, Reflect)]
#[relationship_target(relationship = Route)]
#[reflect(Component)]
pub struct RoutedTo(Vec<Entity>);

/// A grant: a link entity, `ChildOf` the agent, naming one tool the agent
/// advertises. Advertisement order is the agent's sibling (`Children`) order.
#[derive(Component, Debug, Clone, Copy, PartialEq, Eq, Reflect)]
#[relationship(relationship_target = Grants)]
#[reflect(Component)]
pub struct Grant(pub Entity);

/// Tool execution and permission policy, independent of request advertisements.
///
/// Resolved from the run, then its agent. Assembly stores
/// the effective snapshot on the turn before issuing its completion. Later
/// run changes therefore affect future turns only. On a turn this component
/// is the runtime's retained snapshot, not an input override. The synthetic output tool
/// remains governed by the output configuration rather than these ordinary
/// tool permissions.
#[derive(Component, Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, Reflect)]
#[reflect(opaque, Component, Debug, PartialEq, Serialize, Deserialize)]
pub struct ToolAccess {
    /// Executable names and their registered handler keys. `None` uses the
    /// advertised handlers; an explicit map may include unadvertised tools.
    pub executable: Option<std::collections::BTreeMap<String, rig_core::effect::HandlerKey>>,
    /// Names accepted by invalid-call policy. `None` uses executable names;
    /// `Some(empty)` denies all ordinary tools without changing the request.
    /// Kept independently for diagnostics even when a name is not executable.
    pub allowed: Option<std::collections::BTreeSet<String>>,
}

/// The grants naming this tool.
#[derive(Component, Debug, Default, Reflect)]
#[relationship_target(relationship = Grant)]
#[reflect(Component)]
pub struct Grants(Vec<Entity>);

/// A context link: a link entity, `ChildOf` the agent, naming one document
/// every turn carries as static context, in sibling (`Children`) order.
#[derive(Component, Debug, Clone, Copy, PartialEq, Eq, Reflect)]
#[relationship(relationship_target = ContextOf)]
#[reflect(Component)]
pub struct Context(pub Entity);

/// The context links naming this document.
#[derive(Component, Debug, Default, Reflect)]
#[relationship_target(relationship = Context)]
#[reflect(Component)]
pub struct ContextOf(Vec<Entity>);

// ---------------------------------------------------------------------------
// Documents: entities of their own, shared by attachment.

/// The document's stable id.
#[derive(Component, Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Reflect)]
#[reflect(Component)]
pub struct DocumentId(pub String);

/// The document's text.
#[derive(Component, Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Reflect)]
#[reflect(Component)]
pub struct DocumentText(pub String);

/// The document's string metadata, rendered before its text.
#[derive(Component, Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, Reflect)]
#[reflect(Component)]
pub struct DocumentProps(pub std::collections::HashMap<String, String>);

/// An attachment: a link entity, `ChildOf` a turn, naming one document the
/// turn's request carries, in sibling (`Children`) order.
#[derive(Component, Debug, Clone, Copy, PartialEq, Eq, Reflect)]
#[relationship(relationship_target = AttachedTo)]
#[reflect(Component)]
pub struct Attachment(pub Entity);

/// The attachments naming this document.
#[derive(Component, Debug, Default, Reflect)]
#[relationship_target(relationship = Attachment)]
#[reflect(Component)]
pub struct AttachedTo(Vec<Entity>);

// ---------------------------------------------------------------------------
// Utterances: the conversation, one entity per message, `ChildOf` the run.

/// An utterance: one message of the conversation, `ChildOf` its run, in
/// sibling (`Children`) order. Its parts are content components.
#[derive(
    Component, Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, Reflect,
)]
#[reflect(Component)]
pub struct Utterance;

/// Who spoke.
#[derive(Component, Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Reflect)]
#[serde(rename_all = "snake_case")]
#[reflect(Component)]
pub enum Role {
    /// The user, or a tool result the user side reports.
    User,
    /// The model.
    Assistant,
}

/// The content of one message, by role.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Reflect)]
#[serde(tag = "role", rename_all = "snake_case")]
#[reflect(opaque, Debug, PartialEq, Serialize, Deserialize)]
pub enum MessageParts {
    /// A user message's parts.
    User {
        /// The parts.
        content: Vec<UserContent>,
    },
    /// An assistant message's parts and provider id.
    Assistant {
        /// The provider-assigned message id, when the wire had one.
        id: Option<String>,
        /// The parts.
        content: Vec<AssistantContent>,
    },
}

impl MessageParts {
    /// The message, verbatim.
    pub fn to_message(&self) -> Message {
        match self {
            Self::User { content } => Message::User {
                content: content.clone(),
            },
            Self::Assistant { id, content } => Message::Assistant {
                id: id.clone(),
                content: content.clone(),
            },
        }
    }

    /// From a message; a system message is not an utterance (it is the
    /// preamble) and is refused.
    pub fn from_message(message: &Message) -> Option<Self> {
        match message {
            Message::System { .. } => None,
            Message::User { content } => Some(Self::User {
                content: content.clone(),
            }),
            Message::Assistant { id, content } => Some(Self::Assistant {
                id: id.clone(),
                content: content.clone(),
            }),
        }
    }

    /// The role.
    pub fn role(&self) -> Role {
        match self {
            Self::User { .. } => Role::User,
            Self::Assistant { .. } => Role::Assistant,
        }
    }
}

/// A run's prompt: the parts of the user message it opens with — text,
/// or text and images in the order given. A component on a fresh run
/// (`RunCommands::spawn_run` puts it there; a host assembling a run by
/// hand does the same), consumed when the run opens: the pass after
/// [`Ready`] is on the run, `systems::open_runs` spawns it as the run's
/// last utterance and takes the component off. A checkpoint saved before
/// then carries it (§13).
#[derive(Component, Debug, Clone, PartialEq, Serialize, Deserialize, Reflect)]
#[reflect(Component, opaque, Debug, PartialEq, Serialize, Deserialize)]
pub struct Prompt(pub Vec<UserContent>);

/// The host's word that a run's graph is complete — its history
/// utterances in place, its [`Prompt`] on it — and the run may start.
/// `RunCommands::spawn_run` inserts it once the utterances exist; a host
/// that populates a run by hand inserts it last. `systems::open_runs`
/// gives a `Ready` run its first phase, and `Advance` takes only `Ready`
/// runs: a run without it is never assembled, however complete. Kept for
/// the life of the run; a checkpoint saves and restores it (§13).
#[derive(
    Component, Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, Reflect,
)]
#[reflect(Component)]
pub struct Ready;

impl From<&str> for Prompt {
    fn from(text: &str) -> Self {
        Self(vec![UserContent::text(text)])
    }
}

impl From<String> for Prompt {
    fn from(text: String) -> Self {
        Self(vec![UserContent::text(text)])
    }
}

impl From<UserContent> for Prompt {
    fn from(part: UserContent) -> Self {
        Self(vec![part])
    }
}

impl From<Vec<UserContent>> for Prompt {
    fn from(parts: Vec<UserContent>) -> Self {
        Self(parts)
    }
}

// ---------------------------------------------------------------------------
// Runs and turns.

/// A run: one prompt through the agent to an answer or a failure. Requires
/// what every run carries — its cursor, tallies, output-tool name and usage
/// at their defaults — and stamps its [`RunSeq`] from the world's
/// [`RunCounter`], its [`crate::bus::Scope`] (`{owner}/run#{seq}`, from
/// its [`RunOf`] agent's [`Owner`]) and a `Name` of the same text, as it
/// is added.
#[derive(
    Component, Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, Reflect,
)]
#[require(
    Cursor,
    OutputRetries,
    InvalidRetries,
    ProviderRetried,
    OutputToolName,
    Usage,
    StreamRequested
)]
#[component(on_add = stamp_run)]
#[reflect(Component)]
pub struct Run;

fn stamp_run(
    mut world: bevy_ecs::world::DeferredWorld<'_>,
    context: bevy_ecs::lifecycle::HookContext,
) {
    let entity = context.entity;
    // A world without the counter (a checkpoint's scratch world) stamps
    // nothing: the loaded run carries its own `RunSeq`.
    if world.get::<RunSeq>(entity).is_none()
        && let Some(mut counter) = world.get_resource_mut::<RunCounter>()
    {
        let seq = counter.0;
        counter.0 += 1;
        let owner = world
            .get::<RunOf>(entity)
            .and_then(|run_of| world.get::<Owner>(run_of.0))
            .map(|owner| owner.0.clone())
            .unwrap_or_default();
        world
            .commands()
            .entity(entity)
            .insert((RunSeq(seq), crate::bus::Scope(format!("{owner}/run#{seq}"))));
    }
}

/// The run's agent.
#[derive(Component, Debug, Clone, Copy, PartialEq, Eq, Reflect)]
#[relationship(relationship_target = Runs)]
#[reflect(Component)]
pub struct RunOf(pub Entity);

/// The agent's runs.
#[derive(Component, Debug, Default, Reflect)]
#[relationship_target(relationship = RunOf)]
#[reflect(Component)]
pub struct Runs(Vec<Entity>);

impl Runs {
    /// The run entities.
    pub fn runs(&self) -> &[Entity] {
        &self.0
    }
}

/// The run's place among the world's runs: the order `Assemble` visits
/// them, so effects are minted in a stable order. Stamped at spawn from
/// [`RunCounter`].
#[derive(
    Component,
    Debug,
    Clone,
    Copy,
    Default,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Serialize,
    Deserialize,
    Reflect,
)]
#[reflect(Component)]
pub struct RunSeq(pub u64);

/// The world's one run counter.
#[derive(Resource, Debug, Default, Reflect)]
#[reflect(Resource)]
pub struct RunCounter(pub(crate) u64);

/// Whether the model is asked for a stream.
#[derive(
    Component, Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, Reflect,
)]
#[reflect(Component)]
pub struct StreamRequested(pub bool);

/// Where the run is: the turn it is on.
#[derive(
    Component, Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, Reflect,
)]
#[reflect(Component)]
pub struct Cursor {
    /// Turns begun so far (the next turn's index).
    pub turn: usize,
}

/// Where a run is between its opening and its ending, one component,
/// replaced by insert (immutable: `On<Insert, RunPhase>` sees every
/// change). A `Ready` run without one has not opened; a run with an ending
/// ([`Settled`], [`Failed`]) has none.
#[derive(Component, Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Reflect)]
#[component(immutable)]
#[serde(rename_all = "snake_case")]
#[reflect(Component)]
pub enum RunPhase {
    /// The run's memory load is out; its first turn waits for it.
    LoadingMemory,
    /// The run wants a turn: `Advance` spawns one and `Assemble` folds it.
    Assembling,
    /// The run's current turn has an effect in flight.
    AwaitingModel,
    /// The run's current turn has its tool batch out: one effect per call,
    /// `ChildOf` the turn; the run goes on when every one has landed.
    ResolvingTools,
}

/// A turn whose batch is out: how many calls it holds. Removed when the
/// batch lands and the results are history.
#[derive(
    Component, Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, Reflect,
)]
#[reflect(Component)]
pub struct Batch {
    /// Calls in the batch.
    pub calls: usize,
}

/// Which of the turn's calls a tool effect entity is: what the result is
/// shaped with. On the effect entity, beside the bus module's components.
#[derive(Component, Debug, Clone, PartialEq, Serialize, Deserialize, Reflect)]
#[reflect(Component)]
pub struct ToolCallSlot {
    /// The call's index among the turn's calls, the result's order.
    pub index: usize,
    /// The call's id, as the model gave it.
    #[reflect(remote = crate::agent::reflect::ToolCallIdReflect)]
    pub id: ToolCallId,
    /// The provider's ids for the call, when the wire had them.
    #[reflect(remote = crate::agent::reflect::ProviderCallIdReflect)]
    pub provider: Option<ProviderCallId>,
    /// The tool's name, as dispatched (a repaired call carries its repair).
    pub name: String,
}

/// The run ended with an answer.
#[derive(
    Component, Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, Reflect,
)]
#[reflect(Component)]
pub struct Settled;

/// The run ended without one.
#[derive(Component, Debug, Clone, PartialEq, Serialize, Deserialize, Reflect)]
#[reflect(Component)]
pub struct Failed(pub Failure);

/// Why a run failed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Reflect)]
#[serde(tag = "failure", rename_all = "snake_case")]
#[reflect(opaque, Debug, PartialEq, Serialize, Deserialize)]
pub enum Failure {
    /// The run's content graph or binary source is invalid.
    Content(content::parts::ContentError),
    /// The model-call budget ran out.
    MaxTurns {
        /// The budget.
        limit: usize,
    },
    /// The model called a tool the program does not advertise, and nothing
    /// resolved it.
    UnknownToolCall {
        /// The tool's name.
        name: String,
    },
    /// The completion failed.
    Provider(ErrorReport),
    /// The run was cancelled: the effect in flight was despawned or its
    /// stream dropped.
    Cancelled(ErrorReport),
    /// The run needs what a later stage brings (a tool dispatch, a
    /// resolution kind); named, never silent.
    Unsupported(String),
    /// A granted tool conflicts with the reserved or already minted output name.
    OutputToolCollision {
        /// The name.
        name: String,
    },
    /// A tool call could not be served — the bus closed, the handler gone,
    /// a replay divergence — and the run fails with the report rather
    /// than telling the model its tool failed.
    Tool(ErrorReport),
    /// The conversation could not be loaded: the run fails at the memory
    /// record, before any completion.
    Memory(ErrorReport),
}

/// The run's answer.
#[derive(Component, Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Reflect)]
#[reflect(Component)]
pub struct RunResult(pub String);

/// The run's token usage, summed over its completions.
#[derive(
    Component, Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, Reflect,
)]
#[reflect(Component)]
pub struct Usage(#[reflect(remote = crate::agent::reflect::UsageReflect)] pub WireUsage);

/// Output-tool reprompts spent.
#[derive(
    Component, Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, Reflect,
)]
#[reflect(Component)]
pub struct OutputRetries(pub usize);

/// Invalid-call retries spent.
#[derive(
    Component, Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, Reflect,
)]
#[reflect(Component)]
pub struct InvalidRetries(pub usize);

/// Provider retries spent ([`ProviderRetries`]).
#[derive(
    Component, Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, Reflect,
)]
#[reflect(Component)]
pub struct ProviderRetried(pub usize);

/// The run's next turn re-issues the completion its last turn lost to a
/// retryable provider failure: `Advance` spawns it without counting it
/// against `MaxTurns`, then removes this.
#[derive(
    Component, Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, Reflect,
)]
#[reflect(Component)]
pub struct ProviderRetrying;

/// The name the run's output tool was minted under, once a turn minted it.
#[derive(Component, Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, Reflect)]
#[reflect(Component)]
pub struct OutputToolName(pub Option<String>);

/// A turn: one model call of a run, `ChildOf` the run, in sibling (`Children`) order.
#[derive(
    Component, Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, Reflect,
)]
#[reflect(Component)]
pub struct Turn;

/// An advert: a link entity, `ChildOf` a turn, naming one tool the turn's
/// request advertised, in sibling (`Children`) order.
#[derive(Component, Debug, Clone, Copy, PartialEq, Eq, Reflect)]
#[relationship(relationship_target = AdvertisedOn)]
#[reflect(Component)]
pub struct Advert(pub Entity);

/// The adverts naming this tool.
#[derive(Component, Debug, Default, Reflect)]
#[relationship_target(relationship = Advert)]
#[reflect(Component)]
pub struct AdvertisedOn(Vec<Entity>);

/// The turn's folded assistant content, as it lands (per tick for a
/// stream).
#[derive(Component, Debug, Clone, Default, PartialEq, Serialize, Deserialize, Reflect)]
#[reflect(Component)]
pub struct Outputs {
    /// The parts so far.
    #[reflect(remote = crate::agent::reflect::AssistantContentsReflect)]
    pub content: Vec<AssistantContent>,
    /// The provider's message id, when the answer carried one.
    pub message_id: Option<String>,
    /// Whether the answer is complete.
    pub done: bool,
    /// Whether this turn's completed provider usage has entered the run total.
    /// Invalid-call decisions may now precede completion and survive its arrival.
    #[serde(default)]
    pub usage_recorded: bool,
    /// Number of delivered stream events already checked for invalid names.
    #[serde(default)]
    pub stream_validated: usize,
}

/// A stop, written by any system at any moment: the run ends
/// `Failed(Cancelled)` with this reason, its effects never issued are
/// despawned (no record), the ones in flight left to their handler
/// (CONTRACT §9.1). Serde: a checkpoint saved between the write and the
/// read restores the decision.
#[derive(Component, Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Reflect)]
#[reflect(Component)]
pub struct Cancelled(pub String);

/// A retry of a complete, tool-free turn, written on the turn before
/// `Materialise` reads it (CONTRACT §9.4): with feedback, the turn and the
/// feedback become history and another turn begins; without, nothing
/// becomes history and another turn begins.
#[derive(Component, Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Reflect)]
#[reflect(Component)]
pub struct Retry {
    /// What the model is told, if anything.
    pub feedback: Option<String>,
}

/// A per-turn patch of the request, written on the fresh turn before
/// `Assemble` folds it in (CONTRACT §9.3): what a completion-call hook
/// changed about one model call, as data. For entity-targeted edits, attach ordered
/// [`content::parts::RequestPartEdit`] links to this same fresh turn. These cannot
/// be combined with replacement `history`. Two systems patching one turn
/// [`merge`](Self::merge) in schedule order.
#[derive(Component, Debug, Clone, Default, PartialEq, Serialize, Deserialize, Reflect)]
#[reflect(opaque, Component, Debug, PartialEq, Serialize, Deserialize)]
pub struct RequestPatch {
    /// The preamble the system message is built from, instead of the
    /// agent's.
    pub preamble: Option<String>,
    /// The sampling temperature for the turn.
    pub temperature: Option<f64>,
    /// The token budget for the turn.
    pub max_tokens: Option<u64>,
    /// The tool choice for the turn.
    pub tool_choice: Option<ToolChoice>,
    /// The tools advertised, narrowed to these names.
    pub active_tools: Option<Vec<String>>,
    /// Provider parameters for the turn (an object merges over the
    /// agent's, later keys winning).
    pub additional_params: Option<serde_json::Value>,
    /// Documents appended after the turn's attachments.
    pub extra_context: Vec<rig_core::completion::Document>,
    /// The utterances sent instead of the run's (the prompt stays).
    pub history: Option<Vec<MessageParts>>,
}

impl RequestPatch {
    /// `later` merged over this patch: `extra_context` appends, an object
    /// `additional_params` shallow-merges with later keys winning,
    /// `active_tools` intersect, every other field takes the later value
    /// when set.
    #[must_use = "the merged patch is the returned value"]
    pub fn merge(mut self, later: Self) -> Self {
        self.extra_context.extend(later.extra_context);
        self.additional_params = match (self.additional_params.take(), later.additional_params) {
            (Some(base), Some(patch)) if base.is_object() && patch.is_object() => {
                Some(rig_core::json_utils::merge(base, patch))
            }
            (base, patch) => patch.or(base),
        };
        self.preamble = later.preamble.or(self.preamble);
        self.temperature = later.temperature.or(self.temperature);
        self.max_tokens = later.max_tokens.or(self.max_tokens);
        self.tool_choice = later.tool_choice.or(self.tool_choice);
        self.history = later.history.or(self.history);
        self.active_tools = match (self.active_tools.take(), later.active_tools) {
            (Some(earlier), Some(later)) => Some(
                earlier
                    .into_iter()
                    .filter(|name| later.contains(name))
                    .collect(),
            ),
            (earlier, later) => earlier.or(later),
        };
        self
    }
}

/// A reprompt the next turn carries as its last user message: the output
/// tool was not called, or was called without a required field.
#[derive(Component, Debug, Clone, PartialEq, Serialize, Deserialize, Reflect)]
#[reflect(Component)]
pub struct Reprompt(#[reflect(remote = crate::agent::reflect::MessageReflect)] pub Message);

/// A tool call the program does not advertise, `ChildOf` the turn that
/// made it, awaiting a [`Resolution`].
#[derive(Component, Debug, Clone, PartialEq, Serialize, Deserialize, Reflect)]
#[reflect(Component)]
pub struct InvalidCall {
    /// The completion-local correlation identity of the rejected call.
    #[reflect(remote = crate::agent::reflect::ToolCallIdReflect)]
    pub id: ToolCallId,
    /// The tool's name.
    pub name: String,
    /// The arguments, verbatim.
    #[reflect(remote = crate::agent::reflect::JsonReflect)]
    pub arguments: serde_json::Value,
    /// The actual delivered assistant prefix at early stream validation.
    /// Empty for calls discovered from an already completed turn.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    #[reflect(remote = crate::agent::reflect::AssistantContentsReflect)]
    pub prefix: Vec<AssistantContent>,
    /// Position of the early name event in the completion's delivered events.
    /// This distinguishes reused block identifiers within one stream.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream_offset: Option<usize>,
}

/// What to do with an invalid call. Written by a user system before
/// `Materialise` consumes it, else by the default-policy system from the
/// run's [`InvalidCalls`].
#[derive(Component, Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Reflect)]
#[serde(tag = "resolution", rename_all = "snake_case")]
#[reflect(Component)]
pub enum Resolution {
    /// The run fails with `UnknownToolCall`.
    Fail,
    /// The call is dropped from the turn; what is left goes on.
    Ignore,
    /// Ask the model again: the turn and a tool result carrying `feedback`
    /// for the call (and the invalid-peer notice for every other call of
    /// the turn) become history, nothing is dispatched, and another turn
    /// begins — while `InvalidCalls.retries` are left, else the run fails
    /// `UnknownToolCall`.
    Retry {
        /// What the model is told.
        feedback: String,
    },
    /// Rename the call to a granted tool and dispatch it as such.
    Repair {
        /// The tool's name.
        to: String,
    },
    /// Answer the call with `reason` as its result, dispatch nothing of
    /// the turn (every other call gets the invalid-peer notice), and go
    /// on; refused under `tool_choice: none`.
    Skip {
        /// What the model is told.
        reason: String,
    },
}

// Every state component is serde and entity-free: relationships are the
// only holders of an `Entity`, and a checkpoint remaps them.
const _: () = {
    const fn assert_serde<T: Serialize + serde::de::DeserializeOwned>() {}
    assert_serde::<Owner>();
    assert_serde::<Preamble>();
    assert_serde::<Output>();
    assert_serde::<Failed>();
    assert_serde::<Outputs>();
    assert_serde::<InvalidCall>();
    assert_serde::<Resolution>();
    assert_serde::<ToolPolicy>();
    assert_serde::<ToolContextSpec>();
    assert_serde::<Batch>();
    assert_serde::<ToolCallSlot>();
    assert_serde::<Cancelled>();
    assert_serde::<Retry>();
    assert_serde::<RequestPatch>();
    assert_serde::<Conversation>();
    assert_serde::<Retrieval>();
};

/// Fork `run`: a clone of the run entity and its graph (the utterances,
/// the turns — `ChildOf` is linked, so the clone is deep) under the next
/// run number and its own scope, on the same agent. The effects are not
/// carried: an effect's identity (`Seq`, `Issued`, its answer) belongs to
/// the one dispatch it was, so the clone's turn children hold no effect
/// — a turn already read needs none, and a fresh one folds its own.
/// Best-of-n is `fork` n − 1 times and a system that judges the settled
/// runs (design §3.4). Fork a run at rest — between turns — so no turn
/// is waiting on an effect the clone does not have.
pub fn fork(world: &mut World, run: Entity) -> Entity {
    let seq = {
        let mut counter = world.resource_mut::<RunCounter>();
        let seq = counter.0;
        counter.0 += 1;
        seq
    };
    let owner = world
        .get::<RunOf>(run)
        .and_then(|run_of| world.get::<Owner>(run_of.0))
        .map(|owner| owner.0.clone())
        .unwrap_or_default();
    // Cross-sibling relationship hooks cannot run while linked cloning has
    // only reserved (not yet spawned) their target entities. Restore these
    // edges after the clone queue has completed, using its actual entity map.
    use bevy_ecs::entity::{EntityCloner, EntityHashMap};
    use checkpoint::{AssistantForTurns, ResultsForTurns, TurnAssistant, TurnResults};
    let links: Vec<_> = world
        .query::<(
            Entity,
            &ChildOf,
            Option<&TurnAssistant>,
            Option<&TurnResults>,
        )>()
        .iter(world)
        .filter(|(_, parent, _, _)| parent.parent() == run)
        .map(|(turn, _, assistant, results)| (turn, assistant.copied(), results.copied()))
        .collect();
    let clone = world.spawn_empty().id();
    let mut mapped = EntityHashMap::default();
    mapped.insert(run, clone);
    let mut cloner = EntityCloner::build_opt_out(world);
    cloner.linked_cloning(true).deny::<(
        crate::bus::PendingEffect,
        crate::bus::Seq,
        crate::bus::Issued,
        crate::bus::Reserved,
        crate::bus::EffectOutcome,
        crate::bus::ToolInputs,
        crate::bus::ToolOutputs,
        ToolCallSlot,
        TurnAssistant,
        AssistantForTurns,
        TurnResults,
        ResultsForTurns,
    )>();
    cloner.finish().clone_entity_mapped(world, run, &mut mapped);
    world.flush();
    for (turn, assistant, results) in links {
        if let Some(turn) = mapped.get(&turn).copied() {
            if let Some(TurnAssistant(target)) = assistant {
                let target = mapped.get(&target).copied().unwrap_or(target);
                world.entity_mut(turn).insert(TurnAssistant(target));
            }
            if let Some(TurnResults(target)) = results {
                let target = mapped.get(&target).copied().unwrap_or(target);
                world.entity_mut(turn).insert(TurnResults(target));
            }
        }
    }
    world
        .entity_mut(clone)
        .insert((RunSeq(seq), crate::bus::Scope(format!("{owner}/run#{seq}"))));
    clone
}
