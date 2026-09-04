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
//! | Utterance | [`Utterance`] + [`Role`] + [`Parts`] (the message's parts, verbatim), [`Order`]; `ChildOf` the run |
//! | Run | [`Run`] + [`RunOf`] → agent; [`RunSeq`]; a phase marker ([`Assembling`], [`AwaitingModel`], [`ResolvingTools`], [`Settled`], [`Failed`]); [`Cursor`]; [`RunResult`]; [`Usage`]; retries; [`OutputToolName`]; the run's own overrides of the agent's settings ([`ToolPolicy`], [`ToolContextSpec`] among them) |
//! | Turn | [`Turn`], `ChildOf` the run; [`Advert`] link entities → the tools it advertised; [`Attachment`] link entities → the documents it carried; [`Outputs`]; [`Reprompt`]; [`Batch`] while its tool calls are out |
//! | Effect | the bus module's, `ChildOf` the turn: the completion, then one per tool call ([`ToolCallSlot`] names which) |
//! | Invalid call | [`InvalidCall`] + [`Resolution`], `ChildOf` the turn |

pub mod scene;

use bevy_ecs::prelude::*;
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
#[derive(Component, Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Owner(pub String);

/// The system prompt: `None` is "no system message" (a run without a
/// preamble), `Some("")` an empty one that is still sent.
#[derive(Component, Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Preamble(pub Option<String>);

/// Sampling temperature, if the request names one.
#[derive(Component, Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct Temperature(pub Option<f64>);

/// The answer's token budget, if the request names one.
#[derive(Component, Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct MaxTokens(pub Option<u64>);

/// Provider-specific parameters the request carries verbatim.
#[derive(Component, Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AdditionalParams(pub Option<serde_json::Value>);

/// The program's tool choice: what the request's `tool_choice` starts
/// from before the output mode has its say.
#[derive(Component, Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ToolChoiceSpec(pub Option<ToolChoice>);

/// How the run's answer is asked for and read.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
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
#[derive(Component, Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Output {
    /// The mode.
    pub mode: OutputKind,
    /// The JSON schema of the answer, raw.
    pub schema: Option<serde_json::Value>,
}

/// The model-call budget of a run.
#[derive(Component, Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct MaxTurns(pub usize);

/// What to do with a tool call the program does not advertise when no
/// system resolved it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Unhandled {
    /// The run fails at the record.
    #[default]
    Fail,
    /// The call is dropped; the run goes on.
    Ignore,
}

/// The invalid-call policy: how many retries before `unhandled` applies.
#[derive(Component, Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct InvalidCalls {
    /// Retries the program allows an invalid call.
    pub retries: usize,
    /// What happens when they are spent.
    pub unhandled: Unhandled,
}

/// How many of a turn's tool calls may be in flight at once: 1 (the
/// default) runs them one after another in call order; N keeps N going.
/// The run's, else the agent's.
#[derive(Component, Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
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
#[derive(Component, Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolContextSpec(pub ToolContext);

/// The default `max_turns` the agent was built with, part of its identity
/// (a run-level override is not).
#[derive(Component, Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DefaultMaxTurns(pub Option<usize>);

/// The agent uses this model: a relationship to the model's handler
/// entity (the bus module's `Bound`). A run may carry its own to override.
#[derive(Component, Debug, Clone, Copy, PartialEq, Eq)]
#[relationship(relationship_target = ModelOf)]
pub struct UsesModel(pub Entity);

/// The agents and runs using this model.
#[derive(Component, Debug, Default)]
#[relationship_target(relationship = UsesModel)]
pub struct ModelOf(Vec<Entity>);

impl ModelOf {
    /// Who uses the model.
    pub fn users(&self) -> &[Entity] {
        &self.0
    }
}

/// A route: a link entity, `ChildOf` the agent, naming another model the
/// agent may be steered to (a system inserting [`UsesModel`] on the run);
/// the required row names it.
#[derive(Component, Debug, Clone, Copy, PartialEq, Eq)]
#[relationship(relationship_target = RoutedTo)]
pub struct Route(pub Entity);

/// The routes naming this model.
#[derive(Component, Debug, Default)]
#[relationship_target(relationship = Route)]
pub struct RoutedTo(Vec<Entity>);

/// A grant: a link entity, `ChildOf` the agent, naming one tool the agent
/// advertises. Advertisement order is [`Order`].
#[derive(Component, Debug, Clone, Copy, PartialEq, Eq)]
#[relationship(relationship_target = Grants)]
pub struct Grant(pub Entity);

/// The grants naming this tool.
#[derive(Component, Debug, Default)]
#[relationship_target(relationship = Grant)]
pub struct Grants(Vec<Entity>);

/// A context link: a link entity, `ChildOf` the agent, naming one document
/// every turn carries as static context, in [`Order`].
#[derive(Component, Debug, Clone, Copy, PartialEq, Eq)]
#[relationship(relationship_target = ContextOf)]
pub struct Context(pub Entity);

/// The context links naming this document.
#[derive(Component, Debug, Default)]
#[relationship_target(relationship = Context)]
pub struct ContextOf(Vec<Entity>);

/// The order of a link, an utterance or a turn among its siblings: the
/// agent modules' own counter ([`OrderCounter`]), never the bus module's
/// `Seq`, which is reserved for effects.
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
    Hash,
    Serialize,
    Deserialize,
)]
pub struct Order(pub u64);

/// The world's one order counter for [`Order`].
#[derive(Resource, Debug, Default)]
pub struct OrderCounter(pub u64);

// ---------------------------------------------------------------------------
// Documents: entities of their own, shared by attachment.

/// The document's stable id.
#[derive(Component, Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DocumentId(pub String);

/// The document's text.
#[derive(Component, Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DocumentText(pub String);

/// The document's string metadata, rendered before its text.
#[derive(Component, Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DocumentProps(pub std::collections::HashMap<String, String>);

/// An attachment: a link entity, `ChildOf` a turn, naming one document the
/// turn's request carries, in [`Order`].
#[derive(Component, Debug, Clone, Copy, PartialEq, Eq)]
#[relationship(relationship_target = AttachedTo)]
pub struct Attachment(pub Entity);

/// The attachments naming this document.
#[derive(Component, Debug, Default)]
#[relationship_target(relationship = Attachment)]
pub struct AttachedTo(Vec<Entity>);

impl AttachedTo {
    /// The attachment link entities.
    pub fn attachments(&self) -> &[Entity] {
        &self.0
    }
}

// ---------------------------------------------------------------------------
// Utterances: the conversation, one entity per message, `ChildOf` the run.

/// An utterance: one message of the conversation, `ChildOf` its run, in
/// [`Order`]. Its parts are content components.
#[derive(Component, Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Utterance;

/// Who spoke.
#[derive(Component, Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    /// The user, or a tool result the user side reports.
    User,
    /// The model.
    Assistant,
}

/// The utterance's parts, in order, as the wire's content: kept as the
/// wire type so the fold writes the message verbatim.
#[derive(Component, Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Parts(pub MessageParts);

/// The content of one message, by role.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "role", rename_all = "snake_case")]
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

// ---------------------------------------------------------------------------
// Runs and turns.

/// A run: one prompt through the agent to an answer or a failure.
#[derive(Component, Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Run;

/// The run's agent.
#[derive(Component, Debug, Clone, Copy, PartialEq, Eq)]
#[relationship(relationship_target = Runs)]
pub struct RunOf(pub Entity);

/// The agent's runs.
#[derive(Component, Debug, Default)]
#[relationship_target(relationship = RunOf)]
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
    Component, Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize,
)]
pub struct RunSeq(pub u64);

/// The world's one run counter.
#[derive(Resource, Debug, Default)]
pub struct RunCounter(pub u64);

/// Whether the model is asked for a stream.
#[derive(Component, Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Streamed(pub bool);

/// Where the run is: the turn it is on.
#[derive(Component, Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Cursor {
    /// Turns begun so far (the next turn's index).
    pub turn: usize,
}

/// The run wants a turn: `Advance` spawns one and `Assemble` folds it.
#[derive(Component, Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Assembling;

/// The run's current turn has an effect in flight.
#[derive(Component, Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AwaitingModel;

/// The run's current turn has its tool batch out: one effect per call,
/// `ChildOf` the turn; the run goes on when every one has landed.
#[derive(Component, Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResolvingTools;

/// A turn whose batch is out: how many calls it holds. Removed when the
/// batch lands and the results are history.
#[derive(Component, Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Batch {
    /// Calls in the batch.
    pub calls: usize,
}

/// Which of the turn's calls a tool effect entity is: what the result is
/// shaped with. On the effect entity, beside the bus module's components.
#[derive(Component, Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolCallSlot {
    /// The call's index among the turn's calls, the result's order.
    pub index: usize,
    /// The call's id, as the model gave it.
    pub id: ToolCallId,
    /// The provider's ids for the call, when the wire had them.
    pub provider: Option<ProviderCallId>,
    /// The tool's name, as dispatched (a repaired call carries its repair).
    pub name: String,
}

/// The run ended with an answer.
#[derive(Component, Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Settled;

/// The run ended without one.
#[derive(Component, Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Failed(pub Failure);

/// Why a run failed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "failure", rename_all = "snake_case")]
pub enum Failure {
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
    /// A granted tool took the output tool's name after it was minted.
    OutputToolCollision {
        /// The name.
        name: String,
    },
    /// A tool call could not be served — the bus closed, the handler gone,
    /// a replay divergence — and the run fails with the report rather
    /// than telling the model its tool failed.
    Tool(ErrorReport),
}

/// The run's answer.
#[derive(Component, Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RunResult(pub String);

/// The run's token usage, summed over its completions.
#[derive(Component, Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Usage(pub WireUsage);

/// Output-tool reprompts spent.
#[derive(Component, Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct OutputRetries(pub usize);

/// Invalid-call retries spent.
#[derive(Component, Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct InvalidRetries(pub usize);

/// The name the run's output tool was minted under, once a turn minted it.
#[derive(Component, Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct OutputToolName(pub Option<String>);

/// A turn: one model call of a run, `ChildOf` the run, in [`Order`].
#[derive(Component, Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Turn;

/// An advert: a link entity, `ChildOf` a turn, naming one tool the turn's
/// request advertised, in [`Order`].
#[derive(Component, Debug, Clone, Copy, PartialEq, Eq)]
#[relationship(relationship_target = AdvertisedOn)]
pub struct Advert(pub Entity);

/// The adverts naming this tool.
#[derive(Component, Debug, Default)]
#[relationship_target(relationship = Advert)]
pub struct AdvertisedOn(Vec<Entity>);

/// The turn's folded assistant content, as it lands (per tick for a
/// stream).
#[derive(Component, Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Outputs {
    /// The parts so far.
    pub content: Vec<AssistantContent>,
    /// The provider's message id, when the answer carried one.
    pub message_id: Option<String>,
    /// Whether the answer is complete.
    pub done: bool,
}

/// A stop, written by any system at any moment: the run ends
/// `Failed(Cancelled)` with this reason, its effects never issued are
/// despawned (no record), the ones in flight left to their handler
/// (CONTRACT §9.1). Serde: a scene saved between the write and the read
/// restores the decision.
#[derive(Component, Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Cancelled(pub String);

/// A retry of a complete, tool-free turn, written on the turn before
/// `Materialise` reads it (CONTRACT §9.4): with feedback, the turn and the
/// feedback become history and another turn begins; without, nothing
/// becomes history and another turn begins.
#[derive(Component, Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Retry {
    /// What the model is told, if anything.
    pub feedback: Option<String>,
}

/// A per-turn patch of the request, written on the fresh turn before
/// `Assemble` folds it in (CONTRACT §9.3): what a completion-call hook
/// changed about one model call, as data. Two systems patching one turn
/// [`merge`](Self::merge) in schedule order.
#[derive(Component, Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
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
#[derive(Component, Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Reprompt(pub Message);

/// A tool call the program does not advertise, `ChildOf` the turn that
/// made it, awaiting a [`Resolution`].
#[derive(Component, Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InvalidCall {
    /// The call's id.
    pub id: String,
    /// The tool's name.
    pub name: String,
    /// The arguments, verbatim.
    pub arguments: serde_json::Value,
}

/// What to do with an invalid call. Written by a user system before
/// `Materialise` consumes it, else by the default-policy system from the
/// run's [`InvalidCalls`].
#[derive(Component, Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "resolution", rename_all = "snake_case")]
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
// only holders of an `Entity`, and a scene remaps them.
const _: () = {
    const fn assert_serde<T: Serialize + serde::de::DeserializeOwned>() {}
    assert_serde::<Owner>();
    assert_serde::<Preamble>();
    assert_serde::<Output>();
    assert_serde::<Parts>();
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
};
