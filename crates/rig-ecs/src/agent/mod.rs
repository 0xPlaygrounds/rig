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
//! | Agent | an entity with [`Owner`], [`Preamble`], [`Temperature`], [`MaxTokens`], [`AdditionalParams`], [`ToolChoiceSpec`], [`Output`], [`MaxTurns`], [`InvalidCalls`]; [`UsesModel`] → the model's handler entity; [`Grant`] link entities → tool handler entities; [`Context`] link entities → document entities |
//! | Model, Tool | the bus module's handler entities (`Bound`) |
//! | Document | [`DocumentId`], [`DocumentText`], [`DocumentProps`]; attached to a turn by an [`Attachment`] link entity |
//! | Utterance | [`Utterance`] + [`Role`] + content parts ([`Text`], [`Reasoning`], [`ToolCallPart`], [`ToolResultPart`]), [`Order`]; `ChildOf` the run |
//! | Run | [`Run`] + [`RunOf`] → agent; [`RunSeq`]; a phase marker ([`Assembling`], [`AwaitingModel`], [`Settled`], [`Failed`]); [`Cursor`]; [`RunResult`]; [`Usage`]; retries; [`OutputToolName`]; the run's own overrides of the agent's settings |
//! | Turn | [`Turn`], `ChildOf` the run; [`Advert`] link entities → the tools it advertised; [`Attachment`] link entities → the documents it carried; [`Outputs`]; [`Reprompt`] |
//! | Effect | the bus module's, `ChildOf` the turn |
//! | Invalid call | [`InvalidCall`] + [`Resolution`], `ChildOf` the turn |

pub mod scene;

use bevy_ecs::prelude::*;
use rig_core::{
    completion::{
        Usage as WireUsage,
        message::{AssistantContent, Message, ToolChoice, UserContent},
    },
    error::ErrorReport,
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
#[derive(Component, Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Resolution {
    /// The run fails with `UnknownToolCall`.
    Fail,
    /// The call is dropped from the turn.
    Ignore,
    /// Ask the model again (a later PR; refused now).
    Retry,
    /// Repair the call (a later PR; refused now).
    Repair,
    /// Skip the call and answer the model with feedback (a later PR;
    /// refused now).
    Skip,
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
};
