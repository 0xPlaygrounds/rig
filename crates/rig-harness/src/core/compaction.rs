//! Compaction: when the conversation nears the model's context window, or
//! the user asks with `/compact`, its older messages are replaced in
//! requests by a structured summary the model writes. The messages stay in
//! the [`Conversation`](super::agent::Conversation), so views still show
//! them: an agent's [`Compacted`] says how many of them requests leave
//! out, and what goes in their place. Its log records the compaction, and a
//! restored session starts from the first message it kept.
//!
//! Automatic compaction first clears old tool outputs, which costs no model
//! call; only when that is not enough does it summarize. The summary is a
//! model call of its own, dispatched and recorded like any other, on a call
//! entity of the turn with a [`Summarizing`], so interrupting the turn
//! cancels it. The state, the request and the clearing are rig-memory's
//! compaction blocks; the coding prompts and the tracked files are here.

use std::borrow::Cow;
use std::ops::{Deref, DerefMut};

use bevy_ecs::prelude::*;
use bevy_reflect::prelude::*;
use rig_core::catalog::ModelSpec;
use rig_core::completion::{CompletionRequest, CompletionResponse, Message};
use rig_core::error::ErrorReport;
use rig_memory::{
    HeuristicTokenCounter, Summarizer, SummaryLimits, SummaryPrompts, SummaryState, TokenCounter,
    TrackArgument, TrackedSet,
};
use serde::{Deserialize, Serialize};

/// Tokens left free below the model's window: past `window - RESERVE` the
/// conversation is compacted before the next call (pi's `reserveTokens`).
pub const RESERVE: u64 = SummaryLimits::DEFAULT.reserve;
/// Tokens of the newest messages a compaction keeps as they are, at most a
/// quarter of the window (pi's `keepRecentTokens`).
const KEEP_RECENT: usize = 20_000;
/// Compactions a turn may make, asked for or not.
pub const MAX_COMPACTIONS: u32 = 2;

/// The agent's rig-memory [`SummaryState`]: which of its messages requests
/// leave out, and the summary sent in their place, with the files the
/// summarized messages read and changed. The default leaves out none.
#[derive(Component, Reflect, Clone, Debug, Default, Serialize, Deserialize)]
#[reflect(opaque, Component, Default, Clone, Debug, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Compacted(pub SummaryState);

impl Deref for Compacted {
    type Target = SummaryState;

    fn deref(&self) -> &SummaryState {
        &self.0
    }
}

impl DerefMut for Compacted {
    fn deref_mut(&mut self) -> &mut SummaryState {
        &mut self.0
    }
}

impl Compacted {
    /// The estimated tokens of the messages a request sends.
    pub fn estimate(&self, messages: &[Message]) -> u64 {
        self.0.estimate(messages, &HeuristicTokenCounter::default()) as u64
    }

    /// Where a new compaction for `spec` should end
    /// ([`SummaryState::cut`]), keeping at least 20k tokens, a quarter of
    /// the window at most.
    pub fn cut(&self, messages: &[Message], spec: &ModelSpec, force: bool) -> Option<usize> {
        let keep = spec.context_window.map_or(KEEP_RECENT, |window| {
            KEEP_RECENT.min(usize::try_from(window / 4).unwrap_or(usize::MAX))
        });
        self.0
            .cut(messages, keep, force, &HeuristicTokenCounter::default())
    }
}

/// Whether a request of `tokens` leaves less than [`RESERVE`] of `spec`'s
/// window free. Never, when the window is not known.
pub fn over_threshold(tokens: u64, spec: &ModelSpec) -> bool {
    spec.context_window
        .is_some_and(|window| tokens > u64::from(window).saturating_sub(RESERVE))
}

/// Why a conversation is compacted.
#[derive(Clone, Debug, PartialEq, Eq, Reflect)]
pub enum CompactReason {
    /// The user asked, with what to focus on, if anything. The turn ends
    /// with the compaction.
    Asked {
        /// What the summary should keep above all; may be empty.
        focus: String,
    },
    /// The next request would leave less than [`RESERVE`] free; the turn
    /// carries on with the model call.
    Threshold,
    /// The model refused the request as too long; the turn carries on with
    /// the model call.
    Overflow,
}

/// Summarize the older messages of the turn's agent.
#[derive(EntityEvent, Reflect, Clone, Debug)]
#[reflect(Event, Clone, Debug)]
pub struct Summarize {
    /// The turn.
    pub entity: Entity,
    /// Why.
    pub reason: CompactReason,
}

/// A summary model call, on a call entity of the turn; its task is a
/// [`Running<Summary>`](super::calls::Running).
#[derive(Component, Clone, Debug)]
pub struct Summarizing {
    /// Why.
    pub reason: CompactReason,
    /// The first message the compaction keeps.
    pub upto: usize,
    /// How many messages the summary replaces anew.
    pub messages: usize,
    /// Their estimated tokens.
    pub tokens: u64,
    /// The files read and changed, with those of earlier compactions.
    pub tracked: Vec<TrackedSet>,
}

/// What a summary call's task returns.
pub struct Summary(pub Result<CompletionResponse, ErrorReport>);

/// What a compaction ending at `upto` summarizes anew: the messages from
/// the last compaction to `upto`, the files they read and changed, and the
/// request for the summary to `spec`. `focus` is what the user asked to
/// keep above all.
pub fn plan(
    compacted: &Compacted,
    messages: &[Message],
    upto: usize,
    spec: &ModelSpec,
    reason: CompactReason,
) -> Result<(Summarizing, CompletionRequest), String> {
    let older = messages
        .get(compacted.upto.min(upto)..upto)
        .ok_or("the conversation is shorter than the compaction")?;
    let mut next = compacted.0.clone();
    next.track(older, TRACKED);
    let focus = match &reason {
        CompactReason::Asked { focus } => focus.trim(),
        _ => "",
    };
    let request = SUMMARIZER
        .request(older, &compacted.summary, focus, spec)
        .map_err(|refusal| refusal.to_string())?;
    let summarizing = Summarizing {
        reason,
        upto,
        messages: older.len(),
        tokens: HeuristicTokenCounter::default().count_all(older) as u64,
        tracked: next.tracked,
    };
    Ok((summarizing, request))
}

/// The files the built-in file tools were called with: `read`'s are read,
/// `edit`'s and `write`'s changed (a later set wins in the summary).
const TRACKED: &[TrackArgument<'static>] = &[
    TrackArgument {
        tool: "read",
        argument: "path",
        set: "read-files",
    },
    TrackArgument {
        tool: "edit",
        argument: "path",
        set: "modified-files",
    },
    TrackArgument {
        tool: "write",
        argument: "path",
        set: "modified-files",
    },
];

/// The summarizer: the coding prompts below, rig-memory's default limits.
const SUMMARIZER: Summarizer = Summarizer {
    prompts: SummaryPrompts {
        system: Cow::Borrowed(SYSTEM_PROMPT),
        initial: Cow::Borrowed(INITIAL_PROMPT),
        update: Cow::Borrowed(UPDATE_PROMPT),
        format: Cow::Borrowed(FORMAT),
    },
    limits: SummaryLimits::DEFAULT,
};

/// The summarizer's system prompt.
const SYSTEM_PROMPT: &str = "You summarize a conversation between a user and a coding agent \
    so that another model can continue the work from the summary alone. Read the \
    conversation and write the summary in the exact format asked for.\n\n\
    Do not continue the conversation. Do not answer questions in it. Output only the summary.";

/// The request for a first summary.
const INITIAL_PROMPT: &str = "The conversation above is to be summarized. Write a structured \
    checkpoint of it that another model will use to continue the work.";

/// The request to fold new messages into an earlier summary.
const UPDATE_PROMPT: &str = "The conversation above is the NEW part of a conversation whose \
    earlier part is summarized in <previous-summary>. Update that summary with it:\n\
    - keep everything in the previous summary that still holds;\n\
    - add the new progress, decisions and context;\n\
    - move items from \"In progress\" to \"Done\" when they were completed;\n\
    - update \"Next steps\" to what is left;\n\
    - drop what is no longer relevant.";

/// The summary's format, after either request.
const FORMAT: &str = "\n\nUse exactly this format:\n\n\
    ## Goal\n\
    [What the user is trying to get done; several items if the session covers several tasks.]\n\n\
    ## Constraints and preferences\n\
    - [What the user asked for or ruled out, or \"(none)\"]\n\n\
    ## Progress\n\
    ### Done\n\
    - [x] [Completed tasks and changes]\n\
    ### In progress\n\
    - [ ] [Current work]\n\
    ### Blocked\n\
    - [What prevents progress, if anything]\n\n\
    ## Key decisions\n\
    - **[Decision]**: [Why]\n\n\
    ## Next steps\n\
    1. [What should happen next, in order]\n\n\
    ## Critical context\n\
    - [Data, examples, commands or references needed to continue, or \"(none)\"]\n\n\
    Keep each section short. Keep exact file paths, function names, commands and error \
    messages.";
