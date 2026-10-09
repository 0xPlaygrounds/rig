//! How a coding agent compacts: a summary in the shape of a coding
//! checkpoint, and the files the built-in file tools read and changed,
//! tracked across compactions. The core's [`CompactionPolicy`].

use std::borrow::Cow;

use bevy_app::prelude::*;
use rig_memory::{Summarizer, SummaryLimits, SummaryPrompts, TrackArgument};

use rig_ecs::compaction::CompactionPolicy;

/// Inserts the coding [`CompactionPolicy`].
pub struct CodingCompactionPlugin;

impl Plugin for CodingCompactionPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(CompactionPolicy {
            summarizer: SUMMARIZER,
            tracked: TRACKED.to_vec(),
        });
    }
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
