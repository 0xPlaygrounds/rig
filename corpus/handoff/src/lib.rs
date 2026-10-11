//! Plugin corpus: 40. Handoff to a fresh agent (spec `40-handoff.md`).
//!
//! `/handoff <next goal>` asks the agent's model, or the model named in
//! [`HandoffSettings`], for a brief of its conversation for that goal, in a
//! [`ModelRequest`] of a turn of its own, so the brief's usage counts like
//! any call and an interrupt cancels it. A new top-level agent on the same
//! model, reasoning setting, system prompt and tools then starts from the
//! brief alone: it is put in the input line to edit and send, or sent at
//! once with `--now` (and in a run without the terminal view). The old
//! agent stays, idle; the new one is [`HandedOffFrom`] it, which names it.

use std::borrow::Cow;

use rig_harness::prelude::*;
use rig_memory::{Summarizer, SummaryLimits, SummaryPrompts};
use rig_tui::Focus;

/// The longest brief asked for, in tokens.
const BRIEF_TOKENS: u64 = 4_000;

/// What the brief's model is told.
const SYSTEM: &str = "You write a handoff brief. A fresh coding agent that has not seen the \
    conversation will start from your brief alone and work on the next goal the user names. \
    Do not continue the conversation and do not answer questions in it. Output only the brief.";

/// The brief's format, after the request.
const FORMAT: &str = "\n\nUse exactly this format:\n\n\
    ## Goal\n[The next goal, stated precisely.]\n\n\
    ## Relevant files\n- [path]: [what it holds that matters for the goal]\n\n\
    ## Decisions\n- **[Decision]**: [Why]\n\n\
    ## Open questions\n- [What is still undecided or unknown, or \"(none)\"]\n\n\
    Leave out everything that does not bear on the next goal. Keep exact file paths, names, \
    commands and error messages.";

/// Adds `/handoff`.
#[derive(Default)]
pub struct HandoffPlugin;

impl Plugin for HandoffPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<HandoffSettings>()
            .add_command(
                "handoff",
                "Start a fresh agent from a brief for the next goal; --now sends it unedited",
                handoff,
            )
            .add_observer(take_brief)
            .add_observer(name_successor);
    }
}

/// Which model writes briefs: the catalog model named (`vendor/model`), or
/// the agent's own when `None`.
#[derive(Resource, Clone, Debug, Default)]
pub struct HandoffSettings {
    /// The summarizer's model.
    pub model: Option<String>,
}

/// On an agent started by `/handoff`: the agent it took over from and the
/// goal it was given. Saved with the session; it names the agent.
#[derive(Component, Reflect, Clone, Debug, Default)]
#[reflect(Component, Saved, Default, Clone, Debug)]
pub struct HandedOffFrom {
    /// The agent handed off from.
    pub agent: AgentId,
    /// The goal handed over.
    pub goal: String,
}

/// On the brief's [`ModelRequest`]: the goal, and whether the brief goes
/// to the new agent unedited.
#[derive(Component, Clone, Debug)]
pub struct Briefing {
    /// The next goal.
    pub goal: String,
    /// Sent without the editing step.
    pub now: bool,
}

/// `/handoff [--now] <next goal>`: starts the brief's call in a turn of
/// the agent's own. Refused while a turn runs.
fn handoff(
    In(args): In<CommandArgs>,
    agents: Query<(
        &Conversation,
        Option<&Condensed>,
        Option<&Connection>,
        Has<ActiveTurn>,
    )>,
    settings: Res<HandoffSettings>,
    models: Res<Models>,
    invoked: Option<Res<Invoked>>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let agent = args.agent;
    let (now, goal) = match args.args.trim().strip_prefix("--now") {
        Some(rest) if rest.is_empty() || rest.starts_with(char::is_whitespace) => (true, rest),
        _ => (false, args.args.as_str()),
    };
    let goal = goal.trim().to_owned();
    if goal.is_empty() {
        notices.write(Notice::error(agent, "/handoff needs the next goal."));
        return;
    }
    let Ok((conversation, condensed, own, busy)) = agents.get(agent) else {
        return;
    };
    if busy {
        notices.write(Notice::turn_running(agent));
        return;
    }
    let summarizer = match &settings.model {
        Some(model) => match models.connect(model) {
            Ok(connection) => Some(connection),
            Err(why) => {
                notices.write(Notice::error(agent, why));
                return;
            }
        },
        None => None,
    };
    let Some(spec) = summarizer.as_ref().or(own).map(|c| c.spec.clone()) else {
        notices.write(Notice::no_model(agent));
        return;
    };
    let messages = match condensed {
        Some(condensed) => condensed.request(conversation.messages()),
        None => conversation.messages().to_vec(),
    };
    if messages.is_empty() {
        notices.write(Notice::info(agent, "Nothing to hand off yet."));
        return;
    }
    let request = brief_writer(&goal).request(&messages, "", "", &spec);
    let request = match request {
        Ok(request) => request,
        Err(why) => {
            notices.write(Notice::error(agent, format!("Cannot hand off: {why}.")));
            return;
        }
    };
    // The editing step needs the terminal view's input line.
    let now = now || !invoked.is_some_and(|invoked| invoked.interactive());
    let turn = commands.spawn((Name::new("handoff"), TurnOf(agent))).id();
    let mut call = commands.spawn((
        Name::new("handoff brief"),
        Briefing { goal, now },
        ModelRequest { request },
        CallOf(turn),
    ));
    if let Some(connection) = summarizer {
        call.insert(connection);
    }
}

/// rig-memory's summarizer, asked for a brief for `goal`.
fn brief_writer(goal: &str) -> Summarizer {
    let ask = format!(
        "The conversation above is being handed off. Write the brief for the agent that takes \
         over, for this next goal: {goal}"
    );
    Summarizer {
        prompts: SummaryPrompts {
            system: Cow::Borrowed(SYSTEM),
            initial: Cow::Owned(ask.clone()),
            update: Cow::Owned(ask),
            format: Cow::Borrowed(FORMAT),
        },
        limits: SummaryLimits {
            summary_tokens: BRIEF_TOKENS,
            ..SummaryLimits::DEFAULT
        },
    }
}

/// Takes a finished brief: ends the handoff's turn and starts the new
/// agent with the old one's settings, shown with the brief in the input
/// line, or sent the brief at once.
fn take_brief(
    done: On<Add<Done<ModelReply>>>,
    calls: Query<(&CallOf, &Briefing, &Done<ModelReply>)>,
    turns: Query<&TurnOf>,
    agents: Query<(
        &AgentId,
        Option<&ModelChoice>,
        &Effort,
        &SystemPrompt,
        &ToolAccess,
        Option<&Connection>,
    )>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
    mut recalled: MessageWriter<Recalled>,
) {
    let call = done.entity;
    let Ok((&CallOf(turn), briefing, Done(reply))) = calls.get(call) else {
        return;
    };
    commands.entity(call).despawn();
    commands.entity(turn).despawn();
    let Ok(&TurnOf(agent)) = turns.get(turn) else {
        return;
    };
    let Ok((id, model, effort, prompt, access, connection)) = agents.get(agent) else {
        return;
    };
    let brief = reply
        .as_ref()
        .map_err(ToString::to_string)
        .and_then(|response| Summarizer::summary_text(response).map_err(|why| why.to_string()));
    let brief = match brief {
        Ok(brief) => brief,
        Err(why) => {
            notices.write(Notice::error(agent, format!("Handoff failed: {why}.")));
            return;
        }
    };
    let from = HandedOffFrom {
        agent: id.clone(),
        goal: briefing.goal.clone(),
    };
    let mut successor = commands.spawn_empty();
    // With the model choice, so the shared connection is kept.
    if let Some(connection) = connection {
        successor.insert(connection.clone());
    }
    successor.insert((Agent, *effort, prompt.clone(), access.clone(), from));
    if let Some(model) = model {
        successor.insert(model.clone());
    }
    let successor = successor.id();
    if briefing.now {
        let origin = Origin::agent(id.clone(), None).titled("handoff");
        let deliver = Deliver::new(successor, brief, DeliveryMode::Queue).with_origin(origin);
        commands.trigger(deliver);
    } else {
        // Workaround: the only public way to fill the view's input line.
        recalled.write(Recalled {
            agent: successor,
            text: brief,
            why: None,
        });
    }
    commands.trigger(Focus { entity: successor });
    notices.write(Notice::info(
        agent,
        "Handed off to a new agent; this one stays idle in /agents.",
    ));
}

/// Names an agent [`HandedOffFrom`] another after its goal and that agent,
/// when it is started or restored.
fn name_successor(
    inserted: On<Insert<HandedOffFrom>>,
    from: Query<&HandedOffFrom>,
    mut commands: Commands,
) {
    if let Ok(from) = from.get(inserted.entity) {
        let name = format!("{} (from {})", from.goal, from.agent.short());
        commands.entity(inserted.entity).insert(Name::new(name));
    }
}

#[cfg(test)]
mod tests;
