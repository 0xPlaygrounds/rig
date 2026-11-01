//! Plugin corpus: 40. Handoff to a fresh agent (spec `40-handoff.md`).
//!
//! `/handoff [--now] <next goal>` asks the agent's model, or the model the
//! `[handoff]` settings name, for a brief of its conversation for that
//! goal. The brief is a [`ModelRequest`] of a turn of the agent's own, made
//! for the purpose `handoff`, so its usage counts like any call and an
//! interrupt cancels it. A new top-level agent on the same model, reasoning
//! setting, system prompt and tools then starts from the brief alone, shown
//! in the terminal view: the brief waits in the input line to be edited and
//! sent, or is sent at once with `--now` (and in a run without the view),
//! by the old agent, titled with the goal.
//! The old agent stays, idle; the new one is [`HandedOffFrom`] it, which
//! names it, and a handoff moves the agent shown ([`ShownAgent`], in any front)
//! from the old agent to the new one, when it happens and again when the
//! session is restored.

use std::borrow::Cow;

use rig_harness::prelude::*;
use rig_memory::{CompactReason, Summarizer, SummaryLimits, SummaryPrompts};
use rig_tui::MainInput;
use serde::Deserialize;

/// The purpose the brief's call is made for, in usage.
pub const PURPOSE: &str = "handoff";

/// The longest brief asked for, in tokens.
const BRIEF_TOKENS: u64 = 4_000;

/// What the brief's model is told.
const SYSTEM: &str = "You write a handoff brief. A fresh coding agent that has not seen the \
    conversation will start from your brief alone and work on the next goal the user names. \
    Do not continue the conversation and do not answer questions in it. Output only the brief.";

/// rig-memory's summarizer, asked for a brief for the goal it is given
/// as its focus.
const BRIEF_WRITER: Summarizer = Summarizer {
    prompts: SummaryPrompts {
        system: Cow::Borrowed(SYSTEM),
        initial: Cow::Borrowed(ASK),
        update: Cow::Borrowed(ASK),
        format: Cow::Borrowed(FORMAT),
        ..SummaryPrompts::DEFAULT
    },
    limits: SummaryLimits {
        summary_tokens: BRIEF_TOKENS,
        ..SummaryLimits::DEFAULT
    },
};

/// What the brief's model is asked, before the goal.
const ASK: &str = "The conversation is being handed off. Write the brief for the agent \
    that takes over, for the next goal, given below as what the user asked the summary to \
    keep.";

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
        app.add_setting::<HandoffSettings>("handoff")
            .init_resource::<ShownAgent>()
            .add_command(
                "handoff",
                "Start a fresh agent from a brief for the next goal; --now sends it unedited",
                handoff,
            )
            .add_systems(Update, show_successor.after(AgentLoop))
            .add_observer(take_brief)
            .add_observer(follow_successor::<Briefing>)
            .add_observer(name_successor);
    }

    fn finish(&self, app: &mut App) {
        let editable = app.runs_front(rig_tui::FRONT);
        app.insert_resource(EditableBrief(editable));
    }
}

/// The `[handoff]` settings table: `model = "vendor/model"` writes briefs
/// on that catalog model rather than the agent's own.
#[derive(Resource, Deserialize, Clone, Debug, Default)]
#[serde(default, deny_unknown_fields)]
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
/// to the new agent unedited. It follows the call to a successor that
/// sends it again.
#[derive(Component, Clone, Debug)]
pub struct Briefing {
    /// The next goal.
    pub goal: String,
    /// Sent without the editing step.
    pub now: bool,
}

/// Whether the terminal view runs, whose input line a brief can wait in.
#[derive(Resource)]
struct EditableBrief(bool);

/// `/handoff [--now] <next goal>`: starts the brief's call in a turn of
/// the agent's own. Refused while a turn runs.
fn handoff(
    In(args): In<CommandArgs>,
    agents: Query<(&Conversation, Option<&Connection>, Has<ActiveTurn>)>,
    settings: Res<HandoffSettings>,
    models: Res<Models>,
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
    let Ok((conversation, own, busy)) = agents.get(agent) else {
        return;
    };
    if busy {
        notices.write(Notice::turn_running(agent));
        return;
    }
    let summarizer = match settings.model.as_deref().map(|model| models.connect(model)) {
        Some(Ok(connection)) => Some(connection),
        Some(Err(why)) => {
            notices.write(Notice::error(agent, why));
            return;
        }
        None => None,
    };
    let Some(spec) = summarizer.as_ref().or(own).map(|c| c.spec.clone()) else {
        notices.write(Notice::no_model(agent));
        return;
    };
    if conversation.messages().is_empty() {
        notices.write(Notice::info(agent, "Nothing to hand off yet."));
        return;
    }
    let brief = CompactReason::Asked {
        focus: goal.clone(),
    };
    let request = match BRIEF_WRITER.request(conversation.messages(), "", &brief, &spec) {
        Ok(request) => request,
        Err(why) => {
            notices.write(Notice::error(agent, format!("Cannot hand off: {why}.")));
            return;
        }
    };
    let request = match summarizer {
        Some(summarizer) => ModelRequest::new(request).on(summarizer),
        None => ModelRequest::new(request),
    };
    let turn = commands.spawn((Name::new("handoff"), TurnOf(agent))).id();
    commands.spawn((
        Name::new("handoff brief"),
        Briefing { goal, now },
        request,
        Effect::default().with_purpose(PURPOSE),
        ForAgent(agent),
        CallOf(turn),
    ));
}

/// The settings a new agent takes over from the one handed off from.
type Inherited<'a> = (
    &'a AgentId,
    Option<&'a ModelChoice>,
    &'a ReasoningSetting,
    &'a SystemPrompt,
    &'a ToolAccess,
    Option<&'a Connection>,
);

/// Takes a settled brief: ends the handoff's turn and starts the new
/// agent with the old one's settings, with the brief in the input line,
/// or sent the brief at once by the old agent.
fn take_brief(
    settled: On<Add<Settled>>,
    calls: Query<(&CallOf, &ForAgent, &Briefing, &Outcome)>,
    agents: Query<Inherited>,
    editable: Res<EditableBrief>,
    input: Option<MainInput>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let Ok((_, &ForAgent(agent), briefing, outcome)) = calls.get(settled.entity) else {
        return;
    };
    commands.trigger(StopTurn::plugin_alone(agent, "handoff", "handed off"));
    let brief = match (&outcome.0, outcome.completion()) {
        (Err(report), _) => Err(report.to_string()),
        (Ok(_), None) => Err("the model did not answer with a completion".to_owned()),
        (Ok(_), Some(response)) => {
            Summarizer::summary_text(response).map_err(|why| why.to_string())
        }
    };
    let brief = match brief {
        Ok(brief) => brief,
        Err(why) => {
            notices.write(Notice::error(agent, format!("Handoff failed: {why}.")));
            return;
        }
    };
    let Ok((id, model, effort, prompt, access, connection)) = agents.get(agent) else {
        return;
    };
    let from = HandedOffFrom {
        agent: id.clone(),
        goal: briefing.goal.clone(),
    };
    let mut successor = commands.spawn_empty();
    // Before the model choice, so the shared connection is kept.
    if let Some(connection) = connection {
        successor.insert(connection.clone());
    }
    successor.insert((Agent, *effort, prompt.clone(), access.clone(), from));
    if let Some(model) = model {
        successor.insert(model.clone());
    }
    let successor = successor.id();
    match input.filter(|_| editable.0 && !briefing.now) {
        Some(mut input) => {
            input.clear();
            input.set(brief);
        }
        None => {
            let author = Author::agent(id.0.clone(), None).titled(briefing.goal.clone());
            let brief = Deliver::new(successor, brief, DeliveryMode::Queue).with_author(author);
            commands.trigger(brief);
        }
    }
    notices.write(Notice::info(
        agent,
        "Handed off to a new agent; this one stays idle.",
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

/// Moves the agent shown (or the primary one, while none is) along the
/// handoffs from it to the new agents taking over: the successor just
/// started, or, when a session is restored, the end of its handoff chain.
fn show_successor(
    new: Query<(Entity, &HandedOffFrom), Added<HandedOffFrom>>,
    (main, ids): (MainAgentQuery, Query<&AgentId>),
    mut shown: ResMut<ShownAgent>,
) {
    if new.is_empty() {
        return;
    }
    let mut at = shown.0.or_else(|| main_agent(&main));
    // At most one step per new agent.
    for _ in 0..new.iter().count() {
        let id = at.and_then(|agent| ids.get(agent).ok());
        let next = new.iter().find(|(_, from)| Some(&from.agent) == id);
        let Some((successor, _)) = next else {
            break;
        };
        at = Some(successor);
    }
    shown.set_if_neq(ShownAgent(at));
}

#[cfg(test)]
mod tests;
