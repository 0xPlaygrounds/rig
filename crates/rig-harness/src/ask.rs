//! Questions for the user that any front answers: an approval, a choice or a
//! line of text. An [`Ask`] is an open effect, recorded like any and answered
//! by deciding it ([`Decision`]) with [`Answer::outcome`]; the first decision
//! wins. A plugin that waits on the answer spawns it as a hold
//! (`HoldOf(call)`) on what it asks about, so stopping the turn closes it in
//! every front at once.
//!
//! A front that shows questions spawns an [`Answerer`] for its agents, shows
//! the open ones (`Query<(Entity, &Ask), Without<Outcome>>`) and answers in
//! Gate after [`AskSystems`], so a line the user typed wins. A plugin asks only
//! when [`Answerers::can_answer`], else it applies its own policy, such as
//! refusing. While a question that takes text is open, the user's next line to
//! the agent answers it, unless the question is [`inline`](Ask::inline). The
//! plugin guide's
//! [questions section](crate::plugin_guide#asking-the-user-questions-and-pickers)
//! has the rest.
//!
//! ```
//! use rig_harness::prelude::*;
//!
//! /// Asks whether to keep the agent's changes, when someone can answer.
//! fn ask(agent: Entity, answerers: Answerers, mut commands: Commands) {
//!     if answerers.can_answer(Some(agent)) {
//!         let ask = Ask::yes_no("Keep the changes?").body("3 files changed");
//!         commands.spawn((ask, ForAgent(agent)));
//!     }
//! }
//!
//! /// Reads each answer as it comes.
//! fn answered(asks: Query<(&Ask, &Outcome), Added<Outcome>>) {
//!     for (ask, outcome) in &asks {
//!         let keep = Answer::of(outcome).is_some_and(|answer| answer.is("yes"));
//!         println!("{}: {keep}", ask.title);
//!     }
//! }
//! ```

use bevy_app::prelude::*;
use bevy_ecs::lifecycle::HookContext;
use bevy_ecs::prelude::*;
use bevy_ecs::system::SystemParam;
use bevy_ecs::world::DeferredWorld;
use rig_core::conversation::AuthorKind;
use rig_core::effect::{Answered as EffectOutcome, Decider, EffectKind};
use rig_ecs::inbox::Incoming;
use rig_ecs::stage::{
    AgentPass, Decision, Effect, EffectRequest, ForAgent, IncomingMessages, Outcome, PluginEffects,
    ReadyToStart, Settled, Stage,
};

use crate::commands::CommandSystems;
use serde::{Deserialize, Serialize};

/// A question for the user. Spawn it with the agent it is about
/// (`ForAgent`), if any; it is an open effect, recorded with its answer.
/// To change an open question, spawn the new one and insert
/// `SupersededBy(new)` on the old one: the old one settles as superseded
/// and a front shows the new one in its place, keeping the user's ticks.
/// An old one that is a hold releases what it holds as it settles, so move
/// the hold first.
#[derive(Component, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[component(on_add = request)]
#[require(Name = Name::new("ask"))]
pub struct Ask {
    /// The question, in a line.
    pub title: String,
    /// What it is about, such as a command to run; may be empty.
    pub body: String,
    /// The answers to choose from, in order; the first is the default.
    pub choices: Vec<Choice>,
    /// Whether a line of text answers it too, alone or as a note on a
    /// choice.
    pub free_text: bool,
    /// The text a typed answer starts from, for the user to edit.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub initial: Option<String>,
    /// Whether a typed answer may span lines.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub multiline: bool,
    /// Whether a front shows it beside the work, leaving the keys to the
    /// input, rather than in front of it. The user's next line to the
    /// agent is then a message, not its answer.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub inline: bool,
    /// Whether the user may take several of its choices, which the
    /// [`Answer`] then holds.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub multiple: bool,
}

/// One answer an [`Ask`] offers.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Choice {
    /// What an [`Answer`] names it by, such as `yes`.
    pub id: String,
    /// What the user reads, such as "Yes, run it".
    pub label: String,
    /// Whether it starts ticked, in a question that takes several
    /// ([`Ask::multiple`]).
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub checked: bool,
    /// More about it than its label, such as a call's full arguments, which
    /// a front shows when the user asks.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub more: Option<String>,
}

impl Choice {
    /// The choice `id`, read as `label`.
    pub fn new(id: impl Into<String>, label: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            label: label.into(),
            checked: false,
            more: None,
        }
    }

    /// The same choice, ticked at first.
    #[must_use]
    pub fn checked(mut self) -> Self {
        self.checked = true;
        self
    }

    /// The same choice, with `text` to show when the user asks for more.
    #[must_use]
    pub fn more(mut self, text: impl Into<String>) -> Self {
        self.more = Some(text.into());
        self
    }
}

impl Ask {
    /// A question titled `title`, without choices yet.
    pub fn new(title: impl Into<String>) -> Self {
        Self {
            title: title.into(),
            body: String::new(),
            choices: Vec::new(),
            free_text: false,
            initial: None,
            multiline: false,
            inline: false,
            multiple: false,
        }
    }

    /// A question answered `yes` or `no`.
    pub fn yes_no(title: impl Into<String>) -> Self {
        Self::new(title).choice("yes", "Yes").choice("no", "No")
    }

    /// The same question, about `body`.
    pub fn body(mut self, body: impl Into<String>) -> Self {
        self.body = body.into();
        self
    }

    /// The same question, offering `id` read as `label`.
    pub fn choice(self, id: impl Into<String>, label: impl Into<String>) -> Self {
        self.with(Choice::new(id, label))
    }

    /// The same question, offering `choice`.
    pub fn with(mut self, choice: Choice) -> Self {
        self.choices.push(choice);
        self
    }

    /// The same question, also answered by a line of text.
    pub fn free_text(mut self) -> Self {
        self.free_text = true;
        self
    }

    /// The same question, answered by text that starts from `text`, which
    /// the user edits.
    pub fn initial(mut self, text: impl Into<String>) -> Self {
        self.initial = Some(text.into());
        self.free_text = true;
        self
    }

    /// The same question, answered by text that may span lines.
    pub fn multiline(mut self) -> Self {
        self.multiline = true;
        self.free_text = true;
        self
    }

    /// The same question, shown beside the work: the user keeps the input
    /// and answers it when they choose.
    pub fn inline(mut self) -> Self {
        self.inline = true;
        self
    }

    /// The same question, answered by any number of its choices, none
    /// included.
    pub fn multiple(mut self) -> Self {
        self.multiple = true;
        self
    }
}

/// How an [`Ask`] was answered: the choices taken and the text typed,
/// either or both, such as a choice with a note on it.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Answer {
    /// The choices taken, by id in the question's order: one, or any
    /// number for a question that takes several ([`Ask::multiple`]); none
    /// for an answer only typed.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub choices: Vec<String>,
    /// The text the user typed, alone or as a note on the choices, for a
    /// question that takes text ([`Ask::free_text`]).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
}

impl Answer {
    /// The answer that takes the choice `id`.
    pub fn choice(id: impl Into<String>) -> Self {
        Self::choices([id])
    }

    /// The answer that takes the choices `ids`, in the question's order.
    pub fn choices(ids: impl IntoIterator<Item = impl Into<String>>) -> Self {
        Self {
            choices: ids.into_iter().map(Into::into).collect(),
            text: None,
        }
    }

    /// The answer the user typed as `text`.
    pub fn text(text: impl Into<String>) -> Self {
        Self::default().with_text(text)
    }

    /// The same answer, with `text` typed as its note.
    pub fn with_text(mut self, text: impl Into<String>) -> Self {
        self.text = Some(text.into());
        self
    }

    /// The outcome that answers an `Ask` so.
    pub fn outcome(&self) -> Outcome {
        Outcome::custom(serde_json::to_value(self).unwrap_or_default())
    }

    /// The answer an `Ask`'s outcome holds: `None` for one cancelled or
    /// refused.
    pub fn of(outcome: &Outcome) -> Option<Self> {
        match &outcome.0 {
            Ok(EffectOutcome::Custom { payload }) => Self::deserialize(payload).ok(),
            _ => None,
        }
    }

    /// Whether it takes the choice `id`.
    pub fn is(&self, id: &str) -> bool {
        self.choices.iter().any(|choice| choice == id)
    }
}

/// On a front's entity: it shows and answers questions about this agent,
/// or with `None`, about every agent and the app's own.
#[derive(Component, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Answerer {
    /// The agent, or `None` for all.
    pub agent: Option<Entity>,
}

/// Who can answer questions, for a plugin deciding whether to ask.
#[derive(SystemParam)]
pub struct Answerers<'w, 's> {
    answerers: Query<'w, 's, &'static Answerer>,
}

impl Answerers<'_, '_> {
    /// Whether some front answers questions about `agent` (`None`: the
    /// app's own questions).
    pub fn can_answer(&self, agent: Option<Entity>) -> bool {
        self.answerers
            .iter()
            .any(|answerer| answerer.agent.is_none() || answerer.agent == agent)
    }
}

/// Takes a line the user types as the answer to a question that takes text.
pub struct AskPlugin;

/// The Gate system that answers a question with the line the user typed,
/// after [`CommandSystems`]. A front that answers questions in Gate runs
/// after it.
#[derive(SystemSet, Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct AskSystems;

impl Plugin for AskPlugin {
    fn build(&self, app: &mut App) {
        app.configure_sets(
            AgentPass,
            AskSystems.in_set(Stage::Gate).after(CommandSystems),
        )
        .add_systems(AgentPass, answer_with_text.in_set(AskSystems));
    }
}

/// Makes the question an open effect the kernel records.
fn request(mut world: DeferredWorld, context: HookContext) {
    let Some(ask) = world.get::<Ask>(context.entity) else {
        return;
    };
    let kind = EffectKind::Custom {
        kind: "ask".into(),
        payload: serde_json::to_value(ask).unwrap_or_default(),
    };
    world
        .commands()
        .entity(context.entity)
        .insert_if_new(EffectRequest {
            kind,
            handler: None,
        });
}

/// The user's next line to an agent with a question open that takes text
/// answers the oldest such question that is not inline, and goes no
/// further. A message a driver sent through `rigd` is never an answer: a
/// driver answers with `rig/session/answer`, and its message waits for the
/// agent as any other.
fn answer_with_text(
    mut typed: Query<
        (&Incoming, &ForAgent, &mut Decision),
        (
            ReadyToStart,
            IncomingMessages,
            Without<crate::worker::CommandOf>,
        ),
    >,
    mut asks: Query<
        (Entity, &Ask, &ForAgent, &Effect, &mut Decision),
        (PluginEffects, Without<Outcome>, Without<Settled>),
    >,
) {
    for (incoming, &ForAgent(agent), mut taken) in &mut typed {
        let typed_by_user = incoming.author.kind == AuthorKind::User;
        if !typed_by_user || incoming.command().is_some() || taken.is_decided() {
            continue;
        }
        let open = asks
            .iter()
            .filter(|(_, question, to, _, decision)| {
                question.free_text && !question.inline && to.0 == agent && !decision.is_decided()
            })
            .min_by_key(|(_, _, _, effect, _)| effect.id())
            .map(|(ask, ..)| ask);
        let Some(Ok((.., mut decision))) = open.map(|ask| asks.get_mut(ask)) else {
            continue;
        };
        let answer = Answer::text(incoming.text()).outcome();
        decision.decide(Decider::Person, answer);
        taken.decide(Decider::plugin("rig-harness"), Outcome::custom("answer"));
    }
}

#[cfg(test)]
mod tests;
