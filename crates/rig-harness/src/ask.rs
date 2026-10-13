//! Questions for the user that any front answers: an approval, a choice,
//! or a line of text. An [`Ask`] is an open effect: it moves through the
//! stages and is recorded like any, and it is answered by deciding it
//! ([`Decision`]) with [`Answer::outcome`], the first decision first. A plugin that
//! waits on the answer usually spawns the `Ask` as a hold
//! (`HoldOf(call)`) on what it asks about, so the call waits, and an
//! interrupt, which despawns its turn's holds, closes the question in
//! every front at once.
//!
//! A front that can show questions spawns an [`Answerer`] for the agents
//! it shows, shows the open ones (`Query<(Entity, &Ask), Without<Outcome>>`),
//! and answers them; once one is answered, `Added<Outcome>` on it closes
//! it in the others. A front that answers in Gate does so after
//! [`AskSystems`], so a line the user typed wins. A plugin asks only when [`Answerers::can_answer`] for
//! the agent, else it applies its own policy for a run nobody answers,
//! such as refusing. While a question that takes text is open, the next
//! line the user sends the agent is its answer, not a message to the
//! model, unless the question is [`inline`](Ask::inline).
//!
//! A question may propose an answer for the user to edit
//! ([`Ask::initial`]), take an answer of several lines
//! ([`Ask::multiline`]), and be shown beside the work rather than in front
//! of it ([`Ask::inline`]): the user keeps typing to the agent and answers
//! it when they choose. A front that knows none of these shows it as an
//! ordinary question.
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
use rig_core::effect::{Decider, EffectKind, Outcome as EffectOutcome};
use rig_ecs::inbox::Incoming;
use rig_ecs::stage::{
    AgentPass, Decision, Effect, EffectRequest, ForAgent, IncomingMessages, Outcome, PluginEffects,
    Settled, Stage, Undecided,
};

use crate::commands::CommandSystems;
use serde::{Deserialize, Serialize};

/// A question for the user. Spawn it with the agent it is about
/// (`ForAgent`), if any; it is an open effect, recorded with its answer.
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
    /// Whether a line of text answers it too.
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
}

/// One answer an [`Ask`] offers.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Choice {
    /// What [`Answer::Choice`] names, such as `yes`.
    pub id: String,
    /// What the user reads, such as "Yes, run it".
    pub label: String,
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
    pub fn choice(mut self, id: impl Into<String>, label: impl Into<String>) -> Self {
        self.choices.push(Choice {
            id: id.into(),
            label: label.into(),
        });
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
}

/// How an [`Ask`] was answered.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Answer {
    /// One of its choices, by id.
    Choice(String),
    /// The text the user gave.
    Text(String),
}

impl Answer {
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

    /// Whether this is the choice `id`.
    pub fn is(&self, id: &str) -> bool {
        matches!(self, Self::Choice(choice) if choice == id)
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
/// further.
fn answer_with_text(
    mut typed: Query<(&Incoming, &ForAgent, &mut Decision), (Undecided, IncomingMessages)>,
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
        let answer = Answer::Text(incoming.text.clone()).outcome();
        decision.decide(Decider::Person, answer);
        taken.decide(Decider::plugin("rig-harness"), Outcome::custom("answer"));
    }
}

#[cfg(test)]
mod tests;
