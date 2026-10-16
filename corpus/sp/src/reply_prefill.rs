//! SP-11: reply prefill (splits fresh held-out spec `11-reply-prefill.md`).
//!
//! `/prefill <text>` puts a [`Prefill`] on the agent the command was typed
//! for: the words its next reply starts with (`/prefill` alone shows it,
//! `/prefill clear` takes it off; a plugin inserts or removes the component
//! the same way). As the agent's next turn call is admitted, the plugin
//! looks at the model it goes to. One that continues a request ending in
//! its own message (the catalog's [`Compat::prefill`] while its reasoning
//! is off, or the `[prefill] models` table of the settings) gets the
//! prefill as a last, unfinished assistant message, and the call's
//! [`Partial`] starts with it, so the view shows it before any token
//! streams. In Review the prefill is joined in front of the reply's text,
//! so the conversation keeps one assistant message, its tool calls as the
//! model made them. Another model is refused at Gate with a notice, the
//! prefill kept (`fallback = "refuse"`, the default), or asked in a last
//! user message to begin its reply with the text (`fallback =
//! "instruct"`). The prefill goes once a reply of a call that carried it
//! is stored; a failed or superseded call keeps it for the next try.
//!
//! [`Compat::prefill`]: rig_harness::rig_core::catalog::Compat::prefill

use std::collections::HashMap;

use rig_harness::prelude::*;
use rig_harness::rig_core::catalog::{ModelSpec, ReasoningSupport};
use rig_harness::rig_core::completion::Reasoning;
use rig_harness::rig_core::effect::Answered;
use rig_harness::rig_core::message::Message;
use serde::Deserialize;

/// The plugin's name, as an author and a decider.
const NAME: &str = "reply_prefill";

/// The settings table.
const SECTION: &str = "prefill";

/// What a model that cannot continue a prefill gets.
#[derive(Deserialize, Clone, Copy, Debug, Default, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Fallback {
    /// The prompt is not sent; a notice says why and the prefill stays.
    #[default]
    Refuse,
    /// The request asks the model to begin its reply with the text.
    Instruct,
}

/// The `[prefill]` table of the settings.
#[derive(Resource, Deserialize, Clone, Debug, Default)]
#[serde(default, deny_unknown_fields)]
pub struct PrefillSettings {
    /// What a model without prefill support gets.
    pub fallback: Fallback,
    /// Whether each model (`vendor/model`) continues a prefill, over what
    /// the catalog says.
    pub models: HashMap<String, bool>,
}

/// On an agent: the text its next reply starts with. Trailing whitespace
/// is not sent, since providers refuse an assistant message ending in it.
#[derive(Component, Reflect, Clone, Debug, PartialEq, Eq)]
#[reflect(Component)]
pub struct Prefill(pub String);

/// On a turn's model call admitted while its agent had a [`Prefill`]: how
/// the prefill went with it.
#[derive(Component, Clone, Debug, PartialEq, Eq)]
pub enum Prefilled {
    /// Sent as an unfinished assistant message, which the reply continues.
    Continued(String),
    /// Asked for in a last user message.
    Instructed(String),
    /// Not sent: the model named cannot continue a prefill.
    Refused {
        /// The model the call goes to.
        model: String,
    },
}

/// The stage systems of the plugin, for other plugins to order against.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PrefillSystems {
    /// Puts the prefill in a turn call's request (`PrepareSystems::Edit`);
    /// a router that changes the call's model goes before it.
    Prepare,
    /// Refuses a call whose model cannot continue it (`Stage::Gate`).
    Refuse,
    /// Joins the prefill in front of the reply (`ReviewSystems::Edit`).
    Join,
}

/// Starts an agent's next reply with a text of the user's or a plugin's.
#[derive(Default)]
pub struct ReplyPrefillPlugin;

impl Plugin for ReplyPrefillPlugin {
    fn build(&self, app: &mut App) {
        app.add_setting::<PrefillSettings>(SECTION)
            .register_type::<Prefill>()
            .configure_sets(
                AgentPass,
                (
                    PrefillSystems::Prepare.in_set(PrepareSystems::Edit),
                    PrefillSystems::Refuse.in_set(Stage::Gate),
                    PrefillSystems::Join.in_set(ReviewSystems::Edit),
                ),
            )
            .add_systems(
                AgentPass,
                (
                    prepare.in_set(PrefillSystems::Prepare),
                    refuse.in_set(PrefillSystems::Refuse),
                    join.in_set(PrefillSystems::Join),
                ),
            )
            .add_observer(clear_once_stored)
            .add_command(
                "prefill",
                "Start the next reply with a text: /prefill <text>, /prefill, /prefill clear",
                prefill,
            );
    }
}

/// On a reply the prefill was joined to (Review sees it each pass).
#[derive(Component)]
#[component(storage = "SparseSet")]
struct Joined;

/// `/prefill <text>` sets it, `/prefill` shows it, `/prefill clear` takes
/// it off.
fn prefill(
    In(args): In<CommandArgs>,
    agents: Query<&Prefill>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let agent = args.agent;
    let text = match args.args.as_str() {
        "" => match agents.get(agent) {
            Ok(Prefill(text)) => format!("The next reply starts with: {text}"),
            Err(_) => "No prefill is set.".to_owned(),
        },
        "clear" => {
            commands.entity(agent).try_remove::<Prefill>();
            "Prefill cleared.".to_owned()
        }
        text => {
            commands.entity(agent).try_insert(Prefill(text.to_owned()));
            format!("The next reply starts with: {text}")
        }
    };
    notices.write(Notice::info(agent, text));
}

/// Whether `spec`'s model continues a request ending in its own message
/// with `reasoning` asked: the settings' word on it, else the catalog's,
/// which holds while reasoning is off.
fn continues(spec: &ModelSpec, reasoning: Option<&Reasoning>, settings: &PrefillSettings) -> bool {
    if let Some(&on) = settings.models.get(&spec.reference()) {
        return on;
    }
    let off = match reasoning {
        Some(reasoning) => matches!(reasoning, Reasoning::Off),
        None => matches!(spec.reasoning, ReasoningSupport::None),
    };
    spec.compat.prefill && off
}

/// The note that asks a model without prefill support for the text.
fn instruction(text: &str) -> String {
    format!(
        "Begin your reply with exactly the following text, character for character, then \
         continue it:\n{text}"
    )
}

/// Puts the agent's prefill in each turn call as it is admitted, by what
/// the model it goes to takes, and marks the call with how.
fn prepare(
    mut calls: Query<
        (Entity, &mut ModelRequest, &mut Partial, &ForAgent),
        (Added<Admitted>, With<TurnCall>),
    >,
    agents: Query<(&Prefill, Option<&Connection>)>,
    settings: Res<PrefillSettings>,
    mut commands: Commands,
) {
    for (call, mut request, mut partial, &ForAgent(agent)) in &mut calls {
        let Ok((Prefill(text), own)) = agents.get(agent) else {
            continue;
        };
        let text = text.trim_end();
        let Some(spec) = request
            .connection
            .as_ref()
            .or(own)
            .map(|it| it.spec.clone())
        else {
            continue;
        };
        if text.is_empty() {
            continue;
        }
        let supported = continues(&spec, request.request.options.reasoning.as_ref(), &settings);
        let Some(history) = request.history.as_mut() else {
            continue;
        };
        let by = Author::plugin(NAME);
        let marker = match (supported, settings.fallback) {
            (true, _) => {
                history.push(Entry::new(Message::assistant(text)).by(by));
                partial.text = text.to_owned();
                Prefilled::Continued(text.to_owned())
            }
            (false, Fallback::Instruct) => {
                history.push(Entry::new(Message::user(instruction(text))).by(by));
                Prefilled::Instructed(text.to_owned())
            }
            (false, Fallback::Refuse) => Prefilled::Refused {
                model: spec.reference(),
            },
        };
        commands.entity(call).insert(marker);
    }
}

/// Refuses a turn call whose model cannot continue the prefill, before it
/// is sent: the turn fails with the reason as its notice and keeps the
/// user's message, and the prefill stays.
fn refuse(
    mut calls: Query<(&Prefilled, &mut Decision), (ReadyToStart, ModelCalls, With<TurnCall>)>,
) {
    for (prefilled, mut decision) in &mut calls {
        let Prefilled::Refused { model } = prefilled else {
            continue;
        };
        let why = format!(
            "{model} cannot continue a prefilled reply, so the prompt was not sent. The prefill \
             is kept: switch to a model that supports it, `/prefill clear`, or set `fallback = \
             \"instruct\"` under [prefill]."
        );
        decision.decide(Decider::plugin(NAME), Outcome::refused(why));
    }
}

/// Joins the prefill in front of the reply that continued it, once: into
/// its first text, else as a text before its calls.
fn join(
    mut calls: Query<
        (Entity, &Prefilled, &mut Outcome),
        (
            ReadyToSettle,
            ModelCalls,
            With<TurnCall>,
            Without<SupersededBy>,
            Without<Joined>,
        ),
    >,
    mut commands: Commands,
) {
    for (call, prefilled, mut outcome) in &mut calls {
        let Prefilled::Continued(text) = prefilled else {
            continue;
        };
        let Ok(Answered::Completion(response)) = &mut outcome.0 else {
            continue;
        };
        let response = std::sync::Arc::make_mut(response);
        let first = response.choice.iter_mut().find_map(|block| match block {
            AssistantContent::Text(text) => Some(text),
            _ => None,
        });
        match first {
            Some(first) => first.text.insert_str(0, text),
            None => {
                let mut blocks = response.choice.iter();
                let at = blocks
                    .position(|block| !matches!(block, AssistantContent::Reasoning(_)))
                    .unwrap_or(response.choice.len());
                response
                    .choice
                    .insert(at, AssistantContent::text(text.clone()));
            }
        }
        commands.entity(call).insert(Joined);
    }
}

/// Takes the agent's prefill off once a reply of a call that carried it
/// settles to be stored, unless the prefill changed meanwhile.
fn clear_once_stored(
    settled: On<Add<Settled>>,
    calls: Query<(&Prefilled, &ForAgent, &Outcome), (With<TurnCall>, Without<SupersededBy>)>,
    agents: Query<&Prefill>,
    mut commands: Commands,
) {
    let Ok((prefilled, &ForAgent(agent), outcome)) = calls.get(settled.entity) else {
        return;
    };
    let (Prefilled::Continued(sent) | Prefilled::Instructed(sent)) = prefilled else {
        return;
    };
    let stored = matches!(outcome.0, Ok(Answered::Completion(_)));
    if stored
        && agents
            .get(agent)
            .is_ok_and(|Prefill(now)| now.trim_end() == sent)
    {
        commands.entity(agent).try_remove::<Prefill>();
    }
}

#[cfg(test)]
mod tests;
