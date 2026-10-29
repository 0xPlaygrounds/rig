//! Questions to the client. An approval asks for permission to run the call
//! it holds; another question is a form, when the client takes forms; else
//! it is closed with a reason the model reads, so the turn goes on. A
//! picker is closed with a notice: the client has a command line.

use agent_client_protocol::schema::v1 as acp;
use bevy_ecs::system::SystemParam;
use rig_basics::approval::{ALLOW, DENY, NeedsApproval};
use rig_harness::pick::{Pick, Picker};
use rig_harness::prelude::*;

use super::output::Out;
use super::{Request, Session};
use crate::translate;

/// Why a question the client cannot show is closed, for the model.
const UNSHOWN: &str = "The client cannot show this question. Go on without an answer: make \
                       reasonable assumptions and say what you assumed.";

/// The form field of a question's choices, and of its typed answer.
const CHOICE: &str = "choice";
const TEXT: &str = "text";

/// What asking a question reads and decides.
#[derive(SystemParam)]
pub(crate) struct Asking<'w, 's> {
    asks: Query<'w, 's, (&'static Ask, Option<&'static HoldOf>, &'static mut Decision)>,
    approvals: Query<'w, 's, (), With<NeedsApproval>>,
    calls: Query<'w, 's, &'static ToolRequest>,
}

impl Asking<'_, '_> {
    /// Asks the client the question on `entity`, if it is one.
    pub(super) fn ask(mut self, entity: Entity, out: &mut Out) {
        let Ok((ask, hold, mut decision)) = self.asks.get_mut(entity) else {
            return;
        };
        let held = hold.map(|&HoldOf(effect)| effect);
        let call = held
            .filter(|effect| self.approvals.contains(*effect))
            .and_then(|effect| self.calls.get(effect).ok());
        out.stream();
        let session = &mut *out.session;
        let (id, cx) = (session.id.clone(), session.cx.clone());
        let (sender, wake) = (session.sender.clone(), session.wake.clone());
        let task = if let Some(request) = call {
            let tool = translate::tool_status(&request.call, acp::ToolCallStatus::Pending);
            let options = vec![
                acp::PermissionOption::new(ALLOW, "Allow", acp::PermissionOptionKind::AllowOnce),
                acp::PermissionOption::new(DENY, "Deny", acp::PermissionOptionKind::RejectOnce),
            ];
            let asked = acp::RequestPermissionRequest::new(id, tool, options);
            session.runtime.spawn(async move {
                let chosen = match cx.send_request(asked).block_task().await {
                    Ok(answer) => match answer.outcome {
                        acp::RequestPermissionOutcome::Selected(selected)
                            if selected.option_id.to_string() == ALLOW =>
                        {
                            ALLOW
                        }
                        _ => DENY,
                    },
                    Err(_) => DENY,
                };
                let answer = Answer {
                    choices: vec![chosen.to_owned()],
                    text: None,
                };
                Session::post(&sender, &wake, Request::Answered(entity, Some(answer)));
            })
        } else if session
            .client
            .elicitation
            .as_ref()
            .is_some_and(acp::ElicitationCapabilities::supports_form)
        {
            let mode =
                acp::ElicitationFormMode::new(acp::ElicitationSessionScope::new(id), form(ask));
            let message = match ask.body.is_empty() {
                true => ask.title.clone(),
                false => format!("{}\n\n{}", ask.title, ask.body),
            };
            let asked = acp::CreateElicitationRequest::new(mode, message);
            let ask = ask.clone();
            session.runtime.spawn(async move {
                let answered = cx.send_request(asked).block_task().await;
                let answer = answered
                    .ok()
                    .and_then(|answered| answer(&ask, answered.action));
                Session::post(&sender, &wake, Request::Answered(entity, answer));
            })
        } else {
            decision.decide(Decider::plugin("rig-acp"), Outcome::refused(UNSHOWN));
            return;
        };
        session.asking.insert(entity, task.abort_handle());
    }
}

/// The form a question is: its choices, one or several, and its typed
/// answer, when it takes one.
fn form(ask: &Ask) -> acp::ElicitationSchema {
    let options = ask
        .choices
        .iter()
        .map(|choice| acp::EnumOption::new(choice.id.clone(), choice.label.clone()));
    let mut schema = acp::ElicitationSchema::new();
    if !ask.choices.is_empty() {
        let required = !ask.free_text;
        schema = match ask.multiple {
            true => schema.property(
                CHOICE,
                acp::MultiSelectPropertySchema::titled(options.collect()),
                required,
            ),
            false => schema.property(
                CHOICE,
                acp::StringPropertySchema::new().one_of(options.collect::<Vec<_>>()),
                required,
            ),
        };
    }
    if ask.free_text {
        let text = acp::StringPropertySchema::new().default_value(ask.initial.clone());
        schema = schema.property(TEXT, text, ask.choices.is_empty());
    }
    schema
}

/// The answer a form's content gives `ask`: `None` when declined,
/// cancelled, or when it names no choice the question offers and has no
/// text it takes.
fn answer(ask: &Ask, action: acp::ElicitationAction) -> Option<Answer> {
    let acp::ElicitationAction::Accept(accepted) = action else {
        return None;
    };
    let content = accepted.content.unwrap_or_default();
    let choices = match content.get(CHOICE) {
        Some(acp::ElicitationContentValue::String(choice)) => vec![choice.clone()],
        Some(acp::ElicitationContentValue::StringArray(choices)) => choices.clone(),
        _ => Vec::new(),
    };
    let offered = |choice: &String| ask.choices.iter().any(|offered| &offered.id == choice);
    if !choices.iter().all(offered) {
        return None;
    }
    let text = match content.get(TEXT) {
        Some(acp::ElicitationContentValue::String(text)) if ask.free_text => Some(text.clone()),
        _ => None,
    };
    (!choices.is_empty() || text.is_some()).then_some(Answer { choices, text })
}

/// Answers the question on `ask` as the client did, or closes it; a
/// question that ended meanwhile, as a stopped turn's, is left alone.
pub(super) fn answer_with(commands: &mut Commands, ask: Entity, answer: Option<Answer>) {
    commands.queue(move |world: &mut World| {
        if world.get::<Settled>(ask).is_some() || world.get::<Ask>(ask).is_none() {
            return;
        }
        match answer {
            Some(answer) => {
                if let Some(mut decision) = world.get_mut::<Decision>(ask) {
                    decision.decide(Decider::Person, answer.outcome());
                }
            }
            None => {
                world.trigger(Pick::person(ask, None));
            }
        }
    });
}

/// Closes a picker, which the client cannot show, saying how to do it
/// with a command instead.
pub(super) fn picker(
    added: On<Add<Picker>>,
    pickers: Query<&Picker>,
    mut session: ResMut<Session>,
    mut commands: Commands,
) {
    let Ok(picker) = pickers.get(added.entity) else {
        return;
    };
    let text = format!(
        "{}: this client cannot show a list to pick from; type the command with what you pick, \
         such as /model <name>.",
        picker.title
    );
    session.note(text, NoticeLevel::Warn);
    commands.trigger(Pick::person(added.entity, None));
}
