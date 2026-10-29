//! The session's options: the main agent's model, from the models rig can
//! reach with the user's sign-in, and its reasoning level, from that
//! model's choices.

use agent_client_protocol::Error;
use agent_client_protocol::schema::v1 as acp;
use bevy_ecs::system::SystemParam;
use rig_harness::prelude::*;

use super::{Reply, Session};

/// The option that picks the model.
const MODEL: &str = "model";

/// The option that picks the reasoning level.
const REASONING: &str = "reasoning";

/// What the options read and change.
#[derive(SystemParam)]
pub(super) struct Options<'w, 's> {
    models: Res<'w, Models>,
    agents: Query<
        'w,
        's,
        (
            Option<&'static ModelChoice>,
            Option<&'static ReasoningSetting>,
            Has<ActiveTurn>,
        ),
    >,
    primary: MainAgentQuery<'w, 's>,
    commands: Commands<'w, 's>,
}

impl Options<'_, '_> {
    /// The options of `agent`: the model, the reachable ones with the chosen
    /// one first, and the chosen model's reasoning levels.
    pub(super) fn of(&self, agent: Entity) -> Vec<acp::SessionConfigOption> {
        let Ok((chosen, reasoning, _)) = self.agents.get(agent) else {
            return Vec::new();
        };
        let catalog = self.models.0.catalog();
        let chosen = chosen.and_then(|choice| catalog.resolve(&choice.0).ok());
        let mut models: Vec<_> = self.models.0.reachable().into_iter().cloned().collect();
        if let Some(chosen) = &chosen {
            models.retain(|spec| spec.reference() != chosen.spec.reference());
            models.insert(0, chosen.spec.clone());
        }
        let Some(current) = models.first() else {
            return Vec::new();
        };
        let entries = models.iter().map(|spec| {
            acp::SessionConfigSelectOption::new(spec.reference(), spec.display_name.clone())
        });
        let model = acp::SessionConfigOption::select(
            MODEL,
            "Model",
            current.reference(),
            entries.collect::<Vec<_>>(),
        )
        .category(acp::SessionConfigOptionCategory::Model);
        let choices = current.reasoning.choices();
        let setting = reasoning.and_then(|setting| setting.0);
        let level = choices
            .iter()
            .find(|choice| choice.reasoning == setting)
            .map_or("default", |choice| choice.name);
        let levels = choices
            .iter()
            .map(|choice| acp::SessionConfigSelectOption::new(choice.name, choice.label()));
        let reasoning = acp::SessionConfigOption::select(
            REASONING,
            "Reasoning",
            level,
            levels.collect::<Vec<_>>(),
        )
        .category(acp::SessionConfigOptionCategory::ThoughtLevel);
        vec![model, reasoning]
    }

    /// Changes the option `change` names, if the value is one it offers and
    /// no turn runs; the answer follows once the model connected or was
    /// refused ([`changed`]).
    pub(super) fn set(
        &mut self,
        change: &acp::SetSessionConfigOptionRequest,
    ) -> Result<(String, String), Error> {
        let invalid = |why: String| Error::invalid_params().data(why);
        let agent = main_agent(&self.primary).ok_or_else(|| invalid("no agent".to_owned()))?;
        let (chosen, _, busy) = self
            .agents
            .get(agent)
            .map_err(|_| invalid("no agent".to_owned()))?;
        if busy {
            return Err(invalid(
                "a turn is running: change it once the turn ended".to_owned(),
            ));
        }
        let value = match &change.value {
            acp::SessionConfigOptionValue::ValueId { value } => value.to_string(),
            other => return Err(invalid(format!("not a choice of a list: {other:?}"))),
        };
        let catalog = self.models.0.catalog();
        match change.config_id.to_string().as_str() {
            MODEL => {
                let found = catalog
                    .resolve(&value)
                    .map_err(|why| invalid(why.to_string()))?;
                let model = found.spec.reference();
                self.commands.trigger(SetModel {
                    entity: agent,
                    model,
                });
            }
            REASONING => {
                let model = chosen.ok_or_else(|| invalid("choose a model first".to_owned()))?;
                let found = catalog
                    .resolve(&model.0)
                    .map_err(|why| invalid(why.to_string()))?;
                let choice = found
                    .spec
                    .reasoning
                    .named(&value)
                    .map_err(|why| invalid(why.to_string()))?;
                let effort = ReasoningSetting(choice.reasoning);
                self.commands.trigger(SetReasoning {
                    entity: agent,
                    effort,
                });
            }
            other => return Err(invalid(format!("no option `{other}`"))),
        }
        Ok((change.config_id.to_string(), value))
    }
}

/// A change the client asked for, answered once it took effect: the
/// option and the value it asked for.
#[derive(Resource)]
pub(super) struct Changing {
    pub(super) reply: Reply<acp::SetSessionConfigOptionResponse>,
    pub(super) option: String,
    pub(super) value: String,
}

/// Answers a change the client asked for, with the options as they are
/// now, or why it did not take effect, such as the refusal of the model it
/// chose; and tells the client the options changed when anything else
/// changed them, such as `/model`.
pub(super) fn changed(
    changing: Option<Res<Changing>>,
    changes: Query<
        (),
        Or<(
            Changed<ModelChoice>,
            Changed<ReasoningSetting>,
            Added<ModelRefused>,
        )>,
    >,
    refusals: Query<&ModelRefused>,
    options: Options,
    mut session: ResMut<Session>,
    mut commands: Commands,
) {
    let Some(agent) = main_agent(&options.primary) else {
        return;
    };
    let current = options.of(agent);
    if let Some(changing) = changing {
        commands.remove_resource::<Changing>();
        // The requests that waited for it go on.
        session.wake.wake();
        let took = current.iter().any(|option| {
            let value = match &option.kind {
                acp::SessionConfigKind::Select(select) => Some(select.current_value.to_string()),
                _ => None,
            };
            option.id.to_string() == changing.option && value.as_ref() == Some(&changing.value)
        });
        let answer = match (took, refusals.get(agent)) {
            (_, Ok(ModelRefused(why))) => Err(Error::invalid_params().data(why.clone())),
            (true, Err(_)) => Ok(acp::SetSessionConfigOptionResponse::new(current)),
            (false, Err(_)) => {
                let why = "rig did not change it: its notice says why";
                Err(Error::invalid_params().data(why))
            }
        };
        changing.reply.send(answer);
        return;
    }
    if session.opening.is_none() && changes.contains(agent) {
        let update = acp::ConfigOptionUpdate::new(current);
        session.send(acp::SessionUpdate::ConfigOptionUpdate(update));
    }
}
