//! Typed application audit data exercises all admitted JSON answer shapes.

use super::{Case, Error, Host};
use bevy_ecs::prelude::*;
use rig_core::effect::CustomEffect;
use rig_ecs::bus::{Answer, Asked, EffectOutcome, Handlers, PendingEffect, RigSchedule};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

const KEY: &str = "maintenance/custom:audit";
const SCOPE: &str = "maintenance-audit";

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AuditValue(Value);

impl CustomEffect for AuditValue {
    const KIND: &'static str = "maintenance/audit/v1";
    type Answer = Value;
}

fn values() -> [Value; 7] {
    [
        json!("approved"),
        json!(42),
        json!(true),
        Value::Null,
        json!([1, "two", null]),
        json!({"approved":true}),
        json!({"outcome":"user data","payload":[false]}),
    ]
}

pub(super) fn install(world: &mut World, case: &Case) -> Result<(), Error> {
    if case.custom_answers {
        Handlers::with(world, |handlers| handlers.register_world::<AuditValue>(KEY))??;
        world.resource_mut::<Schedules>().add_systems(
            RigSchedule,
            answer
                .after(super::persistence::capture_early)
                .before(super::serve_tools),
        );
    }
    Ok(())
}

type UnansweredAudit = (Without<Answer<AuditValue>>, With<rig_ecs::bus::InFlight>);

fn answer(
    asked: Query<(Entity, &Asked<AuditValue>), UnansweredAudit>,
    mut commands: Commands,
    mut host: ResMut<Host>,
) {
    let count = asked.iter().count();
    if count != 0
        && !host
            .observations
            .iter()
            .any(|item| item.data.get("key") == Some(&json!(KEY)))
    {
        let expected = if host.case.serial_keys {
            1
        } else {
            values().len()
        };
        if count != expected {
            host.failure = Some(format!(
                "same-key serving expected {expected} in-flight audit effects, observed {count}"
            ));
        }
    }
    for (entity, asked) in &asked {
        commands
            .entity(entity)
            .insert(Answer::<AuditValue>(asked.0.0.clone()));
    }
}

pub(super) fn start(world: &mut World, run: Entity, case: &Case) -> Result<(), Error> {
    if case.custom_answers {
        for value in values() {
            world.spawn((
                PendingEffect::custom(KEY, &AuditValue(value))?,
                ChildOf(run),
                rig_ecs::bus::Scope(SCOPE.into()),
            ));
        }
    }
    Ok(())
}

pub(super) fn stamp(
    world: &mut World,
    case: &Case,
    replay: Option<&rig_effect_log::EffectLog>,
) -> Result<(), Error> {
    if !case.custom_answers {
        return Ok(());
    }
    let mut required = rig_core::effect::EffectRow::new();
    required.insert(KEY.into(), rig_core::effect::EffectFamily::Custom);
    let identity = rig_effect_log::ProgramIdentity {
        required,
        policy: rig_effect_log::stable_hash(&json!("maintenance/audit/v1"))?,
    };
    if let Some(log) = replay
        && serde_json::to_value(log.header.programs.get(SCOPE))?
            != serde_json::to_value(Some(&identity))?
    {
        return Err(Error::Invariant(
            "custom audit scope identity differs from recording".into(),
        ));
    }
    world
        .resource::<rig_ecs::bus::EffectLogResource>()
        .0
        .set_program_identity(SCOPE, identity);
    Ok(())
}

pub(super) fn validate(world: &mut World) -> Result<(), Error> {
    if !world.resource::<Host>().case.custom_answers {
        return Ok(());
    }
    let actual: Vec<_> = world
        .query::<(&PendingEffect, &EffectOutcome)>()
        .iter(world)
        .filter(|(effect, _)| effect.key.as_str() == KEY)
        .map(|(_, outcome)| outcome.custom::<AuditValue>())
        .collect::<Result<_, _>>()?;
    let mut expected = values().to_vec();
    for value in actual {
        let Some(index) = expected.iter().position(|expected| expected == &value) else {
            return Err(Error::Invariant(
                "custom audit answer changed or duplicated".into(),
            ));
        };
        expected.remove(index);
    }
    if !expected.is_empty() {
        return Err(Error::Invariant(
            "custom audit answers missing after consumer execution/restore".into(),
        ));
    }
    Ok(())
}
