//! Application observers and turn patches; no legacy hooks or agent interpreter.
use crate::ecs_agent::EcsAgent;
use bevy_ecs::prelude::*;
use rig::{
    completion::CompletionModel,
    effect::{EffectKind, Outcome},
};
use rig_ecs::{
    agent::{
        Cursor, DefaultMaxTurns, Outputs, Owner, RequestPatch, RunOf, Temperature, ToolCallSlot,
        Turn,
    },
    bus::{BusSet, EffectOutcome, Issued, PendingEffect, RigSchedule},
    systems::{Fresh, Materialised, RigSet, spawn_run},
};
use std::{
    collections::BTreeSet,
    sync::{Arc, Mutex},
};
#[derive(Clone, Debug)]
pub(super) struct Breadcrumb {
    pub(super) tag: &'static str,
    pub(super) turn: usize,
}
#[derive(Clone, Default)]
pub(super) struct EventTap {
    breadcrumbs: Arc<Mutex<Vec<Breadcrumb>>>,
    run_ids: Arc<Mutex<BTreeSet<String>>>,
    streaming: Arc<Mutex<Option<bool>>>,
    agent_name: Arc<Mutex<Option<String>>>,
    call_ids: Arc<Mutex<Vec<String>>>,
    result_ids: Arc<Mutex<Vec<String>>>,
}
impl EventTap {
    pub(super) fn breadcrumbs(&self) -> Vec<Breadcrumb> {
        self.breadcrumbs.lock().expect("crumbs").clone()
    }
    pub fn distinct_run_ids(&self) -> usize {
        self.run_ids.lock().expect("run ids").len()
    }
    pub fn is_streaming(&self) -> Option<bool> {
        *self.streaming.lock().expect("streaming")
    }
    pub fn agent_name(&self) -> Option<String> {
        self.agent_name.lock().expect("name").clone()
    }
    pub fn count(&self, tag: &str) -> usize {
        self.breadcrumbs
            .lock()
            .expect("crumbs")
            .iter()
            .filter(|c| c.tag == tag)
            .count()
    }
    pub fn distinct_turns(&self) -> Vec<usize> {
        self.breadcrumbs
            .lock()
            .expect("crumbs")
            .iter()
            .map(|c| c.turn)
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect()
    }
    pub fn call_ids(&self) -> Vec<String> {
        self.call_ids.lock().expect("calls").clone()
    }
    pub fn result_ids(&self) -> Vec<String> {
        self.result_ids.lock().expect("results").clone()
    }
    fn record(
        &self,
        run: Entity,
        turn: usize,
        streamed: bool,
        name: Option<&str>,
        tag: &'static str,
    ) {
        self.run_ids
            .lock()
            .expect("run ids")
            .insert(format!("{run:?}"));
        *self.streaming.lock().expect("streaming") = Some(streamed);
        *self.agent_name.lock().expect("name") = name.map(str::to_owned);
        self.breadcrumbs
            .lock()
            .expect("crumbs")
            .push(Breadcrumb { tag, turn });
    }
}
#[derive(Clone, Default)]
pub(super) struct ScratchpadReader(Arc<Mutex<Vec<usize>>>);
impl ScratchpadReader {
    pub fn tallies(&self) -> Vec<usize> {
        self.0.lock().expect("tallies").clone()
    }
}
#[derive(Component, Default)]
struct AgentTaps(Vec<EventTap>);
#[derive(Component, Default)]
struct RunTaps(Vec<EventTap>);
#[derive(Component, Default)]
struct Readers(Vec<ScratchpadReader>);
#[derive(Component, Default)]
struct Tally(usize);
#[derive(Component)]
struct ModelTurnObserved;
#[derive(Component)]
struct ConfiguredName(String);
#[derive(Resource, Default)]
struct PatchCount(usize);
#[derive(SystemSet, Debug, Hash, PartialEq, Eq, Clone)]
struct PatchSlot(usize);

pub(super) fn agent(
    model: impl CompletionModel + 'static,
    preamble: &str,
    name: Option<&str>,
    temperature: Option<f64>,
) -> EcsAgent {
    let mut ecs = EcsAgent::new(model, preamble, 1);
    ecs.app.world_mut().entity_mut(ecs.agent).insert((
        DefaultMaxTurns(None),
        Temperature(temperature),
        AgentTaps::default(),
    ));
    if let Some(name) = name {
        ecs.app
            .world_mut()
            .entity_mut(ecs.agent)
            .insert((Owner(name.into()), ConfiguredName(name.into())));
    }
    ecs.app.init_resource::<PatchCount>().add_systems(
        RigSchedule,
        (
            observe_calls.after(RigSet::Assemble).before(RigSet::Patch),
            observe_issued
                .after(BusSet::Dispatch)
                .before(BusSet::Collect),
            observe_results.after(BusSet::Collect).before(RigSet::Fold),
            observe_response
                .after(BusSet::Collect)
                .before(BusSet::Judge),
            observe_model_turn.after(RigSet::Fold).before(RigSet::Judge),
        ),
    );
    ecs
}
pub(super) fn agent_tap(ecs: &mut EcsAgent, tap: EventTap) {
    ecs.app
        .world_mut()
        .get_mut::<AgentTaps>(ecs.agent)
        .expect("agent taps")
        .0
        .push(tap);
}
pub(super) async fn prompt(
    ecs: &mut EcsAgent,
    prompt: &str,
    max_turns: usize,
    taps: Vec<EventTap>,
    readers: Vec<ScratchpadReader>,
) -> String {
    prompt_with_mode(ecs, prompt, max_turns, false, taps, readers).await
}
pub(super) async fn prompt_with_mode(
    ecs: &mut EcsAgent,
    prompt: &str,
    max_turns: usize,
    streamed: bool,
    taps: Vec<EventTap>,
    readers: Vec<ScratchpadReader>,
) -> String {
    let run = spawn_run(
        ecs.app.world_mut(),
        ecs.agent,
        &[],
        prompt,
        streamed,
        Some(max_turns),
    );
    ecs.app
        .world_mut()
        .entity_mut(run)
        .insert((RunTaps(taps), Readers(readers), Tally::default()));
    ecs.wait_for_success(run).await
}
type FreshTurns<'w, 's> =
    Query<'w, 's, (Entity, &'static ChildOf, Option<&'static RequestPatch>), Added<Fresh>>;
pub(super) fn install_patch(ecs: &mut EcsAgent, patch: RequestPatch, first_only: bool) {
    let slot = ecs.app.world().resource::<PatchCount>().0;
    ecs.app.world_mut().resource_mut::<PatchCount>().0 += 1;
    let system = move |fresh: FreshTurns, runs: Query<&Cursor>, mut commands: Commands| {
        for (entity, parent, previous) in &fresh {
            if !first_only || runs.get(parent.parent()).expect("run cursor").turn == 1 {
                commands
                    .entity(entity)
                    .insert(previous.cloned().unwrap_or_default().merge(patch.clone()));
            }
        }
    };
    ecs.app.add_systems(
        RigSchedule,
        system
            .in_set(PatchSlot(slot))
            .after(RigSet::Select)
            .before(RigSet::Assemble),
    );
    if slot > 0 {
        ecs.app
            .configure_sets(RigSchedule, PatchSlot(slot).after(PatchSlot(slot - 1)));
    }
}
// Each observation follows the actual effect->turn->run graph. Optional display
// name is application metadata; the native owner is also used for unnamed runs.
fn emit(world: &mut World, run: Entity, tag: &'static str, id: Option<String>) {
    let cursor = world.get::<Cursor>(run).expect("run cursor").turn;
    let streamed = world
        .get::<rig_ecs::agent::RunStreaming>(run)
        .expect("run stream mode")
        .0;
    let agent = world.get::<RunOf>(run).expect("agent relationship").0;
    let name = world.get::<ConfiguredName>(agent).map(|n| n.0.clone());
    let taps: Vec<_> = world
        .get::<AgentTaps>(agent)
        .expect("default taps")
        .0
        .iter()
        .chain(world.get::<RunTaps>(run).expect("request taps").0.iter())
        .cloned()
        .collect();
    for tap in taps {
        tap.record(run, cursor, streamed, name.as_deref(), tag);
        if tag == "ToolCall" {
            tap.call_ids
                .lock()
                .expect("call ids")
                .push(id.clone().expect("actual tool id"));
            world.get_mut::<Tally>(run).expect("run tally").0 += 1;
        } else if tag == "ToolResult" {
            tap.result_ids
                .lock()
                .expect("result ids")
                .push(id.clone().expect("actual tool id"));
        }
    }
}
fn run_for(world: &World, effect: Entity) -> Entity {
    let turn = world.get::<ChildOf>(effect).expect("effect turn").parent();
    assert!(world.get::<Turn>(turn).is_some());
    world.get::<ChildOf>(turn).expect("turn run").parent()
}
fn observe_calls(world: &mut World) {
    let effects: Vec<_> = world
        .query_filtered::<(Entity, &PendingEffect), Added<PendingEffect>>()
        .iter(world)
        .filter(|(_, p)| matches!(p.kind, EffectKind::Completion { .. }))
        .map(|(e, _)| e)
        .collect();
    for effect in effects {
        emit(world, run_for(world, effect), "CompletionCall", None);
    }
}
fn observe_issued(world: &mut World) {
    let mut effects: Vec<_> = world
        .query_filtered::<(Entity, &ToolCallSlot), Added<Issued>>()
        .iter(world)
        .map(|(e, s)| (s.index, e, s.id.to_string()))
        .collect();
    effects.sort_by_key(|(i, _, _)| *i);
    for (_, effect, id) in effects {
        emit(world, run_for(world, effect), "ToolCall", Some(id));
    }
}
fn observe_results(world: &mut World) {
    let mut effects: Vec<_> = world
        .query_filtered::<(Entity, &ToolCallSlot), Added<EffectOutcome>>()
        .iter(world)
        .map(|(e, s)| (s.index, e, s.id.to_string()))
        .collect();
    effects.sort_by_key(|(i, _, _)| *i);
    for (_, effect, id) in effects {
        emit(world, run_for(world, effect), "ToolResult", Some(id));
    }
}
fn observe_response(world: &mut World) {
    let effects: Vec<_> = world
        .query_filtered::<(Entity, &EffectOutcome), Added<EffectOutcome>>()
        .iter(world)
        .filter(|(_, o)| matches!(o.0, Ok(Outcome::Completion(_))))
        .map(|(e, _)| e)
        .collect();
    for effect in effects {
        let run = run_for(world, effect);
        emit(world, run, "CompletionResponse", None);
    }
}
fn observe_model_turn(world: &mut World) {
    let turns: Vec<_> = world
        .query_filtered::<(Entity, &ChildOf, &Outputs), (
            With<Turn>,
            Without<Materialised>,
            Without<ModelTurnObserved>,
        )>()
        .iter(world)
        .filter(|(_, _, outputs)| outputs.done)
        .map(|(turn, parent, _)| (turn, parent.parent()))
        .collect();
    for (turn, run) in turns {
        emit(world, run, "ModelTurnFinished", None);
        let tally = world.get::<Tally>(run).expect("run tally").0;
        for reader in &world.get::<Readers>(run).expect("readers").0 {
            reader.0.lock().expect("tallies").push(tally);
        }
        world.entity_mut(turn).insert(ModelTurnObserved);
    }
}
