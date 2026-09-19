//! Application policies use fresh turns and native RequestPatch merging.
//! Each registered system acts at the same boundary as its legacy counterpart;
//! no legacy hook or effect replayer participates in execution.
use crate::goldens::{PIRATE_PREAMBLE, SHAPING_CONTEXT, SHAPING_CONTEXT_ID};
use bevy_ecs::prelude::*;
use rig_core::{
    completion::Document,
    message::{Message, ToolChoice},
};
use rig_ecs::{
    agent::{Cursor, MessageParts, RequestPatch, RunOf, UsesModel},
    systems::Fresh,
};

type FreshTurns<'w, 's> =
    Query<'w, 's, (Entity, &'static ChildOf, Option<&'static RequestPatch>), Added<Fresh>>;

// Expand concrete systems, sharing only traversal and native merge semantics.
macro_rules! patch_system {
    ($name:ident, $turn:ident, $patch:expr) => {
        pub(super) fn $name(fresh: FreshTurns, runs: Query<&Cursor>, mut commands: Commands) {
            for (entity, parent, previous) in &fresh {
                let $turn = runs
                    .get(parent.parent())
                    .expect("fresh turn belongs to a run")
                    .turn;
                let patch: Option<RequestPatch> = $patch;
                if let Some(patch) = patch {
                    commands
                        .entity(entity)
                        .insert(previous.cloned().unwrap_or_default().merge(patch));
                }
            }
        }
    };
}
patch_system!(
    required_first,
    turn,
    (turn == 1).then(|| RequestPatch {
        tool_choice: Some(ToolChoice::Required),
        ..Default::default()
    })
);
patch_system!(
    none_second,
    turn,
    (turn == 2).then(|| RequestPatch {
        tool_choice: Some(ToolChoice::None),
        ..Default::default()
    })
);
patch_system!(
    extra_context,
    _turn,
    Some(RequestPatch {
        extra_context: vec![Document {
            id: SHAPING_CONTEXT_ID.into(),
            text: SHAPING_CONTEXT.into(),
            additional_props: Default::default()
        }],
        ..Default::default()
    })
);
patch_system!(
    preamble_always,
    _turn,
    Some(RequestPatch {
        preamble: Some(PIRATE_PREAMBLE.into()),
        ..Default::default()
    })
);
patch_system!(
    preamble_second,
    turn,
    (turn == 2).then(|| RequestPatch {
        preamble: Some(PIRATE_PREAMBLE.into()),
        ..Default::default()
    })
);
patch_system!(
    max_tokens_second,
    turn,
    (turn == 2).then(|| RequestPatch {
        max_tokens: Some(5),
        ..Default::default()
    })
);
patch_system!(
    thinking_second,
    turn,
    (turn == 2).then(|| RequestPatch {
        temperature: Some(1.0),
        additional_params: Some(
            serde_json::json!({"thinking":{"type":"enabled","budget_tokens":1024}})
        ),
        ..Default::default()
    })
);
patch_system!(
    active_tools_second,
    turn,
    (turn == 2).then(|| RequestPatch {
        active_tools: Some(vec![]),
        ..Default::default()
    })
);
patch_system!(
    history_first,
    turn,
    (turn == 1).then(|| RequestPatch {
        history: Some(
            [
                Message::user("My name is Ada."),
                Message::assistant("Hello, Ada.")
            ]
            .iter()
            .map(|message| MessageParts::from_message(message).expect("text exchange"))
            .collect()
        ),
        ..Default::default()
    })
);

#[derive(Resource)]
pub(super) struct SelectedRoute(pub Entity);

pub(super) fn route_first(
    fresh: Query<&ChildOf, Added<Fresh>>,
    runs: Query<(&RunOf, &Cursor)>,
    agents: Query<&UsesModel>,
    selected: Res<SelectedRoute>,
    mut commands: Commands,
) {
    for parent in &fresh {
        let (run_of, cursor) = runs.get(parent.parent()).expect("fresh turn's run");
        let model = if cursor.turn == 1 {
            selected.0
        } else {
            agents.get(run_of.0).expect("default model").0
        };
        commands.entity(parent.parent()).insert(UsesModel(model));
    }
}
pub(super) fn route_always(
    fresh: Query<&ChildOf, Added<Fresh>>,
    selected: Res<SelectedRoute>,
    mut commands: Commands,
) {
    for parent in &fresh {
        commands
            .entity(parent.parent())
            .insert(UsesModel(selected.0));
    }
}
