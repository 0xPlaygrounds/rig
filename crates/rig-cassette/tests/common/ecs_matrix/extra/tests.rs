use super::*;

#[test]
fn after_terminal_cut_waits_for_completed_collection() {
    let mut app = App::new();
    app.insert_resource(CancelAt {
        cut: Cut::AfterTerminal,
        done: false,
    });
    app.add_systems(bevy_app::Update, cancel_at_cut);
    let run = app.world_mut().spawn_empty().id();
    let turn = app.world_mut().spawn((Turn, ChildOf(run))).id();
    let outcome = Err(rig_core::error::ErrorReport::new(
        ErrorKind::ProviderResponse,
        "terminal",
    ));
    let effect = app
        .world_mut()
        .spawn((
            ChildOf(turn),
            Streamed {
                outcome: Some(outcome.clone()),
                ..Streamed::default()
            },
        ))
        .id();
    app.update();
    assert!(!app.world().resource::<CancelAt>().done);
    assert!(app.world().get::<Cancelled>(run).is_none());
    app.world_mut()
        .entity_mut(effect)
        .insert(EffectOutcome(outcome));
    app.update();
    assert!(app.world().resource::<CancelAt>().done);
    assert!(app.world().get::<Cancelled>(run).is_some());
}
