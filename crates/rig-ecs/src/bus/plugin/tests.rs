#![allow(clippy::unwrap_used)]
use super::*;
use crate::bus::{Handlers, Issued, PendingEffect};
use rig_core::effect::{EffectKind, FamilyDescriptor};

fn custom() -> EffectKind {
    EffectKind::Custom {
        kind: "test".into(),
        payload: serde_json::Value::Null,
    }
}

fn app_with_capacity(command_capacity: usize, pending: usize) -> App {
    let mut app = App::new();
    app.add_plugins(BusPlugin::with_policy(ServingPolicy {
        command_capacity,
        ..ServingPolicy::default()
    }));
    let world = app.world_mut();
    Handlers::with(world, |handlers| {
        handlers.register_open(
            "open",
            FamilyDescriptor::Custom {
                kind: "test".into(),
            },
        )
    })
    .unwrap()
    .unwrap();
    for _ in 0..pending {
        world.spawn(PendingEffect::new("open", custom()));
    }
    app
}

fn issued(app: &mut App) -> usize {
    let world = app.world_mut();
    world.query::<&Issued>().iter(world).count()
}

/// The intake bound measures issues per update: one update takes at most
/// `command_capacity` effects, and the rest wait for later updates rather
/// than for anything else.
#[test]
fn an_update_bounds_intake_and_the_rest_wait_for_later_updates() {
    let mut app = app_with_capacity(2, 5);
    app.update();
    assert_eq!(issued(&mut app), 2, "one update issues at most the bound");
    app.update();
    assert_eq!(issued(&mut app), 4);
    app.update();
    assert_eq!(issued(&mut app), 5);
}

/// A finished task raises the wake; a runner waiting on it returns at once,
/// and an untouched wake times out as not raised.
#[test]
fn the_wake_is_raised_by_signal_and_taken_by_wait() {
    let wake = Wake::default();
    assert!(!wake.wait(Duration::from_millis(1)));
    let raiser = wake.clone();
    std::thread::spawn(move || raiser.signal());
    assert!(wake.wait(Duration::from_secs(5)));
    assert!(!wake.take(), "wait took the signal");
}
