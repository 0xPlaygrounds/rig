use std::time::{Duration, SystemTime};

use bevy_app::prelude::*;

use super::{ClockPlugin, Due, WakeAt};

#[test]
fn what_passed_is_due_until_it_is_armed_again() {
    let mut app = App::new();
    app.add_plugins(ClockPlugin);
    let past = app.world_mut().spawn(WakeAt(SystemTime::UNIX_EPOCH)).id();
    let later = app
        .world_mut()
        .spawn(WakeAt::after(Duration::from_secs(3600)))
        .id();
    app.update();
    assert!(app.world().entity(past).contains::<Due>());
    assert!(!app.world().entity(later).contains::<Due>());
    app.world_mut()
        .entity_mut(past)
        .insert(WakeAt::after(Duration::from_secs(3600)));
    app.update();
    assert!(!app.world().entity(past).contains::<Due>());
}
