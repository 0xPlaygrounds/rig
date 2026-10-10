use rig_harness::harness_protocol::Invocation;
use rig_harness::prelude::*;

use super::*;

/// The frames a panel drew.
#[derive(Resource, Default)]
struct Drawn(usize);

#[test]
fn a_panel_plugin_is_inert_in_a_print_run() {
    let mut app = App::new();
    // A system that cannot run fails the test.
    app.set_error_handler(rig_harness::error::panic)
        .insert_resource(RunMode(Invocation {
            print: Some(String::new()),
            model: None,
        }))
        .add_plugins(TuiPlugin)
        .init_resource::<Drawn>()
        .add_systems(Update, |mut redraw: MessageWriter<RequestRedraw>| {
            redraw.write(RequestRedraw);
        })
        .add_systems(
            PostUpdate,
            (|mut drawn: ResMut<Drawn>| drawn.0 += 1).in_set(TuiSystems::Draw),
        );
    app.update();
    app.update();
    let world = app.world();
    assert_eq!(world.resource::<Drawn>().0, 0);
    assert!(!world.contains_resource::<Front>());
}
