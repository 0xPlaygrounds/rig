use rig_harness::prelude::*;

use super::super::view::TuiView;
use crate::ArgCompletion;
use crate::tests::{press, type_text, typed, view_app};

/// Whether a completion list is open.
fn open(app: &App) -> bool {
    app.world().resource::<TuiView>().completion.is_some()
}

#[test]
fn a_completed_argument_does_not_open_the_list_for_the_next_word() {
    let (mut app, _) = view_app();
    app.add_command("model", "Choose the model", |_: In<CommandArgs>| {});
    app.add_command("pair", "Two arguments", |_: In<CommandArgs>| {});
    app.world_mut()
        .spawn(ArgCompletion::new("model").candidate("acme/fast", ""));
    app.world_mut()
        .spawn(ArgCompletion::new("pair").at(1).candidate("second", ""));
    type_text(&mut app, "/model acme");
    assert!(open(&app));
    press(&mut app, "tab");
    assert_eq!(typed(&app), "/model acme/fast ");
    assert!(!open(&app));
    type_text(&mut app, " ");
    assert!(!open(&app));
    // A completion of another position opens there only.
    press(&mut app, "ctrl+c");
    type_text(&mut app, "/pair ");
    assert!(!open(&app));
    type_text(&mut app, "first ");
    assert!(open(&app));
}
