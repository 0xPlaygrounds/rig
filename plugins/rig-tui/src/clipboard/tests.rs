use std::io::{self, ErrorKind};
use std::path::PathBuf;

use bevy_ecs::prelude::*;
use rig_ecs::agent::Notice;
use rig_ecs::requests::RequestOf;
use rig_harness::main_agent::ShownAgent;

use super::{CopyMissed, Paste, Pasted, read_either, type_pasted, warn_copy_missed};
use crate::editor::{Editor, MainEditor};
use crate::view::TuiView;

/// A pasted image is typed in the editor that asked for it, not in
/// whichever is the main one.
#[test]
fn a_pasted_image_is_typed_in_the_editor_that_asked_for_it() {
    let mut world = World::new();
    world.init_resource::<ShownAgent>();
    world.init_resource::<TuiView>();
    world.init_resource::<Messages<Notice>>();
    world.add_observer(type_pasted);
    let main = world.spawn((Editor::default(), MainEditor)).id();
    let panel = world.spawn(Editor::default()).id();
    let pasted = Pasted(Ok(Paste::Image(PathBuf::from("shot.png"))));
    world.spawn((RequestOf(panel), pasted));
    world.flush();
    let text = |editor| world.get::<Editor>(editor).map(|it| it.text().to_owned());
    assert_eq!(text(main).as_deref(), Some(""));
    assert!(text(panel).is_some_and(|text| text.contains("shot.png")));
}

/// Text a Ctrl+V found is typed as a paste is.
#[test]
fn pasted_text_is_typed() {
    let mut world = World::new();
    world.init_resource::<ShownAgent>();
    world.init_resource::<TuiView>();
    world.init_resource::<Messages<Notice>>();
    world.add_observer(type_pasted);
    let main = world.spawn((Editor::default(), MainEditor)).id();
    let pasted = Pasted(Ok(Paste::Text("two\nlines".to_owned())));
    world.spawn((RequestOf(main), pasted));
    world.flush();
    let text = world.get::<Editor>(main).map(|it| it.text().to_owned());
    assert_eq!(text.as_deref(), Some("two\nlines"));
    // Like a bracketed paste, it marks the input pasted for the completion.
    assert!(world.resource::<TuiView>().pasted);
}

/// A copy that found no clipboard program says so once a session.
#[test]
fn a_missed_copy_is_told_once() {
    let mut world = World::new();
    world.init_resource::<ShownAgent>();
    world.init_resource::<TuiView>();
    world.init_resource::<Messages<Notice>>();
    world.add_observer(warn_copy_missed);
    for _ in 0..2 {
        let missing = io::Error::new(ErrorKind::NotFound, "wl-copy is not installed");
        world.spawn(CopyMissed(Some(missing)));
        world.flush();
    }
    let notices = world.resource::<Messages<Notice>>();
    let told: Vec<&Notice> = notices.iter_current_update_messages().collect();
    assert_eq!(told.len(), 1, "{told:?}");
}

/// An image reader that is not installed, as `pngpaste` often is on macOS,
/// does not stop the text read; it is named when there is no text. A read
/// that timed out ends the paste, as the next would wait again.
#[test]
fn a_missing_image_reader_still_pastes_text() {
    let missing = || {
        Err(io::Error::new(
            ErrorKind::NotFound,
            "pngpaste is not installed",
        ))
    };
    let found = read_either(missing, || Ok(Some("copied".to_owned())));
    assert!(matches!(found, Ok(Paste::Text(text)) if text == "copied"));
    let found = read_either(missing, || Ok(None));
    assert!(
        matches!(&found, Err(why) if why.contains("pngpaste")),
        "{:?}",
        found.err()
    );

    let mut read_text = false;
    let hung = || {
        Err(io::Error::new(
            ErrorKind::TimedOut,
            "wl-paste did not answer",
        ))
    };
    let found = read_either(hung, || {
        read_text = true;
        Ok(Some("copied".to_owned()))
    });
    assert!(
        matches!(&found, Err(why) if why.contains("wl-paste")),
        "{:?}",
        found.err()
    );
    assert!(!read_text);
}
