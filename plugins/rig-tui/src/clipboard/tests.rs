use std::path::PathBuf;

use bevy_ecs::prelude::*;
use rig_ecs::agent::Notice;
use rig_ecs::requests::RequestOf;
use rig_harness::main_agent::ShownAgent;

use super::{PastedImage, type_pasted};
use crate::editor::{Editor, MainEditor};

/// A pasted image is typed in the editor that asked for it, not in
/// whichever is the main one.
#[test]
fn a_pasted_image_is_typed_in_the_editor_that_asked_for_it() {
    let mut world = World::new();
    world.init_resource::<ShownAgent>();
    world.init_resource::<Messages<Notice>>();
    world.add_observer(type_pasted);
    let main = world.spawn((Editor::default(), MainEditor)).id();
    let panel = world.spawn(Editor::default()).id();
    let pasted = PastedImage(Ok(PathBuf::from("shot.png")));
    world.spawn((RequestOf(panel), pasted));
    world.flush();
    let text = |editor| world.get::<Editor>(editor).map(|it| it.text().to_owned());
    assert_eq!(text(main).as_deref(), Some(""));
    assert!(text(panel).is_some_and(|text| text.contains("shot.png")));
}
