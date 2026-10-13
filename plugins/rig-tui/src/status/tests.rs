use bevy_app::prelude::*;
use bevy_ecs::prelude::*;

use super::*;

const CONTEXT: StatusItem = StatusItem::new("context", Side::Right, 40, 5);

/// Whether the gauge plugin hides the `context` item instead of
/// replacing it.
#[derive(Resource, Default)]
struct Hiding(bool);

/// The owner shows its `context` item every frame.
fn show(mut items: Query<&mut StatusItems, With<Agent>>) {
    for mut items in &mut items {
        items.show(CONTEXT.says("ctx 30k/1M (3%)", Tone::Dim));
    }
}

/// Another plugin replaces it, or hides it, by name.
fn gauge(mut items: Query<&mut StatusItems, With<Agent>>, hiding: Res<Hiding>) {
    for mut items in &mut items {
        match hiding.0 {
            true => items.hide("context"),
            false => {
                items.show(StatusItem::new("context", Side::Right, 40, 5).says("▮▯▯", Tone::Red))
            }
        }
    }
}

fn shown(app: &mut App, agent: Entity) -> Vec<(&'static str, String)> {
    let items = app.world().get::<StatusItems>(agent);
    let items = items.into_iter().flat_map(StatusItems::iter);
    items.map(|item| (item.name, item.text.clone())).collect()
}

#[test]
fn a_plugin_replaces_or_hides_another_plugins_item_by_name() {
    let mut app = App::new();
    add(&mut app);
    app.init_resource::<Hiding>()
        // Added first, the replacing plugin still runs after the owner.
        .add_systems(PostUpdate, gauge.in_set(StatusSystems::Edit))
        .add_systems(PostUpdate, show.in_set(StatusSystems::Show));
    let agent = app.world_mut().spawn(Agent).id();
    app.update();
    assert_eq!(shown(&mut app, agent), [("context", "▮▯▯".to_owned())]);
    app.world_mut().resource_mut::<Hiding>().0 = true;
    app.update();
    assert_eq!(shown(&mut app, agent), []);
}
