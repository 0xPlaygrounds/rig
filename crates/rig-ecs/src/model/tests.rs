use std::sync::atomic::{AtomicUsize, Ordering};

use rig_core::catalog::{Catalog, Transport};
use rig_core::operation::Completion;
use rig_core::providers::registry::ProviderId;
use rig_core::serve::ErasedHandler;
use rig_core::serve::adapters::ModelAdapter;
use rig_core::test_utils::MockCompletionModel;

use super::*;
use crate::spawn::spawn_root;

/// Models serving `test/one`, counting each connection made in `made`.
fn counted(made: &Arc<AtomicUsize>) -> Option<Models> {
    let id = ProviderId::declare("test").ok()?;
    let mut connector = Connector::new(Catalog::default());
    connector
        .catalog_mut()
        .insert(ModelSpec::new(id, "one").with_tools(true));
    let made = Arc::clone(made);
    connector.provide(id, move |spec: &ModelSpec, _: &Catalog, _: &Transport| {
        made.fetch_add(1, Ordering::SeqCst);
        let model = MockCompletionModel::text("hi");
        Ok(ErasedHandler::new(ModelAdapter::<Completion>::new(
            spec.reference(),
            model,
        )))
    });
    Some(Models(connector))
}

#[test]
fn choosing_a_model_for_an_unconnected_agent_connects_it_once() {
    let made = Arc::new(AtomicUsize::new(0));
    let mut world = World::new();
    world.init_resource::<Messages<Notice>>();
    if let Some(models) = counted(&made) {
        world.insert_resource(models);
    }
    world.add_observer(on_set_model);
    world.add_observer(connect);
    let agent = spawn_root(&mut world, ModelRefused("Cannot use test/gone.".into())).id();
    world.trigger(SetModel {
        entity: agent,
        model: "test/one".into(),
        kind: ChoiceKind::ByUser,
    });
    world.flush();
    assert_eq!(made.load(Ordering::SeqCst), 1);
    let chosen = world
        .get::<ModelChoice>(agent)
        .map(|choice| choice.0.clone());
    assert_eq!(chosen.as_deref(), Some("test/one"));
    let connected = world.get::<Connection>(agent).map(|c| c.spec.reference());
    assert_eq!(connected.as_deref(), Some("test/one"));
    assert!(world.get::<ModelRefused>(agent).is_none());
}
