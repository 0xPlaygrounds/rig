use rig_ecs::AgentPlugin;
use rig_harness::harness_protocol::Invocation;

use super::*;

/// The error notices written so far.
#[derive(Resource, Default)]
struct Errors(Vec<String>);

#[test]
fn a_restored_model_that_cannot_be_used_fails_the_run_once() {
    let mut app = App::new();
    app.add_plugins((TaskPoolPlugin::default(), AgentPlugin, PrintPlugin))
        .insert_resource(RunMode(Invocation {
            print: Some("hi".to_owned()),
            model: None,
        }))
        .init_resource::<Errors>()
        .add_systems(
            Last,
            |mut notices: MessageReader<Notice>, mut errors: ResMut<Errors>| {
                let failed = notices
                    .read()
                    .filter(|notice| notice.level == NoticeLevel::Error);
                errors.0.extend(failed.map(|notice| notice.text.clone()));
            },
        );
    app.world_mut()
        .spawn((Agent, ModelChoice("nobody/nothing".to_owned())));
    app.finish();
    app.cleanup();
    for _ in 0..10 {
        app.update();
    }
    assert_eq!(app.should_exit(), Some(AppExit::from_code(1)));
    let errors = &app.world().resource::<Errors>().0;
    assert_eq!(errors.len(), 1, "{errors:?}");
}
