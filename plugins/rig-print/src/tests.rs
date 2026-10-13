use rig_harness::harness_protocol::Invocation;

use super::*;

/// A print run of `prompt` with `--answer=answers` when given, after its
/// first frame.
fn print_run(answers: Option<&str>) -> App {
    let mut app = App::new();
    let flags = answers.map(|ids| (ANSWER.to_owned(), Some(ids.to_owned())));
    app.insert_resource(Invoked {
        args: Invocation {
            print: Some("hi".to_owned()),
            flags: flags.into_iter().collect(),
            ..Invocation::default()
        },
        terminal: false,
    })
    .add_plugins((TaskPoolPlugin::default(), AgentPlugin, PrintPlugin));
    app.finish();
    app.cleanup();
    app.update();
    app
}

/// How each of `asks` was answered after a frame: `Some(None)` when
/// refused, `None` when not at all.
fn answered(app: &mut App, asks: impl IntoIterator<Item = Ask>) -> Vec<Option<Option<Answer>>> {
    let asks: Vec<Entity> = asks
        .into_iter()
        .map(|ask| app.world_mut().spawn(ask).id())
        .collect();
    app.update();
    let world = app.world();
    let outcome = |ask: &Entity| world.get::<Outcome>(*ask).map(Answer::of);
    asks.iter().map(outcome).collect()
}

#[test]
fn questions_are_answered_from_answer_or_refused() {
    let approval = || {
        Ask::new("Run rm -rf build?")
            .choice("deny", "Deny")
            .choice("allow", "Allow")
    };
    let mut app = print_run(Some("allow,yes"));
    let answerers = app
        .world_mut()
        .query::<&Answerer>()
        .iter(app.world())
        .count();
    assert_eq!(answerers, 1, "plugins may ask");
    let several = Ask::yes_no("Which?").choice("allow", "Allow").multiple();
    let answers = answered(
        &mut app,
        [approval(), Ask::new("Name?").free_text(), several],
    );
    let allowed = Some(Answer::choice("allow"));
    let both = Some(Answer::choices(["yes", "allow"]));
    assert_eq!(answers, [Some(allowed), Some(None), Some(both)]);

    let mut app = print_run(None);
    let answerers = app
        .world_mut()
        .query::<&Answerer>()
        .iter(app.world())
        .count();
    assert_eq!(answerers, 0, "plugins apply their own policy");
    assert_eq!(answered(&mut app, [approval()]), [Some(None)]);
}

#[test]
fn sections_follow_the_answer_by_tag() {
    let sections = [
        PrintSection::new("todo", "todo 1/2\n[x] read\n[ ] fix\n"),
        PrintSection::new("cost", "$0.02"),
        PrintSection::new("empty", ""),
    ];
    let printed = output(Some("Fixed.\n"), &sections);
    assert_eq!(
        printed.as_deref(),
        Some("Fixed.\n\n$0.02\n\ntodo 1/2\n[x] read\n[ ] fix")
    );
    assert_eq!(output(None, &[]), None);
}
