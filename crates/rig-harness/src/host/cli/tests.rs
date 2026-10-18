use bevy_app::prelude::*;
use rig_harness_protocol::Invocation;

use super::{AppCliExt, Cli, Flag, Front, refusal};
use crate::host::launcher::Invoked;

/// An app started with `args`, on a terminal or not, whose plugins offer
/// the terminal view, `--print` and a JSON front, and take `--serve=ADDR`.
fn app(args: &[&str], terminal: bool) -> App {
    let mut app = App::new();
    app.insert_resource(Invoked {
        args: Invocation::parse(args).unwrap_or_default(),
        terminal,
    })
    .add_front(Front::new("tui").on_terminal())
    .add_front(Front::new("print").headless())
    .add_front(Front::new("json").on_flag("json", "Stream JSON lines"))
    .add_flag(Flag::new("serve", "Serve HTTP").value("ADDR"));
    app
}

#[test]
fn the_flag_given_else_the_kind_of_run_picks_the_front() {
    for (args, terminal, front) in [
        (&[][..], true, "tui"),
        (&[], false, "print"),
        (&["--json"], true, "json"),
        (&["--json", "-p", "hi"], false, "json"),
    ] {
        let mut app = app(args, terminal);
        assert!(app.runs_front(front), "{args:?} on a terminal: {terminal}");
        assert_eq!(refusal(&mut app), None);
        assert_eq!(app.world().resource::<Cli>().front(), Some(front));
    }
}

#[test]
fn options_no_plugin_takes_are_refused_and_the_rest_listed() {
    let mut app = app(&["--serve=127.0.0.1:80", "--jsno"], false);
    let refused = refusal(&mut app).unwrap_or_default();
    assert!(refused.contains("unknown option `--jsno`"), "{refused}");
    assert!(refused.contains("--serve ADDR"), "{refused}");
    assert!(refused.contains("--json"), "{refused}");
    for args in [&["--serve"][..], &["--json=yes"]] {
        assert!(refusal(&mut self::app(args, false)).is_some(), "{args:?}");
    }
    let invoked = app.world().resource::<Invoked>().clone();
    assert_eq!(invoked.flag_value("serve"), Some("127.0.0.1:80"));
    assert!(invoked.flag("jsno") && !invoked.flag("json"));
}

#[test]
fn a_terminal_no_front_takes_and_two_fronts_given_are_refused() {
    let mut bare = App::new();
    bare.insert_resource(Invoked {
        terminal: true,
        ..Invoked::default()
    });
    assert!(refusal(&mut bare).is_some_and(|why| why.contains("no front")));
    let mut two = app(&["--json", "--web"], true);
    two.add_front(Front::new("web").on_flag("web", "Serve a web page"));
    assert!(!two.runs_front("json") && !two.runs_front("web"));
    assert!(refusal(&mut two).is_some_and(|why| why.contains("each choose the front")));
    // Without a terminal and without a front, the run goes on headless.
    let mut headless = App::new();
    assert_eq!(refusal(&mut headless), None);
}

#[test]
fn a_front_with_a_command_or_option_takes_only_the_runs_that_name_it() {
    let mut app = App::new();
    app.insert_resource(Invoked::default())
        .add_front(
            Front::new("stats")
                .headless()
                .on_command("stats", "", "Count"),
        )
        .add_front(
            Front::new("web")
                .headless()
                .on_flag("web", "Serve a web page"),
        )
        .add_front(Front::new("print").headless());
    assert!(app.runs_front("print"));
    assert_eq!(refusal(&mut app), None);
}
