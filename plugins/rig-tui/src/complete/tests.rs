use rig_harness::prelude::*;

use super::super::view::TuiView;
use crate::tests::{press, scratch, type_text, typed as typed_text, view_app};
use crate::{CompletionKind, CompletionSource, InputContext};

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
        .spawn(CompletionSource::arg("model").candidate("acme/fast", ""));
    app.world_mut()
        .spawn(CompletionSource::arg("pair").at(1).candidate("second", ""));
    type_text(&mut app, "/model acme");
    assert!(open(&app));
    press(&mut app, "tab");
    assert_eq!(typed_text(&app), "/model acme/fast ");
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

#[test]
fn a_marker_source_completes_a_word_anywhere_and_follows_its_lookup() {
    let (mut app, _) = view_app();
    let issues = app.world_mut().spawn(CompletionSource::marker('#')).id();
    type_text(&mut app, "fixes #14");
    // Nothing matches yet, but a lookup sees what is typed.
    assert!(!open(&app));
    let typed = app.world().resource::<InputContext>().typed();
    assert_eq!(typed, Some((CompletionKind::Marker('#'), "14")));
    // The lookup's answer opens the list without another key.
    let found = CompletionSource::marker('#')
        .candidate("142", "Crash on empty input")
        .candidate("7", "Unrelated");
    app.world_mut().entity_mut(issues).insert(found);
    app.update();
    assert!(open(&app));
    press(&mut app, "tab");
    assert_eq!(typed_text(&app), "fixes #142 ");
    // `/` and `@` stay the view's own.
    app.world_mut()
        .spawn(CompletionSource::marker('@').candidate("nobody", ""));
    press(&mut app, "ctrl+c");
    type_text(&mut app, "/mod");
    let typed = app.world().resource::<InputContext>().typed();
    assert_eq!(typed, Some((CompletionKind::Command, "mod")));
}

#[test]
fn at_completes_the_paths_of_the_project_root() {
    let root = scratch("complete-root");
    std::fs::create_dir_all(root.join("src")).ok();
    std::fs::write(root.join("src").join("only_in_root.rs"), "").ok();
    let (mut app, _) = view_app();
    app.insert_resource(ProjectRoot(root.clone()));
    type_text(&mut app, "@only_in");
    let started = std::time::Instant::now();
    let items = |app: &App| {
        let completion = app.world().resource::<TuiView>().completion.as_ref();
        let items = completion.into_iter().flat_map(|open| &open.items);
        items.map(|item| item.text.clone()).collect::<Vec<_>>()
    };
    while items(&app).is_empty() && started.elapsed() < std::time::Duration::from_secs(10) {
        app.update();
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    assert_eq!(items(&app), ["src/only_in_root.rs"]);
    std::fs::remove_dir_all(&root).ok();
}

/// `@` lists the paths of the shown agent's work directory, where its
/// references resolve, not the project root's.
#[test]
fn at_completes_the_paths_of_the_shown_agents_work_dir() {
    let root = scratch("complete-workdir-root");
    let dir = scratch("complete-workdir-dir");
    std::fs::write(root.join("only_in_root.rs"), "").ok();
    std::fs::write(dir.join("only_in_dir.rs"), "").ok();
    let (mut app, agent) = view_app();
    app.insert_resource(ProjectRoot(root.clone()));
    app.world_mut()
        .entity_mut(agent)
        .insert(WorkDir(dir.clone()));
    type_text(&mut app, "@only_in");
    let started = std::time::Instant::now();
    while !open(&app) && started.elapsed() < std::time::Duration::from_secs(10) {
        app.update();
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    let completion = app.world().resource::<TuiView>().completion.as_ref();
    let items: Vec<&str> = (completion.into_iter())
        .flat_map(|open| &open.items)
        .map(|item| item.text.as_str())
        .collect();
    assert_eq!(items, ["only_in_dir.rs"]);
    std::fs::remove_dir_all(&root).ok();
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn a_file_made_while_at_is_typed_is_listed_without_waiting() {
    use super::FileIndex;
    use std::time::{Duration, Instant};

    let root = scratch("complete-watch");
    std::fs::create_dir_all(root.join("src")).ok();
    std::fs::write(root.join("src").join("a.rs"), "").ok();
    let (mut app, _) = view_app();
    app.add_plugins(rig_harness::watch::WatchPlugin)
        .insert_resource(ProjectRoot(root.clone()));
    let items = |app: &App| {
        let completion = app.world().resource::<TuiView>().completion.as_ref();
        let items = completion.into_iter().flat_map(|open| &open.items);
        items.map(|item| item.text.clone()).collect::<Vec<_>>()
    };
    let run_until = |app: &mut App, done: &dyn Fn(&App) -> bool| {
        let started = Instant::now();
        while !done(app) && started.elapsed() < Duration::from_secs(10) {
            app.update();
            std::thread::sleep(Duration::from_millis(5));
        }
    };
    type_text(&mut app, "@b_new");
    let walked = |app: &App| {
        let index = app.world().resource::<FileIndex>();
        index.paths.iter().any(|path| path == "src/a.rs")
    };
    run_until(&mut app, &walked);
    // The watch's first look, and the walk it asks for, settle.
    let settle = Instant::now();
    run_until(&mut app, &|_| settle.elapsed() > Duration::from_millis(300));
    assert!(items(&app).is_empty());
    std::fs::write(root.join("src").join("b_new.rs"), "").ok();
    run_until(&mut app, &|app| !items(app).is_empty());
    assert_eq!(items(&app), ["src/b_new.rs"]);
    std::fs::remove_dir_all(&root).ok();
}

/// A picked path that a space would cut is put in quoted, so it attaches
/// that file, not one named by its first word; a directory stays open for
/// the path below it, and Enter closes an open quote before it sends.
#[test]
fn a_picked_path_with_a_space_is_quoted() {
    use crate::KeyInput;
    use crate::tests::push;
    use crossterm::event::Event;

    let root = scratch("complete-space");
    std::fs::create_dir_all(root.join("my dir")).ok();
    std::fs::write(root.join("attach notes.txt"), "").ok();
    std::fs::write(root.join("attach"), "").ok();
    std::fs::write(root.join("my dir").join("inner file.txt"), "").ok();
    std::fs::create_dir_all(root.join("~")).ok();
    std::fs::write(root.join("~").join("tilde note.txt"), "").ok();
    if cfg!(unix) {
        std::fs::write(root.join("back\\slash.txt"), "").ok();
        std::fs::write(root.join("loop:notes.txt"), "").ok();
    }
    let (mut app, _) = view_app();
    app.insert_resource(ProjectRoot(root.clone()));
    let walked = |app: &mut App| {
        let started = std::time::Instant::now();
        while !open(app) && started.elapsed() < std::time::Duration::from_secs(10) {
            app.update();
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
    };
    type_text(&mut app, "@att");
    walked(&mut app);
    press(&mut app, "down");
    press(&mut app, "tab");
    assert_eq!(typed_text(&app), "@\"attach notes.txt\" ");
    press(&mut app, "ctrl+c");
    type_text(&mut app, "@my");
    walked(&mut app);
    press(&mut app, "tab");
    assert_eq!(typed_text(&app), "@\"my dir/");
    type_text(&mut app, "inn");
    walked(&mut app);
    press(&mut app, "tab");
    assert_eq!(typed_text(&app), "@\"my dir/inner file.txt\" ");
    // A directory named `~` here is not the home directory.
    press(&mut app, "ctrl+c");
    type_text(&mut app, "@tilde");
    walked(&mut app);
    press(&mut app, "tab");
    assert_eq!(typed_text(&app), "@\"./~/tilde note.txt\" ");
    // On Unix a `\` is part of a name, not a directory's end.
    if cfg!(unix) {
        press(&mut app, "ctrl+c");
        type_text(&mut app, "@back");
        walked(&mut app);
        press(&mut app, "tab");
        assert_eq!(typed_text(&app), "@back\\slash.txt ");
        // A local name that reads as `HOST:PATH` stays local.
        press(&mut app, "ctrl+c");
        type_text(&mut app, "@loop:");
        walked(&mut app);
        press(&mut app, "tab");
        assert_eq!(typed_text(&app), "@./loop:notes.txt ");
    }
    press(&mut app, "ctrl+c");
    push(
        &mut app,
        KeyInput::Event(Event::Paste("@\"my dir/inner file.txt".to_owned())),
    );
    app.update();
    walked(&mut app);
    press(&mut app, "enter");
    assert_eq!(typed_text(&app), "@\"my dir/inner file.txt\" ");
    std::fs::remove_dir_all(&root).ok();
}

/// A dropped image whose path has a space becomes a quoted reference: a
/// quoted drop is taken as it is, an escaped one unescaped, even when a
/// file is named with the backslash too.
#[test]
fn a_dropped_image_with_a_space_is_typed_quoted() {
    use crate::KeyInput;
    use crate::tests::push;
    use crossterm::event::Event;

    let root = scratch("drop-space");
    let image = root.join("shot (1).png");
    std::fs::write(&image, b"\x89PNG\r\n\x1a\n").ok();
    let shown = image.display().to_string();
    let mut drops = vec![(format!("'{shown}'"), shown.clone())];
    if cfg!(unix) {
        let named = root.join("shot\\ (1).png").display().to_string();
        std::fs::write(&named, b"\x89PNG\r\n\x1a\n").ok();
        let root = root.display().to_string().replace(' ', "\\ ");
        let escaped = format!("{root}/shot\\ \\(1\\).png");
        drops.push((escaped, shown.clone()));
        drops.push((format!("'{named}'"), named));
    }
    let (mut app, _) = view_app();
    for (pasted, path) in drops {
        press(&mut app, "ctrl+c");
        type_text(&mut app, "see");
        push(&mut app, KeyInput::Event(Event::Paste(pasted)));
        app.update();
        assert_eq!(typed_text(&app), format!("see @\"{path}\" "));
    }
    std::fs::remove_dir_all(&root).ok();
}
