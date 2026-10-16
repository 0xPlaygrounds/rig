//! `#` completion in the terminal view's input: a [`CompletionSource`] for
//! the `#` marker whose candidates are the open issues, numbers listed
//! with their titles, loaded in the background the first time the user
//! types a `#` word ([`InputContext::typed`]).

use bevy_tasks::{IoTaskPool, TaskPool};
use rig_harness::prelude::*;
use rig_tui::{CompletionKind, CompletionSource, InputContext};

use crate::issue_references::{Locate, State, load_open_issues};

/// The tracker's open issues, number and title, loaded the first time the
/// user types `#`, once per session.
#[derive(Resource, Default)]
pub struct OpenIssues {
    /// The issues, once loaded; empty when they could not be.
    pub issues: Option<Vec<(u64, String)>>,
    loading: bool,
}

/// The entity holding the `#` source.
#[derive(Component)]
struct IssueSource;

/// On the entity loading the open issues: what they loaded as.
#[derive(Component)]
struct OpenLoaded(Result<Vec<(u64, String)>, String>);

impl From<Result<Vec<(u64, String)>, String>> for OpenLoaded {
    fn from(loaded: Result<Vec<(u64, String)>, String>) -> Self {
        Self(loaded)
    }
}

pub(crate) fn add(app: &mut App) {
    app.init_resource::<OpenIssues>()
        .add_systems(Startup, offer_marker)
        .add_systems(Update, load_on_hash)
        .add_observer(open_loaded);
}

/// The `#` marker, with no candidates until the issues load.
fn offer_marker(mut commands: Commands) {
    commands.spawn((IssueSource, CompletionSource::marker('#')));
}

/// Starts loading the open issues the first time a `#` word is typed.
fn load_on_hash(
    input: Option<Res<InputContext>>,
    mut open: ResMut<OpenIssues>,
    state: Res<State>,
    locate: Locate,
    wake: Res<Wake>,
    mut commands: Commands,
) {
    let typed = input.as_deref().and_then(InputContext::typed);
    if open.loading || !matches!(typed, Some((CompletionKind::Marker('#'), _))) {
        return;
    }
    open.loading = true;
    let work = load_open_issues(state.0.clone(), locate.place());
    let pool = IoTaskPool::get_or_init(TaskPool::default);
    let running = Running::spawn_into::<OpenLoaded, _>(pool, &wake, work);
    commands.spawn((Name::new("loading the open issues"), running));
}

/// Keeps the open issues once loaded and offers them for `#`.
fn open_loaded(
    loaded: On<Insert<OpenLoaded>>,
    results: Query<&OpenLoaded>,
    sources: Query<Entity, With<IssueSource>>,
    mut open: ResMut<OpenIssues>,
    mut commands: Commands,
) {
    let Ok(OpenLoaded(result)) = results.get(loaded.entity) else {
        return;
    };
    let issues = result.clone().unwrap_or_default();
    let source = issues
        .iter()
        .fold(CompletionSource::marker('#'), |source, (number, title)| {
            source.candidate(number.to_string(), title.as_str())
        });
    for entity in &sources {
        commands.entity(entity).insert(source.clone());
    }
    open.issues = Some(issues);
    commands.entity(loaded.entity).despawn();
}
