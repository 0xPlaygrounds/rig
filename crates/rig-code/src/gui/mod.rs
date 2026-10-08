//! A window beside the terminal, with feature `gui`: the agent graph, a
//! timeline of every turn, model call and tool call per agent, the
//! selected call's arguments, result and diff, cost by agent and by call,
//! the calls waiting for approval with buttons to answer them, and a prompt
//! line.
//!
//! It is a view like the terminal one and the JSON stream: it reads the
//! agents' components and sends them the same requests ([`Submit`],
//! [`FollowUp`], [`Interrupt`], [`Approve`], [`Focus`], [`Retry`],
//! [`Compact`]). It never drives the turn loop. The two views run in one
//! app: winit's loop replaces the headless runner, and the [`Wake`] that
//! finished calls, streamed fragments and terminal input already call
//! sends winit a `WakeUp`, so the window sleeps between events like the
//! terminal did (winit's reactive mode).
//!
//! `plugins.toml` turns it on with an entry for `rig_code::gui::GuiPlugin`;
//! the launcher then builds rig-code with this feature. Without a display
//! (`DISPLAY` or `WAYLAND_DISPLAY` on Linux) the plugin adds nothing and the
//! terminal view carries on alone.

mod panels;
mod style;
mod timeline;

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::{Duration, Instant};

use bevy::input_focus::{AutoFocus, InputFocus};
use bevy::picking::events::{PointerClick, PointerScroll};
use bevy::picking::pointer::PointerButton;
use bevy::prelude::*;
use bevy::text::EditableText;
use bevy::ui::ComputedNode;
use bevy::ui_widgets::TextInput;
use bevy::window::{ExitCondition, RequestRedraw, WindowClosed};
use bevy::winit::{EventLoopProxyWrapper, UpdateMode, WinitSettings, WinitUserEvent};
use rig::code_protocol::Mode;

use crate::core::agent::{
    ActiveTurn, Agent, Compact, Focus, Interrupt, Notice, NoticeLevel, Retry, Submit,
};
use crate::core::approval::{ApprovalAnswer, Approve};
use crate::core::calls::Wake;
use crate::core::inbox::FollowUp;
use crate::core::turn::PollCalls;
use crate::core::usage::Spending;
use crate::host::headless::{PrimaryQuery, RunMode, primary};

use style::{BACKGROUND, BODY, BORDER, DIM, PANEL, TEXT};
use timeline::Timeline;

/// Frames run after a wake, as the headless runner does: a woken frame
/// often leaves work for the next one.
const SETTLE_FRAMES: u32 = 4;
/// How often the bars of running calls grow.
const GROW_EVERY: Duration = Duration::from_millis(250);
/// The shortest time between two rebuilds of a panel.
const REBUILD_EVERY: Duration = Duration::from_millis(100);

/// Opens the window. List it after the terminal view in `plugins.toml`.
#[derive(Default)]
pub struct GuiPlugin;

impl Plugin for GuiPlugin {
    fn build(&self, app: &mut App) {
        // A print run answers on stdout and exits; no window for that.
        if app
            .world()
            .get_resource::<RunMode>()
            .is_some_and(|mode| matches!(mode.mode(), Mode::Print { .. }))
        {
            return;
        }
        if !display_available() {
            bevy_log::warn!("no display (DISPLAY or WAYLAND_DISPLAY), so no window");
            return;
        }
        app.add_plugins(window_plugins());
        if let Some(proxy) = app
            .world()
            .get_resource::<EventLoopProxyWrapper>()
            .map(|proxy| (**proxy).clone())
        {
            let settle = Settle::default();
            let frames = Arc::clone(&settle.0);
            app.insert_resource(Wake::new(move || {
                frames.store(SETTLE_FRAMES, Ordering::Relaxed);
                proxy.send_event(WinitUserEvent::WakeUp).ok();
            }))
            .insert_resource(settle);
        }
        app.insert_resource(WinitSettings {
            focused_mode: UpdateMode::reactive(GROW_EVERY),
            unfocused_mode: UpdateMode::reactive_low_power(Duration::from_secs(1)),
        })
        .init_resource::<GuiView>()
        .init_resource::<Timeline>()
        .init_resource::<Dirty>()
        .init_resource::<panels::NoticeLog>()
        .add_systems(Startup, spawn_layout)
        .add_systems(
            Update,
            (
                follow_agents,
                collect_notices,
                keys,
                mark_dirty,
                panels::rebuild.run_if(any_with_component::<Window>),
            )
                .chain()
                .after(PollCalls),
        )
        .add_systems(Last, (settle_frames, exit_on_close))
        .add_observer(on_click)
        .add_observer(on_scroll)
        .add_observer(on_focus)
        .add_observer(timeline::on_turn_start)
        .add_observer(timeline::on_turn_end)
        .add_observer(timeline::on_call_start)
        .add_observer(timeline::on_call_end)
        .add_observer(timeline::on_model_call)
        .add_observer(timeline::on_model_done)
        .add_observer(timeline::on_tool_call)
        .add_observer(timeline::on_tool_running)
        .add_observer(timeline::on_tool_done)
        .add_observer(timeline::on_approval_asked)
        .add_observer(timeline::on_approval_answered)
        .add_observer(timeline::on_summary_start)
        .add_observer(timeline::on_summary_done)
        .add_observer(timeline::on_backoff);
    }

    fn finish(&self, app: &mut App) {
        if !app.world().contains_resource::<GuiView>() {
            return;
        }
        // Closing the window ends an interactive app that has no terminal
        // view; next to the terminal, or in an RPC or eval run, the app
        // carries on without it.
        let mode = app.world().get_resource::<RunMode>().cloned();
        let interactive = mode
            .as_ref()
            .is_none_or(|mode| matches!(mode.mode(), Mode::Interactive));
        #[cfg(feature = "tui")]
        let terminal = interactive && app.is_plugin_added::<crate::tui::TuiPlugin>();
        #[cfg(not(feature = "tui"))]
        let terminal = false;
        app.insert_resource(ExitOnClose(interactive && !terminal));
    }
}

/// Bevy's windowing, rendering and UI plugins, without what
/// [`HeadlessPlugins`](crate::HeadlessPlugins) already added: the log, the
/// task pools and the signal handler. Winit's loop replaces the headless
/// runner.
fn window_plugins() -> bevy::app::PluginGroupBuilder {
    let group = DefaultPlugins.build().set(WindowPlugin {
        primary_window: Some(Window {
            title: "rig".to_owned(),
            resolution: (1440, 900).into(),
            ..default()
        }),
        exit_condition: ExitCondition::DontExit,
        close_when_requested: true,
        ..default()
    });
    let group = without::<bevy_log::LogPlugin>(group);
    let group = without::<bevy::app::TaskPoolPlugin>(group);
    #[cfg(any(unix, windows))]
    let group = without::<bevy::app::TerminalCtrlCHandlerPlugin>(group);
    group
}

/// `group` with `T` disabled, when it has `T`: which plugins
/// `DefaultPlugins` holds depends on the Bevy features the app turned on.
fn without<T: Plugin>(group: bevy::app::PluginGroupBuilder) -> bevy::app::PluginGroupBuilder {
    if group.contains::<T>() {
        group.disable::<T>()
    } else {
        group
    }
}

/// Whether a window can open: on Linux and the BSDs, an X11 or Wayland
/// display is named; elsewhere there always is one.
fn display_available() -> bool {
    if cfg!(any(
        target_os = "linux",
        target_os = "freebsd",
        target_os = "openbsd",
        target_os = "netbsd",
        target_os = "dragonfly"
    )) {
        ["DISPLAY", "WAYLAND_DISPLAY"]
            .iter()
            .any(|name| std::env::var_os(name).is_some_and(|value| !value.is_empty()))
    } else {
        true
    }
}

/// How many more frames a wake asks for.
#[derive(Resource, Default)]
struct Settle(Arc<AtomicU32>);

/// Asks winit for another frame while a wake's settle frames last.
fn settle_frames(settle: Option<Res<Settle>>, mut redraw: MessageWriter<RequestRedraw>) {
    let Some(settle) = settle else {
        return;
    };
    let left = settle.0.load(Ordering::Relaxed);
    if left > 0 {
        settle.0.store(left - 1, Ordering::Relaxed);
        redraw.write(RequestRedraw);
    }
}

/// Whether closing the window ends the app.
#[derive(Resource)]
struct ExitOnClose(bool);

fn exit_on_close(
    mut closed: MessageReader<WindowClosed>,
    exit_on_close: Option<Res<ExitOnClose>>,
    mut exit: MessageWriter<AppExit>,
) {
    if closed.read().count() > 0 && exit_on_close.is_some_and(|exit| exit.0) {
        exit.write(AppExit::Success);
    }
}

/// The window's own state, apart from any other view's: the agent it
/// shows and sends to, the selected span and the timeline's zoom.
#[derive(Resource, Default)]
pub(crate) struct GuiView {
    agent: Option<Entity>,
    span: Option<u64>,
    zoom: Zoom,
}

/// How much of the past the timeline shows.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum Zoom {
    Minute,
    FiveMinutes,
    HalfHour,
    #[default]
    All,
}

impl Zoom {
    const ALL: [Self; 4] = [Self::Minute, Self::FiveMinutes, Self::HalfHour, Self::All];

    fn label(self) -> &'static str {
        match self {
            Self::Minute => "1m",
            Self::FiveMinutes => "5m",
            Self::HalfHour => "30m",
            Self::All => "all",
        }
    }

    fn window(self) -> Option<Duration> {
        match self {
            Self::Minute => Some(Duration::from_secs(60)),
            Self::FiveMinutes => Some(Duration::from_secs(300)),
            Self::HalfHour => Some(Duration::from_secs(1800)),
            Self::All => None,
        }
    }
}

/// Which panels need building again.
#[derive(Resource)]
struct Dirty {
    graph: bool,
    timeline: bool,
    details: bool,
    cost: bool,
    notices: bool,
    /// When the panels were last built.
    built: Instant,
    /// When the running bars last grew.
    grown: Instant,
}

impl Default for Dirty {
    fn default() -> Self {
        Self {
            graph: true,
            timeline: true,
            details: true,
            cost: true,
            notices: true,
            built: Instant::now(),
            grown: Instant::now(),
        }
    }
}

impl Dirty {
    fn all(&mut self) {
        self.graph = true;
        self.timeline = true;
        self.details = true;
        self.cost = true;
        self.notices = true;
    }
}

/// The containers the panels are built into.
#[derive(Component, Clone, Copy, Debug, PartialEq, Eq)]
enum Region {
    Header,
    Graph,
    Cost,
    Timeline,
    Notices,
    Details,
    Status,
}

/// The prompt line.
#[derive(Component)]
struct PromptInput;

/// What a click on a node does.
#[derive(Component, Clone, Debug)]
enum Action {
    /// Show this agent here and in every other view.
    Agent(Entity),
    /// Show this span's details.
    Span(u64),
    /// Show the agent's overview instead of a span.
    Overview,
    Zoom(Zoom),
    /// Answer this call's approval.
    Approve(Entity, ApprovalAnswer),
    /// Send the prompt line: as a message, or queued after the turn.
    Send {
        queue: bool,
    },
    Stop,
    Retry,
    Compact,
}

/// The layout: a header; the agent graph and the cost on the left; the
/// timeline and the notices in the middle; the details on the right; the
/// prompt line at the bottom. The panels' contents are built into the
/// [`Region`]s by [`panels::rebuild`].
fn spawn_layout(mut commands: Commands) {
    commands.spawn(Camera2d);
    let column = |width: Val| Node {
        width,
        height: percent(100),
        flex_direction: FlexDirection::Column,
        padding: UiRect::all(px(8)),
        border: UiRect::right(px(1)),
        min_height: px(0),
        ..default()
    };
    let scroll = |grow: f32| Node {
        flex_direction: FlexDirection::Column,
        flex_grow: grow,
        min_height: px(0),
        overflow: Overflow {
            x: OverflowAxis::Clip,
            y: OverflowAxis::Scroll,
        },
        ..default()
    };
    commands.spawn((
        Node {
            width: percent(100),
            height: percent(100),
            flex_direction: FlexDirection::Column,
            ..default()
        },
        BackgroundColor(BACKGROUND),
        children![
            (
                Region::Header,
                Node {
                    height: px(30),
                    padding: UiRect::axes(px(10), px(6)),
                    column_gap: px(16),
                    align_items: AlignItems::Center,
                    border: UiRect::bottom(px(1)),
                    ..default()
                },
                BorderColor::all(BORDER),
                BackgroundColor(PANEL),
            ),
            (
                Node {
                    flex_grow: 1.0,
                    min_height: px(0),
                    ..default()
                },
                children![
                    (
                        column(px(320)),
                        BorderColor::all(BORDER),
                        children![
                            (style::heading("Agents")),
                            (Region::Graph, scroll(1.0), ScrollPosition::default()),
                            (
                                Region::Cost,
                                Node {
                                    flex_direction: FlexDirection::Column,
                                    padding: UiRect::top(px(8)),
                                    border: UiRect::top(px(1)),
                                    row_gap: px(4),
                                    ..default()
                                },
                                BorderColor::all(BORDER),
                            ),
                        ],
                    ),
                    (
                        Node {
                            flex_grow: 1.0,
                            flex_direction: FlexDirection::Column,
                            padding: UiRect::all(px(8)),
                            border: UiRect::right(px(1)),
                            min_height: px(0),
                            min_width: px(0),
                            ..default()
                        },
                        BorderColor::all(BORDER),
                        children![
                            (Region::Timeline, scroll(1.0), ScrollPosition::default()),
                            (
                                Region::Notices,
                                Node {
                                    flex_direction: FlexDirection::Column,
                                    height: px(110),
                                    padding: UiRect::top(px(6)),
                                    border: UiRect::top(px(1)),
                                    overflow: Overflow::clip(),
                                    ..default()
                                },
                                BorderColor::all(BORDER),
                            ),
                        ],
                    ),
                    (
                        column(px(520)),
                        BorderColor::all(BORDER),
                        children![(Region::Details, scroll(1.0), ScrollPosition::default())],
                    ),
                ],
            ),
            (
                Node {
                    height: px(44),
                    padding: UiRect::axes(px(10), px(6)),
                    column_gap: px(8),
                    align_items: AlignItems::Center,
                    border: UiRect::top(px(1)),
                    ..default()
                },
                BorderColor::all(BORDER),
                BackgroundColor(PANEL),
                children![
                    (
                        Region::Status,
                        Node {
                            width: px(300),
                            overflow: Overflow::clip(),
                            ..default()
                        },
                    ),
                    (
                        PromptInput,
                        TextInput,
                        EditableText {
                            allow_newlines: false,
                            ..default()
                        },
                        TextLayout::no_wrap(),
                        TextFont {
                            font_size: FontSize::Px(BODY + 1.0),
                            ..default()
                        },
                        TextColor(TEXT),
                        AutoFocus,
                        Node {
                            flex_grow: 1.0,
                            padding: UiRect::axes(px(8), px(4)),
                            border: UiRect::all(px(1)),
                            border_radius: BorderRadius::all(px(4)),
                            ..default()
                        },
                        BorderColor::all(BORDER),
                        BackgroundColor(BACKGROUND),
                    ),
                    button("Send", Action::Send { queue: false }),
                    button("Queue", Action::Send { queue: true }),
                    button("Stop", Action::Stop),
                ],
            ),
        ],
    ));
}

/// A button that does `action`.
fn button(name: &str, action: Action) -> impl Bundle {
    (
        action,
        style::button_node(false),
        children![style::label(name, BODY, TEXT)],
    )
}

/// Keeps the shown agent: the primary one until the user picks another,
/// and the primary one again when the shown agent goes away.
fn follow_agents(
    mut view: ResMut<GuiView>,
    agents: PrimaryQuery,
    live: Query<(), With<Agent>>,
    mut dirty: ResMut<Dirty>,
) {
    if view.agent.is_none_or(|agent| !live.contains(agent))
        && let Some(agent) = primary(&agents)
    {
        view.agent = Some(agent);
        view.span = None;
        dirty.all();
    }
}

/// Another view focused an agent: show it here too.
fn on_focus(
    focus: On<Focus>,
    agents: Query<(), With<Agent>>,
    mut view: ResMut<GuiView>,
    mut dirty: ResMut<Dirty>,
) {
    if agents.contains(focus.entity) && view.agent != Some(focus.entity) {
        view.agent = Some(focus.entity);
        view.span = None;
        dirty.all();
    }
}

fn collect_notices(
    mut notices: MessageReader<Notice>,
    mut log: ResMut<panels::NoticeLog>,
    mut dirty: ResMut<Dirty>,
) {
    for notice in notices.read() {
        log.push(
            notice.agent,
            notice.text.clone(),
            notice.level == NoticeLevel::Error,
        );
        dirty.notices = true;
    }
}

/// Enter sends the prompt line (steering a running turn), Ctrl+Enter
/// queues it after the turn, and Esc stops the shown agent's turn.
fn keys(
    keyboard: Res<ButtonInput<KeyCode>>,
    focus: Res<InputFocus>,
    mut prompt: Query<(Entity, &mut EditableText), With<PromptInput>>,
    view: Res<GuiView>,
    mut commands: Commands,
) {
    let Some(agent) = view.agent else {
        return;
    };
    if keyboard.just_pressed(KeyCode::Escape) {
        commands.trigger(Interrupt { entity: agent });
    }
    if !keyboard.just_pressed(KeyCode::Enter) {
        return;
    }
    let queue = keyboard.any_pressed([KeyCode::ControlLeft, KeyCode::ControlRight]);
    for (entity, mut input) in &mut prompt {
        if focus.get() == Some(entity) && !input.is_composing() {
            send(&mut input, agent, queue, &mut commands);
        }
    }
}

/// Sends the prompt line's text to `agent` and clears it.
fn send(input: &mut EditableText, agent: Entity, queue: bool, commands: &mut Commands) {
    let text = input.value().to_string();
    let text = text.trim();
    if text.is_empty() {
        return;
    }
    let text = text.to_owned();
    if queue {
        commands.trigger(FollowUp {
            entity: agent,
            text,
        });
    } else {
        commands.trigger(Submit {
            entity: agent,
            text,
        });
    }
    input.clear();
}

/// What changed since the last frame decides which panels are built again.
#[allow(clippy::type_complexity)]
fn mark_dirty(
    mut dirty: ResMut<Dirty>,
    mut timeline: ResMut<Timeline>,
    agents: Query<(), Or<(Added<Agent>, Changed<Spending>, Added<ActiveTurn>)>>,
    mut gone: RemovedComponents<Agent>,
    mut idle: RemovedComponents<ActiveTurn>,
) {
    let agents_changed = !agents.is_empty() || gone.read().count() > 0 || idle.read().count() > 0;
    if agents_changed {
        dirty.graph = true;
        dirty.cost = true;
        dirty.details = true;
    }
    if timeline.changed {
        timeline.changed = false;
        dirty.timeline = true;
        dirty.details = true;
        dirty.graph = true;
        dirty.cost = true;
    }
    if timeline.any_running() && dirty.grown.elapsed() >= GROW_EVERY {
        dirty.grown = Instant::now();
        dirty.timeline = true;
    }
}

/// Clicks on the nodes that carry an [`Action`].
fn on_click(
    mut click: On<PointerClick>,
    actions: Query<&Action>,
    mut view: ResMut<GuiView>,
    timeline: Res<Timeline>,
    mut prompt: Query<&mut EditableText, With<PromptInput>>,
    mut dirty: ResMut<Dirty>,
    mut commands: Commands,
) {
    if click.button != PointerButton::Primary {
        return;
    }
    let Ok(action) = actions.get(click.entity) else {
        return;
    };
    click.propagate(false);
    match action.clone() {
        Action::Agent(agent) => {
            view.agent = Some(agent);
            view.span = None;
            commands.trigger(Focus { entity: agent });
            dirty.all();
        }
        Action::Span(id) => {
            view.span = Some(id);
            if let Some(span) = timeline.get(id) {
                view.agent = Some(span.agent);
            }
            dirty.all();
        }
        Action::Overview => {
            view.span = None;
            dirty.all();
        }
        Action::Zoom(zoom) => {
            view.zoom = zoom;
            dirty.timeline = true;
        }
        Action::Approve(call, answer) => {
            commands.trigger(Approve {
                entity: call,
                answer,
            });
            dirty.details = true;
        }
        Action::Send { queue } => {
            if let Some(agent) = view.agent {
                for mut input in &mut prompt {
                    send(&mut input, agent, queue, &mut commands);
                }
            }
        }
        Action::Stop => {
            if let Some(agent) = view.agent {
                commands.trigger(Interrupt { entity: agent });
            }
        }
        Action::Retry => {
            if let Some(agent) = view.agent {
                commands.trigger(Retry { entity: agent });
            }
        }
        Action::Compact => {
            if let Some(agent) = view.agent {
                commands.trigger(Compact {
                    entity: agent,
                    focus: String::new(),
                });
            }
        }
    }
}

/// The mouse wheel scrolls the scrollable panel under the pointer, as in
/// `references/bevy/examples/ui/scroll_and_overflow/scroll.rs`.
fn on_scroll(
    mut scroll: On<PointerScroll>,
    mut panels: Query<(&mut ScrollPosition, &Node, &ComputedNode)>,
) {
    let Ok((mut position, node, computed)) = panels.get_mut(scroll.entity) else {
        return;
    };
    if node.overflow.y != OverflowAxis::Scroll {
        return;
    }
    let line = 20.0;
    let delta = match scroll.unit {
        bevy::input::mouse::MouseScrollUnit::Line => -scroll.y * line,
        bevy::input::mouse::MouseScrollUnit::Pixel => -scroll.y,
    };
    let most = ((computed.content_size().y - computed.size().y) * computed.inverse_scale_factor())
        .max(0.0);
    let next = (position.y + delta).clamp(0.0, most);
    if next != position.y {
        position.y = next;
        scroll.propagate(false);
    }
}

/// Text for the status line under the panels: who the prompt goes to.
fn status_text(title: &str, busy: bool, waiting: usize) -> (String, Color) {
    match (busy, waiting) {
        (_, 1..) => (
            format!("{title}: {waiting} waiting for approval"),
            style::YELLOW,
        ),
        (true, _) => (format!("{title}: working (Enter steers)"), style::BLUE),
        (false, _) => (format!("{title}: idle"), DIM),
    }
}
