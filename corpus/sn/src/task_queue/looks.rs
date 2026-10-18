//! The queue in the terminal view: the agent half publishes the panel's
//! rows ([`QueueRows`]) and keeps the status-line item with how many run and
//! wait, and how the last one ended; the view half
//! ([`TaskQueueView`](super::TaskQueueView)) shows the rows in a panel
//! under the transcript while there are any.

use std::time::Duration;

use bevy_ecs::schedule::SystemCondition;
use rig_harness::channel::{ChannelAppExt, ChannelEvent};
use rig_harness::prelude::*;
use rig_harness::status::{AppStatus, Side, StatusItem, StatusSystems, Tone};
use rig_harness::view::Applied;
use rig_harness::worker::{Publish, WorkerOutbox};
use rig_tui::ratatui::layout::Constraint;
use rig_tui::ratatui::widgets::{Block, Paragraph};
use rig_tui::{PanelCanvas, Placement, RequestRedraw, TuiPanel, TuiSystems};
use serde::{Deserialize, Serialize};

use super::{QueueNote, QueuedTask, RunningTurns, TaskQueue, TaskState, line, live_cost};

/// After the other agents at work; gone early when the line is narrow.
const QUEUE_ITEM: StatusItem = StatusItem::new("sn-task-queue", Side::Left, 75, 9);

/// The most task rows the panel takes.
const ROWS: usize = 8;

/// How often running tasks' rows are sent again, for their time and cost.
const EVERY: Duration = Duration::from_secs(1);

/// The panel's rows, newest last: each task's state, elapsed time, cost
/// and answer; none while the queue is empty.
#[derive(Serialize, Deserialize, Clone, Debug, Default, PartialEq, Eq)]
pub struct QueueRows {
    /// The rows.
    pub rows: Vec<String>,
}

impl ChannelEvent for QueueRows {
    const KIND: &'static str = "x/queue/rows";
}

/// On the queue's panel.
#[derive(Component)]
struct QueuePanel;

/// The agent half's status item and rows.
pub(super) fn add(app: &mut App) {
    app.init_resource::<AppStatus>()
        .add_channel_event::<QueueRows>()
        .add_systems(
            PostUpdate,
            status
                .in_set(StatusSystems::Show)
                .run_if(resource_changed::<TaskQueue>.or_eager(resource_changed::<QueueNote>)),
        )
        .add_systems(
            Last,
            publish
                .run_if(resource_exists::<WorkerOutbox>)
                .run_if(resource_changed::<TaskQueue>.or_eager(on_real_timer(EVERY))),
        );
}

/// The view half's panel.
pub(super) fn add_view(app: &mut App) {
    app.init_resource::<Shown>()
        .add_message::<RequestRedraw>()
        .add_message::<Applied>()
        .add_systems(Update, mirror_rows)
        .add_systems(
            PostUpdate,
            (
                place_panel.run_if(resource_changed::<Shown>),
                draw.in_set(TuiSystems::Draw),
            ),
        );
}

/// The rows as they are now.
pub fn rows(
    queue: &TaskQueue,
    agents: &Query<(Entity, &QueuedTask)>,
    turns: &RunningTurns,
) -> QueueRows {
    let live = |number: u32| {
        let agent = agents.iter().find(|(_, task)| task.0 == number);
        live_cost(agent.map(|(agent, _)| agent), turns)
    };
    let skip = queue.tasks.len().saturating_sub(ROWS);
    let rows = queue
        .tasks
        .iter()
        .skip(skip)
        .map(|task| line(task, live(task.number)))
        .collect();
    QueueRows { rows }
}

/// Publishes the rows when they changed.
fn publish(
    queue: Res<TaskQueue>,
    agents: Query<(Entity, &QueuedTask)>,
    turns: RunningTurns,
    mut sent: Local<QueueRows>,
    mut out: Publish,
) {
    let now = rows(&queue, &agents, &turns);
    if now != *sent {
        out.event(&now);
        *sent = now;
    }
}

/// The rows the agent last sent, in the view.
#[derive(Resource, Default)]
struct Shown(QueueRows);

fn mirror_rows(mut applied: MessageReader<Applied>, mut shown: ResMut<Shown>) {
    for Applied(record) in applied.read() {
        if record.kind != QueueRows::KIND {
            continue;
        }
        if let Ok(rows) = serde_json::from_value::<QueueRows>(record.data.clone()) {
            shown.0 = rows;
        }
    }
}

/// Keeps the panel while there are rows, as tall as they need, and asks
/// for a frame with them.
fn place_panel(
    shown: Res<Shown>,
    mut panels: Query<(Entity, &mut TuiPanel), With<QueuePanel>>,
    mut redraw: MessageWriter<RequestRedraw>,
    mut commands: Commands,
) {
    let rows = shown.0.rows.len();
    let height = u16::try_from(rows + 2).unwrap_or(u16::MAX);
    let panel = TuiPanel::new(Placement::Bottom(Constraint::Length(height)));
    match (panels.single_mut(), rows) {
        (Ok((entity, _)), 0) => commands.entity(entity).despawn(),
        (Ok((_, mut placed)), _) => {
            placed.set_if_neq(panel);
        }
        (Err(_), 0) => {}
        (Err(_), _) => {
            commands.spawn((QueuePanel, panel));
        }
    }
    redraw.write(RequestRedraw);
}

fn draw(mut panels: Query<&mut PanelCanvas, With<QueuePanel>>, shown: Res<Shown>) {
    for mut canvas in &mut panels {
        let block = Block::bordered().title("tasks (/queue)");
        canvas.render(Paragraph::new(shown.0.rows.join("\n")).block(block));
    }
}

/// `queue: 1 running, 2 pending · task 2 done`.
fn status(queue: Res<TaskQueue>, note: Res<QueueNote>, mut app_status: ResMut<AppStatus>) {
    let count = |state| {
        queue
            .tasks
            .iter()
            .filter(|task| task.state == state)
            .count()
    };
    let (running, pending) = (count(TaskState::Running), count(TaskState::Pending));
    let mut parts = Vec::new();
    if running + pending > 0 {
        parts.push(format!("queue: {running} running, {pending} pending"));
    }
    if !note.0.is_empty() {
        parts.push(note.0.clone());
    }
    app_status
        .0
        .show(QUEUE_ITEM.says(parts.join(" · "), Tone::Dim));
}
