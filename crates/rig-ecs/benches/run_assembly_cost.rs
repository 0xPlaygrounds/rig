//! Assembly cost over a long tool loop: renders per turn and elapsed.
//!
//! Run explicitly with `cargo bench -p rig-ecs --bench run_assembly_cost`.
//! Verification compiles this benchmark without executing it.
//! Prints one line per turn — the history length (what an uncached assembly
//! renders every turn), the full part-subtree renders the turn made
//! (`AssemblyStats`), the cache hits, and the elapsed time of the turn —
//! and writes the same as JSON to `$RIG_ASSEMBLY_MEASURE_OUT` when set.
//!
//! To run this at a base without `AssemblyStats`, delete the two lines
//! marked `N2` below: the `history` column is what such a base renders.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::print_stdout
)]

#[path = "../tests/run_support/mod.rs"]
mod run_support;

use std::time::Instant;

use bevy_ecs::prelude::*;
use rig_core::{
    effect::{EffectKind, FamilyDescriptor, HandlerDescriptor, HandlerKey, Outcome},
    error::{ErrorKind, ErrorReport},
    message::AssistantContent,
    serve::{Dispatch, Reply, Serve},
    tool::{ToolOutput, ToolResult},
};
use rig_ecs::{
    agent::{Failed, Grant, MaxTurns, Order, Settled, Utterance},
    bus::{PendingEffect, RigSchedule},
    systems::{RigSet, RunCommands},
};
use run_support::*;

const MODEL: &str = "bench/model:default";
const TOOL: &str = "bench/tool:read#0";
const TOOL_TURNS: usize = 8;
/// Bytes of text each tool result carries: enough that a render is work.
const RESULT_BYTES: usize = 8 * 1024;

/// A tool answering every call with `RESULT_BYTES` of text, in sequence.
struct Reader;

impl Serve for Reader {
    type Family = rig_core::effect::family::Tool;

    fn descriptor(&self) -> HandlerDescriptor {
        HandlerDescriptor {
            key: HandlerKey::from(TOOL),
            family: FamilyDescriptor::Tool {
                name: "read".to_owned(),
                description: "reads a chunk".to_owned(),
                parameters: serde_json::json!({"type": "object", "properties": {"n": {"type": "integer"}}}),
                embedding: None,
            },
            layers: Vec::new(),
        }
    }

    async fn serve(&self, kind: EffectKind, _dispatch: Dispatch) -> Reply {
        let EffectKind::ToolCall { args, .. } = kind else {
            return Reply::Outcome(Err(ErrorReport::new(ErrorKind::Request, "a tool call")));
        };
        let n = serde_json::from_str::<serde_json::Value>(&args)
            .ok()
            .and_then(|args| args.get("n").and_then(serde_json::Value::as_u64))
            .unwrap_or(0);
        let line = format!("chunk {n}: ");
        let mut text = String::with_capacity(RESULT_BYTES + line.len());
        while text.len() < RESULT_BYTES {
            text.push_str(&line);
            text.push_str("lorem ipsum dolor sit amet ");
        }
        Reply::Outcome(Ok(Outcome::ToolResult {
            result: ToolResult::success(ToolOutput::text(text)),
        }))
    }
}

/// One line per folded request: taken in `RigSet::Patch` the pass the
/// request was folded, timed from `RigSet::Select` of the same pass, so the
/// elapsed time is the assembly's (the pass's Select-to-Patch span), not
/// the tool's or the model's.
#[derive(Resource, Default)]
struct Turns {
    started: Option<Instant>,
    rows: Vec<serde_json::Value>,
}

fn start(mut turns: ResMut<Turns>) {
    turns.started = Some(Instant::now());
}

fn measure(
    effects: Query<(&PendingEffect, &ChildOf), Added<PendingEffect>>,
    parents: Query<&ChildOf>,
    utterances: Query<&ChildOf, With<Utterance>>,
    // N2: delete this parameter to run at a base without `AssemblyStats`.
    stats: Res<rig_ecs::agent::content::cache::AssemblyStats>,
    mut turns: ResMut<Turns>,
) {
    for (effect, turn) in &effects {
        let EffectKind::Completion { request, .. } = &effect.kind else {
            continue;
        };
        let run = parents.get(turn.parent()).unwrap().parent();
        let history = utterances
            .iter()
            .filter(|parent| parent.parent() == run)
            .count();
        let elapsed = turns
            .started
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1e3);
        let index = turns.rows.len();
        let mut row = serde_json::json!({
            "turn": index + 1,
            "history": history,
            "chat_history": request.chat_history.len(),
            "assemble_ms": elapsed,
        });
        // N2: delete this statement to run at a base without `AssemblyStats`.
        row["renders"] = serde_json::json!(stats.renders);
        row["hits"] = serde_json::json!(stats.hits);
        turns.rows.push(row);
    }
}

fn main() {
    let mut app = app();
    let mut script: Vec<Vec<AssistantContent>> = (0..TOOL_TURNS)
        .map(|n| vec![call(&format!("c{n}"), "read", serde_json::json!({"n": n}))])
        .collect();
    script.push(vec![AssistantContent::text("done")]);
    let (model, _requests) = Scripted::new(MODEL, script);
    let model = register(&mut app, MODEL, model);
    let tool = register(&mut app, TOOL, Reader);
    let agent = spawn_agent(app.world_mut(), "bench", model);
    app.world_mut()
        .entity_mut(agent)
        .insert(MaxTurns(TOOL_TURNS + 2));
    app.world_mut()
        .spawn((Grant(tool), Order(0), ChildOf(agent)));
    app.insert_resource(Turns::default());
    app.world_mut().resource_mut::<Schedules>().add_systems(
        RigSchedule,
        (start.in_set(RigSet::Select), measure.in_set(RigSet::Patch)),
    );

    let started = Instant::now();
    let run = app
        .world_mut()
        .spawn_run(agent, &[], "read everything", false, None);
    tick_until(&mut app, "the run settles", |world| {
        world.get::<Settled>(run).is_some() || world.get::<Failed>(run).is_some()
    });
    assert!(
        app.world().get::<Settled>(run).is_some(),
        "{:?}",
        app.world().get::<Failed>(run)
    );
    let total_ms = started.elapsed().as_secs_f64() * 1e3;

    let rows = std::mem::take(&mut app.world_mut().resource_mut::<Turns>().rows);
    assert_eq!(rows.len(), TOOL_TURNS + 1);
    // Per-turn deltas of the cumulative counters.
    let mut previous = (0u64, 0u64);
    let mut report = Vec::with_capacity(rows.len());
    println!("turn  history  renders  hits  assemble_ms");
    for row in rows {
        let renders = row.get("renders").and_then(serde_json::Value::as_u64);
        let hits = row.get("hits").and_then(serde_json::Value::as_u64);
        let delta = (
            renders.map(|r| r - previous.0),
            hits.map(|h| h - previous.1),
        );
        previous = (renders.unwrap_or(0), hits.unwrap_or(0));
        let out = serde_json::json!({
            "turn": row["turn"],
            "history": row["history"],
            "renders": delta.0,
            "hits": delta.1,
            "assemble_ms": row["assemble_ms"],
        });
        println!(
            "{:>4}  {:>7}  {:>7}  {:>4}  {:>10.3}",
            out["turn"],
            out["history"],
            delta.0.map_or("-".to_owned(), |r| r.to_string()),
            delta.1.map_or("-".to_owned(), |h| h.to_string()),
            out["assemble_ms"].as_f64().unwrap_or(0.0),
        );
        report.push(out);
    }
    println!("total_ms {total_ms:.3}");
    if let Ok(path) = std::env::var("RIG_ASSEMBLY_MEASURE_OUT") {
        let json = serde_json::json!({
            "tool_turns": TOOL_TURNS,
            "result_bytes": RESULT_BYTES,
            "total_ms": total_ms,
            "turns": report,
        });
        std::fs::write(&path, serde_json::to_string_pretty(&json).unwrap()).unwrap();
        println!("wrote {path}");
    }
}
