//! Debate: `/debate [--rounds N] <question>` spawns three helper agents of
//! the agent shown, `pro` and `con` on two models and a `judge`. Round 1
//! briefs both debaters with the question at once; each later round sends
//! each the other's last answer. Then the judge reads the whole exchange,
//! and its verdict goes to the agent's conversation as a note and to the
//! user as a notice. The debaters keep their conversations, so `/agents`
//! shows the exchange, and each one's line there what it spent.
//!
//! Debaters may use the agent's read-only tools; the judge has none.
//! `/debate stop`, or interrupting the agent, stops all three. The models
//! come from the `[debate]` settings table ([`DebateConfig`]).

use rig_basics::sessions::agents::{AgentColumn, ColumnOf};
use rig_basics::subagents::{SpawnSubagent, SpawnedAgent, SubagentsPlugin};
use rig_basics::telemetry::usage::AgentUsage;
use rig_harness::prelude::*;
use serde::Deserialize;

/// The rounds when neither the settings nor `--rounds` say.
const ROUNDS: u32 = 2;

/// The most rounds `--rounds` takes.
const MAX_ROUNDS: u32 = 10;

/// The `[debate]` settings table. A model left out defaults: `pro` and
/// `judge` to the agent's own, `con` to a reachable model of another
/// provider than `pro`'s.
///
/// ```toml
/// [debate]
/// pro = "openai/gpt-5.1"
/// con = "anthropic/claude-sonnet-4-5"
/// rounds = 3
/// ```
#[derive(Resource, Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct DebateConfig {
    /// The catalog model (`vendor/model`) arguing for.
    pub pro: Option<String>,
    /// The catalog model arguing against.
    pub con: Option<String>,
    /// The catalog model that judges.
    pub judge: Option<String>,
    /// Rounds unless `--rounds` says; 2 when `None`.
    pub rounds: Option<u32>,
}

/// A debate's part.
#[derive(Reflect, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Role {
    /// Argues for.
    #[default]
    Pro,
    /// Argues against.
    Con,
    /// Weighs both and recommends.
    Judge,
}

impl Role {
    fn title(self) -> &'static str {
        match self {
            Self::Pro => "debate pro",
            Self::Con => "debate con",
            Self::Judge => "debate judge",
        }
    }

    fn prompt(self) -> &'static str {
        match self {
            Self::Pro => PRO_PROMPT,
            Self::Con => CON_PROMPT,
            Self::Judge => JUDGE_PROMPT,
        }
    }
}

const PRO_PROMPT: &str = "You argue FOR in a debate on a design decision: for the proposal, or \
    for the first option the question names. You may read the codebase with your tools to \
    ground your points. Make the strongest honest case in a few short paragraphs; in later \
    rounds, answer your opponent's last argument point by point. Concede what is true.";

const CON_PROMPT: &str = "You argue AGAINST in a debate on a design decision: against the \
    proposal, or for the alternatives to the first option the question names. You may read \
    the codebase with your tools to ground your points. Make the strongest honest case in a \
    few short paragraphs; in later rounds, answer your opponent's last argument point by \
    point. Concede what is true.";

const JUDGE_PROMPT: &str = "You judge a debate on a design decision. You get the question \
    and every round of both sides. Summarize the strongest points of each side, say which \
    claims were left unanswered, and end with a clear recommendation and why. Be short.";

/// On a debate's agent: its part. Saved, so a restart keeps it idle.
#[derive(Component, Reflect, Clone, Copy, Debug, Default)]
#[reflect(Component, Saved, Default, Clone, Debug)]
pub struct Debater(pub Role);

/// On a [`SpawnSubagent`] request of a debate: the part of the agent.
#[derive(Component, Clone, Copy, Debug)]
struct Seat(Role);

/// On the `/agents` column that shows what a debater spent.
#[derive(Component)]
struct SpentColumn;

/// One round's answers.
#[derive(Clone, Debug, Default)]
pub struct Round {
    /// The `pro` answer, once given.
    pub pro: Option<String>,
    /// The `con` answer, once given.
    pub con: Option<String>,
}

/// On an agent while its debate runs. Not saved: a restart ends it.
#[derive(Component, Clone, Debug)]
pub struct Debate {
    /// The question.
    pub question: String,
    /// The rounds to argue.
    pub rounds: u32,
    /// The debate's agents once spawned, by part.
    pub seated: Vec<(Role, Entity)>,
    /// The rounds so far, the running one last.
    pub exchange: Vec<Round>,
}

impl Debate {
    fn agent(&self, role: Role) -> Option<Entity> {
        let seat = self.seated.iter().find(|(seated, _)| *seated == role);
        seat.map(|&(_, agent)| agent)
    }

    fn role(&self, agent: Entity) -> Option<Role> {
        let seat = self.seated.iter().find(|(_, seated)| *seated == agent);
        seat.map(|&(role, _)| role)
    }

    /// Whether every round has both answers: the judge's turn.
    fn argued(&self) -> bool {
        let full = |round: &Round| round.pro.is_some() && round.con.is_some();
        self.exchange.len() >= self.rounds as usize && self.exchange.iter().all(full)
    }

    /// The question and every round, for the judge.
    fn transcript(&self) -> String {
        let mut text = format!("The question: {}\n", self.question);
        for (at, round) in self.exchange.iter().enumerate() {
            let said = |answer: &Option<String>| answer.clone().unwrap_or_default();
            text.push_str(&format!(
                "\n## Round {}\n\n### For\n{}\n\n### Against\n{}\n",
                at + 1,
                said(&round.pro),
                said(&round.con)
            ));
        }
        text.push_str("\nWeigh both sides and recommend.");
        text
    }
}

/// Adds `/debate`; see the [crate](self) docs.
#[derive(Default)]
pub struct DebatePlugin;

impl Plugin for DebatePlugin {
    fn build(&self, app: &mut App) {
        if !app.is_plugin_added::<SubagentsPlugin>() {
            app.add_plugins(SubagentsPlugin);
        }
        app.add_setting::<DebateConfig>("debate")
            .add_command(
                "debate",
                "Two models argue a question for N rounds (--rounds N), a judge recommends; \
                 /debate stop stops it",
                debate,
            )
            .add_observer(seat)
            .add_observer(follow_successor::<Seat>)
            .add_observer(add_spent_column)
            .add_observer(step)
            .add_observer(stop_on_interrupt)
            .add_systems(Update, show_spent);
    }
}

/// `--rounds N` and the question.
fn parse(args: &str, rounds: u32) -> Result<(u32, String), String> {
    let (rounds, question) = match args.strip_prefix("--rounds") {
        Some(rest) => {
            let rest = rest.trim_start();
            let (count, question) = rest.split_once(char::is_whitespace).unwrap_or((rest, ""));
            let count: u32 = count
                .parse()
                .map_err(|_| format!("--rounds takes a number, not `{count}`"))?;
            if !(1..=MAX_ROUNDS).contains(&count) {
                return Err(format!("--rounds takes 1 to {MAX_ROUNDS}"));
            }
            (count, question.trim())
        }
        None => (rounds, args),
    };
    if question.is_empty() {
        return Err("/debate needs a question: /debate [--rounds N] <question>".to_owned());
    }
    Ok((rounds, question.to_owned()))
}

/// `/debate [--rounds N] <question>` and `/debate stop`.
fn debate(
    In(args): In<CommandArgs>,
    hosts: Query<(Option<&ModelChoice>, &ToolAccess, Option<&Debate>)>,
    tools: Query<(&ToolDef, &Footprint)>,
    config: Res<DebateConfig>,
    models: Res<Models>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let host = args.agent;
    let Ok((own, access, running)) = hosts.get(host) else {
        return;
    };
    match (args.args.as_str(), running) {
        ("stop", Some(debate)) => {
            stop(&mut commands, host, debate);
            notices.write(Notice::info(host, "Debate stopped."));
            return;
        }
        ("stop", None) => {
            notices.write(Notice::error(host, "No debate runs."));
            return;
        }
        (_, Some(_)) => {
            let why = "A debate runs already; /debate stop stops it.";
            notices.write(Notice::error(host, why));
            return;
        }
        (_, None) => {}
    }
    let started = parse(&args.args, config.rounds.unwrap_or(ROUNDS))
        .and_then(|(rounds, question)| Ok((rounds, question, con_model(&config, own, &models)?)));
    let (rounds, question, con) = match started {
        Ok(started) => started,
        Err(why) => {
            notices.write(Notice::error(host, why));
            return;
        }
    };
    // The agent's read-only tools; the judge gets none.
    let read_only: Vec<String> = tools
        .iter()
        .filter(|(def, footprint)| footprint.0.reads_only() && access.allows(def.0.name.as_str()))
        .map(|(def, _)| def.0.name.as_str().to_owned())
        .collect();
    let brief = format!("The question: {question}\n\nRound 1 of {rounds}: make your case.");
    let seats = [
        (
            Role::Pro,
            config.pro.clone(),
            read_only.clone(),
            brief.clone(),
        ),
        (Role::Con, Some(con), read_only, brief),
        (Role::Judge, config.judge.clone(), Vec::new(), String::new()),
    ];
    for (role, model, tools, brief) in seats {
        let request = SpawnSubagent {
            model,
            tools: Some(tools),
            ..SpawnSubagent::helper(role.title())
                .system_prompt(role.prompt())
                .brief(brief)
        };
        commands.spawn((request, ForAgent(host), Seat(role)));
    }
    notices.write(Notice::info(
        host,
        format!("Debate started: {rounds} rounds; /agents shows pro, con and judge."),
    ));
    commands.entity(host).insert(Debate {
        question,
        rounds,
        seated: Vec::new(),
        exchange: vec![Round::default()],
    });
}

/// The catalog reference of `con`: the settings', else a reachable model
/// that calls tools, of another provider than `pro`'s when one is.
fn con_model(
    config: &DebateConfig,
    own: Option<&ModelChoice>,
    models: &Models,
) -> Result<String, String> {
    if let Some(con) = &config.con {
        return Ok(con.clone());
    }
    let pro = config.pro.as_deref().or(own.map(|own| own.0.as_str()));
    let pro = pro.ok_or("No model for pro: pick one with /model or set `pro` in [debate]")?;
    let pro = models
        .0
        .catalog()
        .resolve(pro)
        .map_err(|why| format!("{pro}: {why}"))?;
    let reachable = models.0.reachable();
    let other = reachable
        .iter()
        .find(|spec| spec.provider != pro.spec.provider)
        .or_else(|| {
            let pro = pro.spec.reference();
            reachable.iter().find(|spec| spec.reference() != pro)
        });
    let other = other.ok_or("No second model is reachable; set `con` in [debate]")?;
    Ok(other.reference())
}

/// Seats each debate agent once its request settled, or stops the debate
/// when one was refused.
fn seat(
    settled: On<Add<Settled>>,
    requests: Query<(&Seat, &ForAgent, &Outcome, Option<&SpawnedAgent>)>,
    mut debates: Query<&mut Debate>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let Ok((&Seat(role), &ForAgent(host), outcome, spawned)) = requests.get(settled.entity) else {
        return;
    };
    let Ok(mut debate) = debates.get_mut(host) else {
        return;
    };
    match (spawned, &outcome.0) {
        (Some(&SpawnedAgent(agent)), Ok(_)) => {
            commands.entity(agent).insert(Debater(role));
            debate.seated.push((role, agent));
        }
        (_, failed) => {
            let why = failed.as_ref().err().map(|why| why.message.clone());
            let why = why.unwrap_or_else(|| "no agent was spawned".to_owned());
            let text = format!("Debate stopped: {}: {why}", role.title());
            notices.write(Notice::error(host, text));
            stop(&mut commands, host, &debate);
        }
    }
}

/// Stops `debate` of `host`: no step follows, and every running turn of
/// it is interrupted.
fn stop(commands: &mut Commands, host: Entity, debate: &Debate) {
    commands.entity(host).remove::<Debate>();
    for &(_, agent) in &debate.seated {
        let reason = Some("the debate stopped".to_owned());
        commands.trigger(StopTurn {
            entity: agent,
            reason,
        });
    }
}

/// Interrupting the agent stops its debate.
fn stop_on_interrupt(interrupt: On<StopTurn>, debates: Query<&Debate>, mut commands: Commands) {
    if let Ok(debate) = debates.get(interrupt.entity) {
        stop(&mut commands, interrupt.entity, debate);
    }
}

/// Gives a debater, spawned or restored, its spend column in `/agents`.
fn add_spent_column(inserted: On<Insert<Debater>>, mut commands: Commands) {
    let column = AgentColumn::new("debate-spent", "");
    commands.spawn((column, ColumnOf(inserted.entity), SpentColumn));
}

/// Keeps each debater's spend column current.
fn show_spent(
    mut columns: Query<(&mut AgentColumn, &ColumnOf), With<SpentColumn>>,
    spent: Query<Ref<AgentUsage>>,
) {
    for (mut column, &ColumnOf(agent)) in &mut columns {
        if let Ok(spending) = spent.get(agent)
            && (spending.is_changed() || column.is_added())
        {
            column.text = format!("spent {}", spending.total.cost_or_tokens());
        }
    }
}

/// What a debater's turn ended with: the next round once both answered,
/// the judge after the last, and the verdict after the judge. A turn that
/// ended without an answer stops the debate.
fn step(
    settled: On<Add<Settled>>,
    turns: Query<(&TurnOutcome, &ForAgent)>,
    parents: Query<&SpawnedBy, With<Debater>>,
    mut debates: Query<&mut Debate>,
    ids: Query<&AgentId>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let Ok((outcome, &ForAgent(agent))) = turns.get(settled.entity) else {
        return;
    };
    let Ok(&SpawnedBy(host)) = parents.get(agent) else {
        return;
    };
    let Ok(mut debate) = debates.get_mut(host) else {
        return;
    };
    let Some(role) = debate.role(agent) else {
        return;
    };
    let answer = match outcome {
        TurnOutcome::Answered(message) => message.final_answer().ok_or("it gave no answer"),
        TurnOutcome::Failed(why) => Err(why.as_str()),
        TurnOutcome::Stopped(_) => Err("it was stopped"),
    };
    let answer = match answer {
        Ok(answer) => answer,
        Err(why) => {
            let text = format!("Debate stopped: {}: {why}", role.title());
            notices.write(Notice::error(host, text));
            stop(&mut commands, host, &debate);
            return;
        }
    };
    let from = |debate: &Debate, role: Role| {
        let id = debate.agent(role).and_then(|agent| ids.get(agent).ok());
        Author::agent(id.cloned().unwrap_or_default(), None).titled(role.title())
    };
    if role == Role::Judge {
        if !debate.argued() {
            return;
        }
        let text = format!("Debate verdict on \"{}\":\n\n{answer}", debate.question);
        let note = Deliver::new(host, text, DeliveryMode::Inform);
        commands.trigger(note.with_author(from(&debate, role)));
        notices.write(Notice::info(
            host,
            format!(
                "Debate verdict:\n{answer}\n/agents shows the exchange and what each side spent."
            ),
        ));
        commands.entity(host).remove::<Debate>();
        return;
    }
    let (rounds, at) = (debate.rounds, debate.exchange.len());
    let Some(round) = debate.exchange.last_mut() else {
        return;
    };
    // A turn the user started after this side answered the round is no
    // answer of the debate.
    let side = if role == Role::Pro {
        &mut round.pro
    } else {
        &mut round.con
    };
    if side.is_some() {
        return;
    }
    *side = Some(answer);
    let (Some(pro), Some(con)) = (round.pro.clone(), round.con.clone()) else {
        return;
    };
    if at < rounds as usize {
        debate.exchange.push(Round::default());
        let next = |said: String| {
            format!(
                "{said}\n\n(Round {} of {rounds}: answer your opponent's argument above.)",
                at + 1
            )
        };
        let replies = [
            (Role::Pro, next(con), Role::Con),
            (Role::Con, next(pro), Role::Pro),
        ];
        for (to, text, by) in replies {
            if let Some(to) = debate.agent(to) {
                let reply = Deliver::new(to, text, DeliveryMode::Queue);
                commands.trigger(reply.with_author(from(&debate, by)));
            }
        }
    } else if let Some(judge) = debate.agent(Role::Judge) {
        let brief = Deliver::new(judge, debate.transcript(), DeliveryMode::Queue);
        commands.trigger(brief.with_author(Author::plugin("debate")));
    }
}

#[cfg(test)]
mod tests;
