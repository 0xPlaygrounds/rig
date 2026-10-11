//! Debate: `/debate [--rounds N] <question>` spawns three subagents of the
//! agent shown, `pro` and `con` on two models and a `judge`. Round 1 sends
//! both debaters the question at once; each later round sends each the
//! other's last answer. Then the judge reads the whole exchange, and its
//! verdict goes to the agent's conversation as a note and to the user as a
//! notice, with what each debater spent. The debaters keep their
//! conversations, so `/agents` shows the exchange.
//!
//! Debaters may use the agent's read-only tools; the judge has none.
//! `/debate stop`, or interrupting the agent, stops all three. The models
//! come from [`DebateConfig`], which `RIG_HOME/debate.json` fills.

use rig_harness::prelude::*;
use rig_telemetry::Spending;
use serde::Deserialize;

/// The rounds when neither the config nor `--rounds` says.
const ROUNDS: u32 = 2;

/// The most rounds `--rounds` takes.
const MAX_ROUNDS: u32 = 10;

/// Who the plugin's messages are from.
const PLUGIN: &str = "debate";

/// The models and rounds of a debate. `None` models default: `pro` and
/// `judge` to the agent's own, `con` to a reachable model of another
/// provider. Insert it before adding the plugin, or write it as
/// `RIG_HOME/debate.json`, which the agent reads at startup.
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
    fn name(self) -> &'static str {
        match self {
            Self::Pro => "pro",
            Self::Con => "con",
            Self::Judge => "judge",
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

/// On a debate's subagent: its part. Saved, so a restart names it again
/// and keeps it idle.
#[derive(Component, Reflect, Clone, Copy, Debug, Default)]
#[reflect(Component, Saved, Default, Clone, Debug)]
pub struct Debater(pub Role);

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
    /// The subagents arguing for and against, and judging.
    pub pro: Entity,
    pub con: Entity,
    pub judge: Entity,
    /// The rounds so far, the running one last.
    pub exchange: Vec<Round>,
}

impl Debate {
    fn debaters(&self) -> [Entity; 3] {
        [self.pro, self.con, self.judge]
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
        app.init_resource::<DebateConfig>()
            .add_command(
                "debate",
                "Two models argue a question for N rounds (--rounds N), a judge recommends; \
                 /debate stop stops it",
                debate,
            )
            .add_systems(Startup, read_config)
            .add_observer(name_debater)
            .add_observer(keep_idle)
            .add_observer(step)
            .add_observer(stop_on_interrupt);
    }
}

/// Reads `RIG_HOME/debate.json` in an agent the launcher started.
fn read_config(
    invoked: Option<Res<Invoked>>,
    mut config: ResMut<DebateConfig>,
    mut notices: MessageWriter<Notice>,
) {
    if invoked.is_none() {
        return;
    }
    let path = Home::from_env().root().join("debate.json");
    let Ok(text) = std::fs::read_to_string(&path) else {
        return;
    };
    match serde_json::from_str(&text) {
        Ok(read) => *config = read,
        Err(why) => {
            notices.write(Notice::error(None, format!("{}: {why}", path.display())));
        }
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

/// What the agent the debate is for has: its model, connection and tools.
type Host<'a> = (
    Option<&'a ModelChoice>,
    Option<&'a Connection>,
    &'a ToolAccess,
    Has<Debate>,
);

/// `/debate [--rounds N] <question>` and `/debate stop`.
fn debate(
    In(args): In<CommandArgs>,
    hosts: Query<Host>,
    debates: Query<&Debate>,
    tools: Query<(&ToolDef, &Footprint)>,
    config: Res<DebateConfig>,
    models: Res<Models>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let host = args.agent;
    if args.args == "stop" {
        match debates.get(host) {
            Ok(debate) => {
                stop(&mut commands, host, debate);
                notices.write(Notice::info(host, "Debate stopped."));
            }
            Err(_) => {
                notices.write(Notice::error(host, "No debate runs."));
            }
        }
        return;
    }
    let Ok((own, connection, access, running)) = hosts.get(host) else {
        return;
    };
    if running {
        notices.write(Notice::error(
            host,
            "A debate runs already; /debate stop stops it.",
        ));
        return;
    }
    let started = parse(&args.args, config.rounds.unwrap_or(ROUNDS))
        .and_then(|(rounds, question)| Ok((rounds, question, choose(&config, own, &models)?)));
    let (rounds, question, [pro, con, judge]) = match started {
        Ok(started) => started,
        Err(why) => {
            notices.write(Notice::error(host, why));
            return;
        }
    };
    // The agent's read-only tools; the judge gets none.
    let read_only: Vec<String> = tools
        .iter()
        .filter(|(def, footprint)| {
            **footprint == Footprint::ReadOnly && access.allows(def.0.name.as_str())
        })
        .map(|(def, _)| def.0.name.as_str().to_owned())
        .collect();
    let mut spawn = |role: Role, model: String, tools: Vec<String>| {
        let mut agent = commands.spawn_empty();
        // Before the model choice, so connecting it keeps this connection.
        if let Some(connection) = connection.filter(|_| own.is_some_and(|own| own.0 == model)) {
            agent.insert(connection.clone());
        }
        agent.insert((
            Agent,
            SpawnedBy(host),
            Debater(role),
            SystemPrompt(role.prompt().to_owned()),
            ToolAccess::Only(tools),
            ModelChoice(model),
        ));
        agent.id()
    };
    let debate = Debate {
        pro: spawn(Role::Pro, pro, read_only.clone()),
        con: spawn(Role::Con, con, read_only),
        judge: spawn(Role::Judge, judge, Vec::new()),
        question,
        rounds,
        exchange: vec![Round::default()],
    };
    let brief = format!(
        "The question: {}\n\nRound 1 of {rounds}: make your case.",
        debate.question
    );
    for debater in [debate.pro, debate.con] {
        let brief = Deliver::new(debater, brief.clone(), DeliveryMode::Queue);
        commands.trigger(brief.with_origin(Origin::plugin(PLUGIN)));
    }
    notices.write(Notice::info(
        host,
        format!("Debate started: {rounds} rounds; /agents shows pro, con and judge."),
    ));
    commands.entity(host).insert(debate);
}

/// The catalog references of `pro`, `con` and `judge`: the config's, else
/// the agent's own model for `pro` and `judge`, and for `con` a reachable
/// model that calls tools, of another provider than `pro`'s when one is.
fn choose(
    config: &DebateConfig,
    own: Option<&ModelChoice>,
    models: &Models,
) -> Result<[String; 3], String> {
    let catalog = models.0.catalog();
    let resolve = |named: &str| {
        catalog
            .resolve(named)
            .map(|found| found.spec)
            .map_err(|why| format!("Debate model {named}: {why}"))
    };
    let own = own.map(|own| own.0.as_str());
    let pick = |named: Option<&String>, part: &str| {
        let named = named.map(String::as_str).or(own).ok_or_else(|| {
            format!("No model for {part}: pick one with /model or set `{part}` in debate.json")
        })?;
        resolve(named)
    };
    let pro = pick(config.pro.as_ref(), "pro")?;
    let judge = pick(config.judge.as_ref(), "judge")?;
    let con = match &config.con {
        Some(named) => resolve(named)?.reference(),
        None => {
            let reachable = models.0.reachable();
            let other = reachable
                .iter()
                .find(|spec| spec.provider != pro.provider)
                .or_else(|| {
                    let pro = pro.reference();
                    reachable.iter().find(|spec| spec.reference() != pro)
                });
            let other = other.ok_or("No second model is reachable; set `con` in debate.json")?;
            other.reference()
        }
    };
    Ok([pro.reference(), con, judge.reference()])
}

/// Stops `debate` of `host`: no step follows, and every running turn of
/// it is interrupted.
fn stop(commands: &mut Commands, host: Entity, debate: &Debate) {
    commands.entity(host).remove::<Debate>();
    for debater in debate.debaters() {
        commands.trigger(Interrupt { entity: debater });
    }
}

/// Interrupting the agent stops its debate.
fn stop_on_interrupt(interrupt: On<Interrupt>, debates: Query<&Debate>, mut commands: Commands) {
    if let Ok(debate) = debates.get(interrupt.entity) {
        stop(&mut commands, interrupt.entity, debate);
    }
}

/// Names a debater by its part, when spawned or restored.
fn name_debater(inserted: On<Insert<Debater>>, debaters: Query<&Debater>, mut commands: Commands) {
    if let Ok(Debater(role)) = debaters.get(inserted.entity) {
        let name = Name::new(format!("debate {}", role.name()));
        commands.entity(inserted.entity).insert(name);
    }
}

/// A restart ends the debate: its subagents stay idle.
fn keep_idle(mut restored: On<Restored>, debaters: Query<(), With<Debater>>) {
    if debaters.contains(restored.entity) {
        restored.event_mut().resume = false;
    }
}

/// What the debate's agents spent, one line each.
type Spent<'a> = (&'a Name, Option<&'a ModelChoice>, Option<&'a Spending>);

/// Takes a debater's answer: the next round once both answered, the
/// judge after the last, and the verdict after the judge. A turn that
/// ended without an answer stops the debate.
fn step(
    ended: On<TurnEnded>,
    parents: Query<&SpawnedBy, With<Debater>>,
    mut debates: Query<&mut Debate>,
    ids: Query<&AgentId>,
    spent: Query<Spent>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let agent = ended.entity;
    let Ok(&SpawnedBy(host)) = parents.get(agent) else {
        return;
    };
    let Ok(mut debate) = debates.get_mut(host) else {
        return;
    };
    let answer = match &ended.outcome {
        TurnOutcome::Answered(message) => final_answer(message),
        _ => None,
    };
    let Some(answer) = answer.filter(|_| debate.debaters().contains(&agent)) else {
        if debate.debaters().contains(&agent) {
            let why = match &ended.outcome {
                TurnOutcome::Failed(why) => why.clone(),
                _ => "it ended without an answer".to_owned(),
            };
            let name = spent.get(agent).map(|(name, ..)| name.to_string());
            let name = name.unwrap_or_default();
            notices.write(Notice::error(
                host,
                format!("Debate stopped: {name}: {why}"),
            ));
            stop(&mut commands, host, &debate);
        }
        return;
    };
    let from = |of: Entity, title: &str| {
        let id = ids.get(of).cloned().unwrap_or_default();
        Origin::agent(id, None).titled(title)
    };
    if agent == debate.judge {
        let text = format!("Debate verdict on \"{}\":\n\n{answer}", debate.question);
        let note = Deliver::new(host, text, DeliveryMode::Note);
        commands.trigger(note.with_origin(from(agent, "debate judge")));
        let spending = spent.iter_many(debate.debaters()).flatten();
        let spending = spending.map(|(name, model, spending)| {
            let model = model.map(|model| model.0.as_str()).unwrap_or("no model");
            let spent = spending.map_or("not counted".to_owned(), |s| s.0.cost_or_tokens());
            format!("  {name} ({model}): {spent}")
        });
        let spending: Vec<String> = spending.collect();
        notices.write(Notice::info(
            host,
            format!(
                "Debate verdict:\n{answer}\nSpent:\n{}\n/agents shows the exchange.",
                spending.join("\n")
            ),
        ));
        commands.entity(host).remove::<Debate>();
        return;
    }
    let (rounds, is_pro) = (debate.rounds, agent == debate.pro);
    let at = debate.exchange.len();
    let Some(round) = debate.exchange.last_mut() else {
        return;
    };
    if is_pro {
        round.pro = Some(answer);
    } else {
        round.con = Some(answer);
    }
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
            (debate.pro, next(con), from(debate.con, "con")),
            (debate.con, next(pro), from(debate.pro, "pro")),
        ];
        for (to, text, origin) in replies {
            let reply = Deliver::new(to, text, DeliveryMode::Queue);
            commands.trigger(reply.with_origin(origin));
        }
    } else {
        let brief = Deliver::new(debate.judge, debate.transcript(), DeliveryMode::Queue);
        commands.trigger(brief.with_origin(Origin::plugin(PLUGIN)));
    }
}

#[cfg(test)]
mod tests;
