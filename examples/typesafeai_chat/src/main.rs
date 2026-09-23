//! Interactive Jev decisions, optionally followed by an ordinary Rig agent.

use anyhow::{Result, bail};
use rig::typesafeai::{
    Choice, ChoiceAnswer, Evaluate, Jev, Noul, NoulAnswer, Query, Score, ScoreAnswer,
};
use rig::{completion::Message, error::ProviderError, prelude::*, providers::openai::OpenAI};
use serde::{Deserialize, Serialize};
use std::{
    collections::VecDeque,
    io::{self, Write},
    time::Duration,
};

const MAX_TURNS: usize = 8;
const TIMEOUT: Duration = Duration::from_secs(90);

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
enum Route {
    Billing,
    Technical,
    General,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
enum Urgency {
    Routine,
    Workaround,
    Significant,
    Critical,
}

#[derive(Debug, Serialize, Deserialize)]
struct Assessment<R = ChoiceAnswer<Route>, U = ScoreAnswer<Urgency>, C = NoulAnswer> {
    route: R,
    urgency: U,
    clarification: C,
}

type AssessmentQuery = Assessment<Choice<Route>, Score<Urgency>, Noul>;

impl AssessmentQuery {
    fn new() -> Result<Self, ProviderError> {
        Ok(Self {
            route: Choice::<Route>::new(
                "Select the most useful support route for the current message, using history for context.",
                [
                    (
                        Route::Billing,
                        "Charges, subscriptions, invoices, and refunds",
                    ),
                    (
                        Route::Technical,
                        "Errors, outages, and broken product functionality",
                    ),
                    (
                        Route::General,
                        "Other requests, unclear intent, and conversation",
                    ),
                ],
            )?,
            urgency: Score::<Urgency>::new(
                "How urgent is the current request based on the evidence in the conversation?",
                [
                    (
                        Urgency::Routine,
                        "Routine conversation or question; no disruption",
                    ),
                    (Urgency::Workaround, "Inconvenience with a workaround"),
                    (
                        Urgency::Significant,
                        "Significant disruption or unexpected charge",
                    ),
                    (
                        Urgency::Critical,
                        "Critical ongoing loss or complete outage",
                    ),
                ],
            )?,
            clarification: Noul::new(
                "A focused clarifying question is needed before useful support advice can be given.",
            )?,
        })
    }
}

impl<R: Query, U: Query, C: Query> Query for Assessment<R, U, C> {
    type Response = Assessment<R::Response, U::Response, C::Response>;
    type Output = Assessment<R::Output, U::Output, C::Output>;

    fn decode(&self, response: Self::Response) -> Result<Self::Output, ProviderError> {
        Ok(Assessment {
            route: self.route.decode(response.route)?,
            urgency: self.urgency.decode(response.urgency)?,
            clarification: self.clarification.decode(response.clarification)?,
        })
    }
}

#[derive(Serialize)]
struct Turn {
    user: String,
    assistant: String,
}

#[derive(Serialize)]
struct State<'a> {
    product: &'static str,
    history: &'a VecDeque<Turn>,
    current_message: &'a str,
}

#[tokio::main]
async fn main() -> Result<()> {
    let mut with_agent = false;
    for argument in std::env::args().skip(1) {
        match argument.as_str() {
            "--agent" => with_agent = true,
            "--help" | "-h" => {
                println!(
                    "Usage: cargo run -p typesafeai_chat -- [--agent]\n\
                    JEV_TOKEN: required for Jev evaluations.\n\
                    --agent: also use OPENAI_API_KEY for gpt-5.6-sol replies.\n\
                    Commands: /reset clears history; /quit exits. EOF also exits."
                );
                return Ok(());
            }
            _ => bail!("unknown argument {argument:?}; use --help"),
        }
    }

    let jev = Jev::from_env()?.bound()?;
    let openai = if with_agent {
        Some(OpenAI::from_env()?.bound()?)
    } else {
        None
    };
    let questions = AssessmentQuery::new()?;
    let mut history = VecDeque::<Turn>::new();
    println!("Hosted workspace support demo. /reset clears history; /quit exits.");
    println!("Jev reports decisions and distributions; --agent adds gpt-5.6-sol replies.");

    loop {
        print!("\nyou> ");
        io::stdout().flush()?;
        let mut input = String::new();
        if io::stdin().read_line(&mut input)? == 0 {
            return Ok(());
        }
        let input = input.trim();
        if input.is_empty() {
            continue;
        }
        if input == "/quit" {
            return Ok(());
        }
        if input == "/reset" {
            history.clear();
            println!("History cleared.");
            continue;
        }
        let state = State {
            product: "A hosted document workspace",
            history: &history,
            current_message: input,
        };
        let evaluation = match tokio::time::timeout(TIMEOUT, jev.evaluate(&state, &questions)).await
        {
            Ok(Ok(evaluation)) => evaluation,
            Ok(Err(error)) => {
                eprintln!("Jev evaluation failed: {error}");
                continue;
            }
            Err(_) => {
                eprintln!("Jev evaluation timed out after 90 seconds.");
                continue;
            }
        };
        let Assessment {
            route,
            urgency,
            clarification,
        } = evaluation.answers;
        println!(
            "jev> {:?}; concentration {:.3}; probabilities {:?}",
            route.choice, route.confidence, route.probabilities
        );
        println!(
            "     urgency {:.3}/3; probabilities {:?}; P(clarify) {:.3}",
            urgency.score, urgency.probabilities, clarification.noul
        );

        let policy = next_action(&route, clarification.noul);
        let reply = if let Some(client) = &openai {
            let agent = client.agent("gpt-5.6-sol")
                .preamble(format!(
                    "You support a hosted document workspace. {policy} \
                     You have no account access or tools; never claim to change settings or issue refunds. \
                     Be concise. The routing decision is advisory and may be mistaken."
                ))
                .build();
            let mut messages = history
                .iter()
                .flat_map(|turn| {
                    [
                        Message::user(&turn.user),
                        Message::assistant(&turn.assistant),
                    ]
                })
                .collect::<Vec<_>>();
            match tokio::time::timeout(TIMEOUT, agent.chat(input, &mut messages)).await {
                Ok(Ok(response)) => response.output,
                Ok(Err(error)) => {
                    eprintln!("Agent reply failed (conversation unchanged): {error}");
                    continue;
                }
                Err(_) => {
                    eprintln!("Agent reply timed out after 90 seconds (conversation unchanged).");
                    continue;
                }
            }
        } else {
            format!("Suggested next action: {policy}")
        };
        println!("assistant> {reply}");
        history.push_back(Turn {
            user: input.to_owned(),
            assistant: reply,
        });
        if history.len() > MAX_TURNS {
            history.pop_front();
        }
    }
}

fn next_action(route: &ChoiceAnswer<Route>, clarification: f64) -> &'static str {
    // Demo policy: concentration and thresholds need tuning on representative data.
    if clarification >= 0.7 || route.confidence < 0.6 {
        return "Ask one focused question about the user's goal or missing evidence.";
    }
    match route.choice {
        Route::Billing => {
            "Help investigate charges and subscriptions; distinguish pending authorizations from settled charges."
        }
        Route::Technical => {
            "Suggest reversible troubleshooting steps suited to the reported error."
        }
        Route::General => "Respond conversationally and help the user clarify what they need.",
    }
}
