//! Evaluate three independent questions about one synthetic support ticket.
//! Requires `JEV_TOKEN`; run with `cargo run -p typesafeai_triage`.

use anyhow::Result;
use rig::error::ProviderError;
use rig::prelude::*;
use rig::typesafeai::{
    Choice, ChoiceAnswer, Evaluate, Jev, Noul, NoulAnswer, Query, Score, ScoreAnswer,
};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
enum Route {
    Billing,
    Technical,
    Account,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
enum Urgency {
    Routine,
    Workaround,
    Significant,
    Critical,
}

#[derive(Debug, Serialize, Deserialize)]
struct Assessment<R = ChoiceAnswer<Route>, U = ScoreAnswer<Urgency>, D = NoulAnswer> {
    route: R,
    urgency: U,
    duplicate_charge: D,
}

type AssessmentQuery = Assessment<Choice<Route>, Score<Urgency>, Noul>;

impl AssessmentQuery {
    fn new() -> Result<Self, ProviderError> {
        Ok(Self {
            route: Choice::<Route>::new(
                "Which support team should investigate this ticket?",
                [
                    (
                        Route::Billing,
                        "Charges, invoices, refunds, and subscriptions",
                    ),
                    (Route::Technical, "Product errors and broken functionality"),
                    (Route::Account, "Login, permissions, and account access"),
                ],
            )?,
            urgency: Score::<Urgency>::new(
                "How urgently does this ticket need attention, given its customer impact?",
                [
                    (Urgency::Routine, "Routine question with no disruption"),
                    (
                        Urgency::Workaround,
                        "Inconvenience with a workable alternative",
                    ),
                    (
                        Urgency::Significant,
                        "Significant disruption or unexpected charge",
                    ),
                    (
                        Urgency::Critical,
                        "Critical ongoing loss or complete service outage",
                    ),
                ],
            )?,
            duplicate_charge: Noul::new(
                "The evidence establishes that the customer has two settled charges for the same purchase.",
            )?,
        })
    }
}

impl<R: Query, U: Query, D: Query> Query for Assessment<R, U, D> {
    type Response = Assessment<R::Response, U::Response, D::Response>;
    type Output = Assessment<R::Output, U::Output, D::Output>;

    fn decode(&self, response: Self::Response) -> Result<Self::Output, ProviderError> {
        Ok(Assessment {
            route: self.route.decode(response.route)?,
            urgency: self.urgency.decode(response.urgency)?,
            duplicate_charge: self.duplicate_charge.decode(response.duplicate_charge)?,
        })
    }
}

#[derive(Serialize)]
struct Ticket<'a> {
    message: &'a str,
    recent_events: &'a [&'a str],
}

#[tokio::main]
async fn main() -> Result<()> {
    let client = Jev::from_env()?.bound()?;
    let ticket = Ticket {
        message: "My card shows two $49 charges after upgrading. Can you fix this?",
        recent_events: &[
            "One upgrade completed yesterday",
            "One $49 charge is settled; another $49 authorization is pending",
        ],
    };

    // Questions share state, but none consumes another question's answer.
    let result = client.evaluate(&ticket, AssessmentQuery::new()?).await?;
    let Assessment {
        route,
        urgency,
        duplicate_charge,
    } = result.answers;
    println!("Route: {:?}", route.choice);
    println!("Route distribution: {:?}", route.probabilities);
    println!("Route confidence: {:.3}", route.confidence);
    println!("Urgency (expected rubric index): {:.3}", urgency.score);
    println!("Urgency distribution: {:?}", urgency.probabilities);
    println!("Urgency confidence: {:.3}", urgency.confidence);
    println!("P(duplicate settled charge): {:.3}", duplicate_charge.noul);
    Ok(())
}
