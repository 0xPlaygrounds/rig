//! `cargo xtask cassette spend`: a conservative cost for every recording
//! attempt in the ledger.
//!
//! A recorded attempt costs what its fixture's own usage counters say, at
//! list prices with cached-input discounts ignored. A reply without usage
//! counts its request at three bytes a token plus its whole output budget. A
//! failed attempt costs at least its cell's recorded cost and never less than
//! the floor, except on local servers, which cost nothing. A run whose
//! `started` row has no result row was interrupted and counts as failed.

use std::collections::BTreeMap;
use std::path::Path;

use serde::Deserialize;
use serde_json::Value;

/// A failed attempt on a paid wire costs at least this much.
pub(crate) const FAILED_ATTEMPT_FLOOR: f64 = 0.05;

/// Local servers: recording costs nothing.
const LOCAL: &[&str] = &["ollama", "llamacpp", "mistralrs"];

/// Dollars per million input and output tokens: roughly the list price of
/// the model most of the provider's cells record with, rounded up. A cell on
/// a pricier model (Claude Opus) costs more than this.
fn prices(provider: &str) -> (f64, f64) {
    match provider {
        "openai" | "cohere" => (2.5, 10.0),
        "gemini" => (0.5, 3.0),
        "anthropic" => (3.0, 15.0),
        "openrouter" => (2.0, 15.0),
        "xai" => (3.0, 15.0),
        "deepseek" => (0.28, 0.42),
        "groq" => (0.1, 0.5),
        "venice" | "mistral" => (0.1, 0.3),
        "doubleword" => (0.5, 2.0),
        "perplexity" => (1.0, 1.0),
        provider if LOCAL.contains(&provider) => (0.0, 0.0),
        _ => (5.0, 15.0),
    }
}

/// Dollars per generated image, rounded up from list prices.
const IMAGE_PRICE: f64 = 0.07;

/// Whether a request path generates images, which report no token usage.
pub(crate) fn is_image_call(path: &str) -> bool {
    path.contains("/images/")
        || path.contains("/image/generate")
        || (path.contains(":generateContent") && path.contains("image"))
}

#[derive(Deserialize)]
struct Interaction {
    when: Request,
    then: Response,
}

#[derive(Deserialize)]
struct Request {
    #[serde(default)]
    path: String,
    #[serde(default)]
    body: Option<String>,
}

#[derive(Deserialize)]
struct Response {
    #[serde(default)]
    body: Option<String>,
}

/// Input and output tokens a reply reports, from the last frame that
/// carries usage.
pub(crate) fn usage(body: &str) -> Option<(u64, u64)> {
    let trimmed = body.trim_start();
    let frames: Vec<Value> = if trimmed.starts_with('{') {
        serde_json::from_str(trimmed).into_iter().collect()
    } else {
        body.lines()
            .filter_map(|line| line.strip_prefix("data:"))
            .filter_map(|data| serde_json::from_str(data.trim()).ok())
            .collect()
    };
    let count = |value: &Value, key: &str| value.get(key).and_then(Value::as_u64);
    frames.iter().rev().find_map(|frame| {
        if let (Some(input), Some(output)) = (
            count(frame, "prompt_eval_count"),
            count(frame, "eval_count"),
        ) {
            return Some((input, output));
        }
        let usage = frame
            .get("usage")
            .or_else(|| frame.get("usageMetadata"))
            .or_else(|| {
                frame
                    .get("response")
                    .and_then(|response| response.get("usage"))
            })
            .or_else(|| {
                frame
                    .get("message")
                    .and_then(|message| message.get("usage"))
            })?;
        if let Some(input) = count(usage, "prompt_tokens") {
            return Some((input, count(usage, "completion_tokens").unwrap_or(0)));
        }
        if let Some(input) = count(usage, "input_tokens") {
            // Anthropic reports cache reads and writes beside `input_tokens`;
            // a write costs 1.25 times an input token, a read less than one.
            let cache_read = count(usage, "cache_read_input_tokens").unwrap_or(0);
            let cache_write = count(usage, "cache_creation_input_tokens").unwrap_or(0);
            return Some((
                input + cache_read + (cache_write * 5).div_ceil(4),
                count(usage, "output_tokens").unwrap_or(0),
            ));
        }
        count(usage, "promptTokenCount").map(|input| {
            (
                input,
                count(usage, "candidatesTokenCount").unwrap_or(0)
                    + count(usage, "thoughtsTokenCount").unwrap_or(0),
            )
        })
    })
}

/// What recording `contents` (a cassette) cost on `provider`, and how many of
/// its replies reported no usage.
pub(crate) fn cassette_cost(provider: &str, contents: &str) -> (f64, usize) {
    let interactions: Vec<Interaction> = serde_yaml::Deserializer::from_str(contents)
        .filter_map(|document| Interaction::deserialize(document).ok())
        .collect();
    let (input_price, output_price) = prices(provider);
    let (mut input, mut output, mut images, mut unknown) = (0_u64, 0_u64, 0_u64, 0_usize);
    for interaction in &interactions {
        if is_image_call(&interaction.when.path) {
            images += 1;
            continue;
        }
        match usage(interaction.then.body.as_deref().unwrap_or_default()) {
            Some((i, o)) => {
                input += i;
                output += o;
            }
            None => {
                unknown += 1;
                let request = interaction.when.body.as_deref().unwrap_or_default();
                input += request.len() as u64 / 3;
                output += serde_json::from_str::<Value>(request)
                    .ok()
                    .and_then(|body| {
                        ["max_tokens", "max_output_tokens", "max_completion_tokens"]
                            .iter()
                            .find_map(|key| body.get(*key).and_then(Value::as_u64))
                    })
                    .unwrap_or(1024);
            }
        }
    }
    let local = LOCAL.contains(&provider);
    let cost = input as f64 / 1e6 * input_price
        + output as f64 / 1e6 * output_price
        + if local {
            0.0
        } else {
            images as f64 * IMAGE_PRICE
        };
    (cost, unknown)
}

/// Spend per provider from the ledger text, pricing each attempt's fixtures
/// from `fixture_root`.
pub(crate) fn spend(ledger: &str, fixture_root: &Path) -> (BTreeMap<String, f64>, usize) {
    // One entry per attempt: its result row, or its `started` row alone.
    let mut attempts: Vec<((String, String), [String; 3])> = Vec::new();
    for line in ledger.lines().skip(1) {
        let columns: Vec<&str> = line.split('\t').collect();
        let [provider, fixtures, test, attempt, exit, _note] = columns.as_slice() else {
            continue;
        };
        if *exit == "skipped" {
            continue;
        }
        let key = ((*test).to_owned(), (*attempt).to_owned());
        let exit = if *exit == super::record::STARTED {
            "1"
        } else {
            exit
        };
        let row = [
            (*provider).to_owned(),
            (*fixtures).to_owned(),
            exit.to_owned(),
        ];
        match attempts.iter_mut().find(|(seen, _)| *seen == key) {
            Some((_, existing)) => *existing = row,
            None => attempts.push((key, row)),
        }
    }
    let mut per_provider = BTreeMap::<String, f64>::new();
    let mut unknown_total = 0;
    for (_, [provider, fixtures, exit]) in &attempts {
        let provider = provider.as_str();
        let mut cost = 0.0;
        for fixture in fixtures.split(';').filter(|fixture| !fixture.is_empty()) {
            if let Ok(contents) = std::fs::read_to_string(fixture_root.join(fixture)) {
                let (fixture_cost, unknown) = cassette_cost(provider, &contents);
                cost += fixture_cost;
                unknown_total += unknown;
            }
        }
        if exit != "0" && !LOCAL.contains(&provider) {
            cost = cost.max(FAILED_ATTEMPT_FLOOR);
        }
        *per_provider.entry(provider.to_owned()).or_default() += cost;
    }
    (per_provider, unknown_total)
}

/// A dollar amount given to `flag`.
fn dollars(value: Option<&String>, flag: &str) -> Result<f64, String> {
    value
        .and_then(|value| value.parse::<f64>().ok())
        .filter(|value| value.is_finite() && *value >= 0.0)
        .ok_or_else(|| format!("{flag} needs a dollar amount"))
}

pub(crate) fn run(root: &Path, args: &[String]) -> Result<(), String> {
    let mut per_wire_cap = None;
    let mut total_cap = None;
    let mut args = args.iter();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--cap-per-wire" => per_wire_cap = Some(dollars(args.next(), "--cap-per-wire")?),
            "--cap-total" => total_cap = Some(dollars(args.next(), "--cap-total")?),
            other => return Err(format!("unknown argument {other}")),
        }
    }
    let ledger_path = super::attempt_root(root).join("recordings.tsv");
    let ledger = std::fs::read_to_string(&ledger_path)
        .map_err(|error| format!("{}: {error}", ledger_path.display()))?;
    let (per_provider, unknown) = spend(
        &ledger,
        &root.join("crates/rig-cassette/fixtures/cassettes"),
    );
    let total: f64 = per_provider.values().sum();
    for (provider, cost) in &per_provider {
        println!("{provider}\t${cost:.4}");
    }
    println!("total\t${total:.4}\t({unknown} replies without usage, priced at their bound)");
    let over_wire = per_wire_cap
        .map(|cap| per_provider.iter().filter(|(_, cost)| **cost > cap).count())
        .unwrap_or(0);
    let over_total = total_cap.is_some_and(|cap| total > cap);
    if over_wire > 0 || over_total {
        return Err(format!(
            "over budget: {over_wire} wire(s) above the per-wire cap, total cap exceeded: {over_total}"
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests;
