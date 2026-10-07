//! What a Chat reply reports beside its message: the sources its answer
//! cites and what the turn cost.
//!
//! Message `annotations` (`url_citation`), Mistral `reference` chunks and
//! the top-level URL lists of Perplexity, xAI live search, Venice and Z.AI
//! become citations of a whole text block: none of them states its offsets
//! in a documented unit with the cited text beside them. A cost is read in
//! USD from wherever the dialect reports it.

use serde_json::{Map, Value};

use crate::completion::Cost;
use crate::json_utils::Lenient;
use crate::message::{Source, SourceLocation};
use crate::wire::WireCitation;

/// Top-level reply fields read at the end of a reply, each as last stated:
/// a stream may restate them on every chunk.
pub(super) const REPLY_FIELDS: [&str; 5] = [
    "citations",
    "search_results",
    "web_search",
    "venice_parameters",
    "cost",
];

/// xAI's cost unit: 10^10 ticks per USD.
const TICKS_PER_USD: f64 = 1e10;

/// A source at `url`, titled when `title` names one, quoting `quoted` when
/// that is not empty.
fn url_source(url: &str, title: Option<&str>, quoted: Option<&str>) -> Source {
    let mut source = Source::new(SourceLocation::Url {
        url: url.to_owned(),
    });
    if let Some(title) = title.filter(|title| !title.is_empty()) {
        source = source.title(title);
    }
    if let Some(quoted) = quoted.filter(|quoted| !quoted.is_empty()) {
        source = source.cited_text(quoted);
    }
    source
}

/// One message annotation as a citation of the whole block: a
/// `url_citation` (OpenAI, OpenRouter's web plugin, MiMo). Its
/// `start_index` and `end_index` stay in the reply, since their unit is
/// undocumented and no cited text checks them.
pub(super) fn annotation(annotation: &Value) -> Option<WireCitation> {
    if annotation.str("type") != Some("url_citation") {
        return None;
    }
    let cited = annotation.get("url_citation").unwrap_or(annotation);
    let url = cited.str("url").filter(|url| !url.is_empty())?;
    let source = url_source(url, cited.str("title"), cited.str("content"));
    Some(WireCitation::new(None, vec![source]))
}

/// A Mistral `reference` chunk as a citation of the text before it: each
/// of its `reference_ids` names a reference a tool result supplied.
pub(super) fn reference(part: &Value) -> Option<WireCitation> {
    if part.str("type") != Some("reference") {
        return None;
    }
    let sources: Vec<Source> = part
        .arr("reference_ids")
        .iter()
        .map(|id| match id {
            Value::String(id) => id.clone(),
            id => id.to_string(),
        })
        .map(|id| {
            Source::new(SourceLocation::Document {
                index: None,
                id: Some(id),
                within: None,
            })
        })
        .collect();
    (!sources.is_empty()).then(|| WireCitation::new(None, sources))
}

/// The URL lists a reply states beside its message, one citation of the
/// whole block per URL: Perplexity's and xAI live search's `citations`
/// (titled from Perplexity's `search_results`), Venice's
/// `venice_parameters.web_search_citations` and Z.AI's `web_search`.
pub(super) fn listed(fields: &Map<String, Value>) -> Vec<WireCitation> {
    let field = |key: &str| fields.get(key).and_then(Value::as_array);
    let results = field("search_results").map_or(&[][..], Vec::as_slice);
    let title = |url: &str| {
        results
            .iter()
            .find(|result| result.str("url") == Some(url))
            .and_then(|result| result.str("title"))
    };
    let urls = field("citations")
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .map(|url| url_source(url, title(url), None));
    let venice = fields
        .get("venice_parameters")
        .map_or(&[][..], |parameters| parameters.arr("web_search_citations"))
        .iter()
        .filter_map(|result| {
            let url = result.str("url")?;
            Some(url_source(url, result.str("title"), result.str("content")))
        });
    let zai = field("web_search")
        .into_iter()
        .flatten()
        .filter_map(|result| {
            let url = result.str("link")?;
            Some(url_source(url, result.str("title"), result.str("content")))
        });
    urls.chain(venice)
        .chain(zai)
        .filter(|source| !matches!(&source.location, SourceLocation::Url { url } if url.is_empty()))
        .map(|source| WireCitation::new(None, vec![source]))
        .collect()
}

/// The cost a reply reports, in USD: OpenRouter's `usage.cost`,
/// Perplexity's `usage.cost` parts, xAI's `usage.cost_in_usd_ticks`,
/// Venice's top-level `cost.usd`, or a gateway's top-level `cost` as a
/// number or a numeric string. `None` when it reports none.
pub(super) fn cost(usage: Option<&Value>, fields: &Map<String, Value>) -> Option<Cost> {
    let finite = |value: f64| value.is_finite().then_some(value);
    let reported = usage.and_then(|usage| match usage.get("cost") {
        Some(parts @ Value::Object(_)) => {
            let part = |key: &str| parts.f64(key).and_then(finite);
            let (input, output) = (part("input_tokens_cost"), part("output_tokens_cost"));
            let total = part("total_cost")
                .or_else(|| Some(input? + output? + part("request_cost").unwrap_or(0.0)))?;
            Some(Cost::from_total(total).input(input).output(output))
        }
        Some(_) => usage.f64("cost").and_then(finite).map(Cost::from_total),
        None => None,
    });
    let ticks = || {
        usage
            .and_then(|usage| usage.f64("cost_in_usd_ticks"))
            .and_then(finite)
            .map(|ticks| Cost::from_total(ticks / TICKS_PER_USD))
    };
    let top = || match fields.get("cost") {
        Some(cost @ Value::Object(_)) => cost.f64("usd"),
        Some(Value::Number(cost)) => cost.as_f64(),
        Some(Value::String(cost)) => cost.trim().parse().ok(),
        _ => None,
    };
    reported
        .or_else(ticks)
        .or_else(|| top().and_then(finite).map(Cost::from_total))
}

#[cfg(test)]
mod tests;
