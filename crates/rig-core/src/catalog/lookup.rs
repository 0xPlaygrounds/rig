//! The one rule every catalog lookup follows, what a lookup returns, and the
//! suggestions a miss carries.

use std::fmt;

use super::ModelSpec;

/// How a requested model id matched its catalog entry.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Matched {
    /// The id is listed as given.
    Exact,
    /// The id is a dated snapshot of the listed id this holds: that id
    /// followed by `-20` and a year (`claude-sonnet-4-5-20250929`,
    /// `gpt-5-2025-08-07`, `claude-opus-5-5-20260601-v1:0`).
    SnapshotOf(String),
}

/// A model the catalog found, and how the requested id matched it.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq)]
pub struct Resolved<'a> {
    /// The catalog entry.
    pub spec: &'a ModelSpec,
    /// How the requested id matched [`Self::spec`].
    pub matched: Matched,
}

/// A reference the catalog does not list, with up to five listed models
/// whose references are close to it.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NotFound {
    /// The reference as given.
    pub reference: String,
    /// Close references to models that are not deprecated, closest first.
    pub suggestions: Vec<String>,
}

impl fmt::Display for NotFound {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "the catalog lists no model `{}`", self.reference)?;
        match self.suggestions.as_slice() {
            [] => Ok(()),
            [only] => write!(f, "; did you mean `{only}`?"),
            many => write!(f, "; did you mean one of `{}`?", many.join("`, `")),
        }
    }
}

impl std::error::Error for NotFound {}

/// The most suggestions a [`NotFound`] carries.
const SUGGESTIONS: usize = 5;

/// The listed ids `model` may be a dated snapshot of, longest first: each
/// prefix of `model` that is followed by `-20` and two more digits.
pub(super) fn snapshot_bases(model: &str) -> impl Iterator<Item = &str> {
    let mut bases: Vec<&str> = model
        .match_indices("-20")
        .filter(|(at, _)| {
            model
                .get(at + 3..at + 5)
                .is_some_and(|year| year.bytes().all(|byte| byte.is_ascii_digit()))
        })
        .filter_map(|(at, _)| model.get(..at))
        .filter(|base| !base.is_empty())
        .collect();
    bases.reverse();
    bases.into_iter()
}

/// Up to [`SUGGESTIONS`] references from `candidates` close to `reference`,
/// closest first. A candidate is close when it contains the reference's
/// model id or its edit distance is at most a third of the reference's
/// length.
pub(super) fn suggestions<'a>(
    reference: &str,
    candidates: impl Iterator<Item = &'a ModelSpec>,
) -> Vec<String> {
    let wanted = reference.to_lowercase();
    let model = wanted
        .split_once('/')
        .map_or(wanted.as_str(), |(_, model)| model);
    let limit = wanted.chars().count() / 3 + 1;
    let mut close: Vec<(usize, String)> = candidates
        .filter(|spec| !spec.deprecated)
        .filter_map(|spec| {
            let candidate = format!("{}/{}", spec.provider.vendor(), spec.id);
            let distance = edit_distance(&wanted, &candidate.to_lowercase());
            let contains = !model.is_empty() && spec.id.to_lowercase().contains(model);
            (contains || distance <= limit).then_some((distance, candidate))
        })
        .collect();
    close.sort();
    close.truncate(SUGGESTIONS);
    close.into_iter().map(|(_, candidate)| candidate).collect()
}

/// The edit distance between `a` and `b`, by character: insertions,
/// deletions, substitutions and swaps of two adjacent characters each count
/// one (optimal string alignment), so `inptu` is one edit from `input`.
pub(crate) fn edit_distance(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let at = |row: &[usize], k: usize| row.get(k).copied().unwrap_or(usize::MAX);
    let mut before: Vec<usize> = Vec::new();
    let mut previous: Vec<usize> = (0..=b.len()).collect();
    for (i, left) in a.iter().enumerate() {
        let mut current = Vec::with_capacity(b.len() + 1);
        current.push(i + 1);
        for (j, right) in b.iter().enumerate() {
            let substitute = at(&previous, j).saturating_add(usize::from(left != right));
            let delete = at(&previous, j + 1).saturating_add(1);
            let insert = at(&current, j).saturating_add(1);
            let mut best = substitute.min(delete).min(insert);
            let swapped = i > 0
                && j > 0
                && a.get(i - 1) == Some(right)
                && b.get(j - 1) == Some(left)
                && left != right;
            if swapped {
                best = best.min(at(&before, j - 1).saturating_add(1));
            }
            current.push(best);
        }
        before = std::mem::replace(&mut previous, current);
    }
    previous.last().copied().unwrap_or(0)
}
