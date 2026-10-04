//! The selection: greedy set cover with fixed tie-breaks, then a pass that
//! drops every chosen test the others make redundant.
//!
//! Elements are what the gate measures and no always-kept test holds. A
//! candidate covers its elements, costs the bytes of the fixtures it owns,
//! and is identified by its position in name order. Two constraints hold
//! throughout: a kept fixture (protected, or named by a kept test) keeps at
//! least one of its owners, and a forced candidate is never dropped.

#[cfg(test)]
mod tests;

use std::collections::{BTreeMap, BTreeSet};

/// One candidate test.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct Candidate {
    /// Element ids, sorted and distinct.
    pub(crate) elements: Vec<usize>,
    /// The bytes of the fixtures it owns.
    pub(crate) cost: u64,
    /// Fixture ids it owns.
    pub(crate) owns: Vec<usize>,
    /// Fixture ids it reads without owning.
    pub(crate) reads: Vec<usize>,
}

/// A selection problem. Candidates are in name order, so a lower index is
/// the alphabetically earlier test.
#[derive(Clone, Debug, Default)]
pub(crate) struct Problem {
    pub(crate) candidates: Vec<Candidate>,
    pub(crate) elements: usize,
    pub(crate) fixtures: usize,
    /// Fixtures kept whatever is chosen.
    pub(crate) protected: BTreeSet<usize>,
    /// Candidates kept whatever is chosen.
    pub(crate) forced: BTreeSet<usize>,
}

/// The chosen candidates, in the order they were taken.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct Selection {
    pub(crate) kept: Vec<usize>,
}

static NONE: Candidate = Candidate {
    elements: Vec::new(),
    cost: 0,
    owns: Vec::new(),
    reads: Vec::new(),
};

fn count(counts: &[u32], at: usize) -> u32 {
    counts.get(at).copied().unwrap_or(0)
}

fn step(counts: &mut [u32], at: usize, up: bool) {
    if let Some(count) = counts.get_mut(at) {
        *count = if up {
            count.saturating_add(1)
        } else {
            count.saturating_sub(1)
        };
    }
}

struct State<'a> {
    problem: &'a Problem,
    kept: Vec<bool>,
    order: Vec<usize>,
    /// How many kept candidates cover each element.
    cover: Vec<u32>,
    /// How many kept candidates name each fixture.
    named: Vec<u32>,
    /// How many kept candidates own each fixture.
    owned: Vec<u32>,
    /// The candidates owning each fixture, in index order.
    owners: Vec<Vec<usize>>,
}

impl<'a> State<'a> {
    fn new(problem: &'a Problem) -> Self {
        let mut owners = vec![Vec::new(); problem.fixtures];
        for (index, candidate) in problem.candidates.iter().enumerate() {
            for fixture in &candidate.owns {
                if let Some(list) = owners.get_mut(*fixture) {
                    list.push(index);
                }
            }
        }
        Self {
            problem,
            kept: vec![false; problem.candidates.len()],
            order: Vec::new(),
            cover: vec![0; problem.elements],
            named: vec![0; problem.fixtures],
            owned: vec![0; problem.fixtures],
            owners,
        }
    }

    fn candidate(&self, index: usize) -> &'a Candidate {
        self.problem.candidates.get(index).unwrap_or(&NONE)
    }

    fn is_kept(&self, index: usize) -> bool {
        self.kept.get(index).copied().unwrap_or(false)
    }

    fn owners(&self, fixture: usize) -> &[usize] {
        self.owners.get(fixture).map_or(&[], Vec::as_slice)
    }

    fn set(&mut self, index: usize, kept: bool) {
        if self.is_kept(index) == kept {
            return;
        }
        if let Some(slot) = self.kept.get_mut(index) {
            *slot = kept;
        }
        if kept {
            self.order.push(index);
        } else {
            self.order.retain(|other| *other != index);
        }
        let candidate = self.candidate(index);
        for element in &candidate.elements {
            step(&mut self.cover, *element, kept);
        }
        for fixture in candidate.owns.iter().chain(&candidate.reads) {
            step(&mut self.named, *fixture, kept);
        }
        for fixture in &candidate.owns {
            step(&mut self.owned, *fixture, kept);
        }
    }

    fn fixture_kept(&self, fixture: usize) -> bool {
        self.problem.protected.contains(&fixture) || count(&self.named, fixture) > 0
    }

    /// A kept fixture with owners but none of them kept.
    fn orphaned(&self, fixture: usize) -> bool {
        self.fixture_kept(fixture)
            && count(&self.owned, fixture) == 0
            && !self.owners(fixture).is_empty()
    }

    /// Keep the cheapest owner of every kept fixture that has none kept.
    fn close(&mut self) {
        while let Some(fixture) = (0..self.problem.fixtures).find(|fixture| self.orphaned(*fixture))
        {
            let cheapest = self
                .owners(fixture)
                .iter()
                .copied()
                .min_by_key(|owner| (self.candidate(*owner).cost, *owner));
            match cheapest {
                Some(owner) => self.set(owner, true),
                None => return,
            }
        }
    }

    /// Whether `index` can go: every element it covers stays covered, and
    /// no kept fixture loses its last kept owner.
    fn redundant(&self, index: usize) -> bool {
        if self.problem.forced.contains(&index) {
            return false;
        }
        let candidate = self.candidate(index);
        if candidate
            .elements
            .iter()
            .any(|element| count(&self.cover, *element) <= 1)
        {
            return false;
        }
        // Only a fixture it owns can lose its last owner while it stays kept.
        !candidate.owns.iter().any(|fixture| {
            let still_named = count(&self.named, *fixture) > 1;
            let kept = self.problem.protected.contains(fixture) || still_named;
            kept && count(&self.owned, *fixture) <= 1
        })
    }
}

/// Choose the kept candidates. Forced candidates and the owners kept
/// fixtures need come first; then, while an element is uncovered, the
/// candidate covering the most uncovered elements, then the cheaper, then
/// the earlier; then every chosen candidate the rest make redundant goes,
/// latest first, until none does.
pub(crate) fn select(problem: &Problem) -> Selection {
    let mut state = State::new(problem);
    for index in &problem.forced {
        state.set(*index, true);
    }
    state.close();
    loop {
        let best = (0..problem.candidates.len())
            .filter(|index| !state.is_kept(*index))
            .map(|index| {
                let gain = state
                    .candidate(index)
                    .elements
                    .iter()
                    .filter(|element| count(&state.cover, **element) == 0)
                    .count();
                (gain, index)
            })
            .filter(|(gain, _)| *gain > 0)
            .max_by(|(gain_a, a), (gain_b, b)| {
                gain_a
                    .cmp(gain_b)
                    .then_with(|| state.candidate(*b).cost.cmp(&state.candidate(*a).cost))
                    .then_with(|| b.cmp(a))
            });
        let Some((_, index)) = best else {
            break;
        };
        state.set(index, true);
        state.close();
    }
    loop {
        let order = state.order.clone();
        let mut dropped = false;
        for index in order.into_iter().rev() {
            if state.redundant(index) {
                state.set(index, false);
                dropped = true;
            }
        }
        if !dropped {
            break;
        }
    }
    Selection { kept: state.order }
}

/// The fewest of `kept` that cover `elements`, most covered first, then
/// the earlier; elements none of them covers are left out.
pub(crate) fn cover_with(elements: &[usize], kept: &[usize], problem: &Problem) -> Vec<usize> {
    let mut by_element: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    let wanted: BTreeSet<usize> = elements.iter().copied().collect();
    for index in kept {
        let candidate = problem.candidates.get(*index).unwrap_or(&NONE);
        for element in &candidate.elements {
            if wanted.contains(element) {
                by_element.entry(*element).or_default().push(*index);
            }
        }
    }
    let mut uncovered: BTreeSet<usize> = by_element.keys().copied().collect();
    let mut chosen = Vec::new();
    while !uncovered.is_empty() {
        let mut gains: BTreeMap<usize, usize> = BTreeMap::new();
        for element in &uncovered {
            for index in by_element.get(element).into_iter().flatten() {
                *gains.entry(*index).or_default() += 1;
            }
        }
        let Some((index, _)) = gains
            .into_iter()
            .max_by(|(a, gain_a), (b, gain_b)| gain_a.cmp(gain_b).then_with(|| b.cmp(a)))
        else {
            break;
        };
        let candidate = problem.candidates.get(index).unwrap_or(&NONE);
        uncovered.retain(|element| candidate.elements.binary_search(element).is_err());
        chosen.push(index);
    }
    chosen.sort_unstable();
    chosen
}
