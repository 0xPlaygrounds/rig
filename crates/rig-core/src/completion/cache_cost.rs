//! What prompt caching cost and saved, priced the same way on every provider.

use std::iter::Sum;
use std::ops::{Add, AddAssign};

use serde::{Deserialize, Serialize};

use super::Usage;

/// How a provider counts cache reads and writes against
/// [`Usage::input_tokens`].
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum CacheAccounting {
    /// Cache reads and writes are reported beside `input_tokens`, so the
    /// prompt is the sum of all three. Anthropic reports this way.
    Alongside,
    /// Cache reads and writes are part of `input_tokens`, so the prompt is
    /// `input_tokens` alone. OpenAI, Gemini and the OpenAI-compatible
    /// providers report this way.
    Subset,
}

/// Prices of one model and tier, in USD per 1M tokens. The caller supplies
/// them from the provider's price list; rig has none built in.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct CacheRates {
    /// Uncached input.
    pub input: f64,
    /// Input read from a cache.
    pub cached_read: f64,
    /// Input written to a cache, or put into a cache resource.
    pub cache_write: f64,
    /// Storing one token for one hour, for providers that bill cache storage.
    pub storage_per_hour: f64,
}

/// A request's or a run's input tokens by how they are billed, plus the
/// token-hours its cache resources were stored. Sum the costs of a run's
/// calls (and of any cache resources beside them) and price the total with
/// [`CacheCost::usd`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct CacheCost {
    /// Input billed at the uncached price.
    pub uncached_input: u64,
    /// Input read from a cache.
    pub cache_reads: u64,
    /// Input written to a cache.
    pub cache_writes: u64,
    /// Σ tokens × hours of stored cache resources.
    pub storage_token_hours: f64,
}

impl CacheCost {
    /// The billed input of one call. An unreported counter counts as zero;
    /// under [`CacheAccounting::Subset`] reads and writes above
    /// `input_tokens` leave no uncached input rather than a negative one.
    pub fn from_usage(usage: &Usage, accounting: CacheAccounting) -> Self {
        let input = usage.input_tokens.unwrap_or(0);
        let cache_reads = usage.cached_input_tokens.unwrap_or(0);
        let cache_writes = usage.cache_creation_input_tokens.unwrap_or(0);
        let uncached_input = match accounting {
            CacheAccounting::Alongside => input,
            CacheAccounting::Subset => input.saturating_sub(cache_reads + cache_writes),
        };
        Self {
            uncached_input,
            cache_reads,
            cache_writes,
            storage_token_hours: 0.0,
        }
    }

    /// Every input token, however it was billed.
    pub fn prompt_tokens(&self) -> u64 {
        self.uncached_input + self.cache_reads + self.cache_writes
    }

    /// Input cost in USD at `rates`: uncached, read, written and stored.
    pub fn usd(&self, rates: &CacheRates) -> f64 {
        (self.uncached_input as f64 * rates.input
            + self.cache_reads as f64 * rates.cached_read
            + self.cache_writes as f64 * rates.cache_write
            + self.storage_token_hours * rates.storage_per_hour)
            / 1e6
    }

    /// Input cost in USD at `rates` had every prompt token been uncached.
    pub fn uncached_usd(&self, rates: &CacheRates) -> f64 {
        self.prompt_tokens() as f64 * rates.input / 1e6
    }

    /// The share of the uncached input cost caching saved: `1 - usd /
    /// uncached_usd`. Negative when caching cost more; zero without input.
    pub fn saving(&self, rates: &CacheRates) -> f64 {
        let uncached = self.uncached_usd(rates);
        if uncached > 0.0 {
            1.0 - self.usd(rates) / uncached
        } else {
            0.0
        }
    }
}

impl Add for CacheCost {
    type Output = Self;

    fn add(mut self, other: Self) -> Self {
        self += other;
        self
    }
}

impl AddAssign for CacheCost {
    fn add_assign(&mut self, other: Self) {
        self.uncached_input += other.uncached_input;
        self.cache_reads += other.cache_reads;
        self.cache_writes += other.cache_writes;
        self.storage_token_hours += other.storage_token_hours;
    }
}

impl Sum for CacheCost {
    fn sum<I: Iterator<Item = Self>>(costs: I) -> Self {
        costs.fold(Self::default(), Add::add)
    }
}

#[cfg(test)]
mod tests;
