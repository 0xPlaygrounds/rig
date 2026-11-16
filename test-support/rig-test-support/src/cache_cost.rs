//! What prompt caching cost and saved, priced the same way on every provider.

use std::iter::Sum;
use std::ops::{Add, AddAssign};

use serde::{Deserialize, Serialize};

use rig_core::completion::Usage;
use rig_core::providers::gemini::CacheReport;

/// Prices of one model and tier, in USD per 1M tokens (storage: per 1M
/// token-hours). The built-in catalog's [`Pricing`](rig_core::catalog::Pricing)
/// holds the input, cached-read and cache-write prices of the models it
/// lists, but no storage price. A provider without a separate write price
/// bills writes at `input`; one without storage leaves `storage_per_hour`
/// at zero.
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
    /// The billed input of one call: the input neither read from nor written
    /// to a cache is `input_tokens - cached_input_tokens -
    /// cache_creation_input_tokens`. An unreported counter counts as zero;
    /// reads and writes above `input_tokens` leave no uncached input rather
    /// than a negative one.
    pub fn from_usage(usage: &Usage) -> Self {
        let input = usage.input_tokens.unwrap_or(0);
        let cache_reads = usage.cached_input_tokens.unwrap_or(0);
        let cache_writes = usage.cache_creation_input_tokens.unwrap_or(0);
        Self {
            uncached_input: input.saturating_sub(cache_reads.saturating_add(cache_writes)),
            cache_reads,
            cache_writes,
            storage_token_hours: 0.0,
        }
    }

    /// Every input token, however it was billed.
    pub fn prompt_tokens(&self) -> u64 {
        self.uncached_input
            .saturating_add(self.cache_reads)
            .saturating_add(self.cache_writes)
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

/// A Gemini cache book's caches as billed beside the calls: every created
/// token as a cache write (priced at the input price) and `token_hours` of
/// storage. Add it to the calls' [`CacheCost::from_usage`] to price a run.
impl From<&CacheReport> for CacheCost {
    fn from(report: &CacheReport) -> Self {
        Self {
            cache_writes: report.created.iter().map(|created| created.tokens).sum(),
            storage_token_hours: report.token_hours,
            ..Self::default()
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
        self.uncached_input = self.uncached_input.saturating_add(other.uncached_input);
        self.cache_reads = self.cache_reads.saturating_add(other.cache_reads);
        self.cache_writes = self.cache_writes.saturating_add(other.cache_writes);
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
