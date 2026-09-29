//! What prompt caching cost and saved, priced the same way on every provider.

use std::iter::Sum;
use std::ops::{Add, AddAssign};

use serde::{Deserialize, Serialize};

use super::Usage;

/// Prices of one model and tier, in USD per 1M tokens (storage: per 1M
/// token-hours). The caller supplies them from the provider's price list;
/// rig has none built in. A provider without a separate write price bills
/// writes at `input`; one without storage leaves `storage_per_hour` at zero.
///
/// ```no_run
/// use rig_core::completion::CacheRates;
///
/// // A 5-minute cache write at 1.25 times the input price, reads at a tenth.
/// let rates = CacheRates { input: 3.0, cached_read: 0.3, cache_write: 3.75, storage_per_hour: 0.0 };
/// # let _ = rates;
/// ```
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
///
/// A run priced the same way on any provider, with the caller's rates from
/// the provider's price list:
///
/// ```no_run
/// use rig_core::completion::{CacheCost, CacheRates, Usage};
/// use rig_core::providers::gemini::{AutoCache, CacheBook};
///
/// # let calls: Vec<Usage> = Vec::new();
/// # let book = CacheBook::new(AutoCache::default());
/// let run: CacheCost = calls.iter().map(CacheCost::from_usage).sum();
///
/// let opus = CacheRates { input: 4.0, cached_read: 0.2, cache_write: 5.0, storage_per_hour: 0.0 };
/// println!("${:.3}, {:.1}% saved", run.usd(&opus), run.saving(&opus) * 100.0);
///
/// // Gemini: the calls, plus the cache book's creation and storage.
/// let gemini = run + CacheCost::from(&book.report());
/// let flash = CacheRates { input: 0.75, cached_read: 0.075, cache_write: 0.75, storage_per_hour: 0.5 };
/// println!("${:.3}", gemini.usd(&flash));
/// ```
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
            uncached_input: input.saturating_sub(cache_reads + cache_writes),
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
