//! When a job runs: `every 30m`, `in 1h` (once), or five cron fields
//! (`minute hour day-of-month month day-of-week`) in the local time zone.
//! Times are milliseconds since the Unix epoch.

use chrono::{Datelike, Days, Local, NaiveDateTime, TimeZone, Timelike};

/// How far ahead a cron schedule is searched: eight years covers any day a
/// valid schedule names, such as February 29 on a Monday.
const SEARCH_DAYS: u64 = 366 * 8;

/// A parsed schedule.
pub(crate) enum Schedule {
    /// Every this many milliseconds, from when it was added.
    Every(u64),
    /// Once, this many milliseconds after it was added.
    Once(u64),
    /// At the minutes the cron fields match.
    Cron(Cron),
}

impl Schedule {
    pub(crate) fn parse(text: &str) -> Result<Self, String> {
        let text = text.trim();
        if let Some(span) = text.strip_prefix("every ") {
            return milliseconds(span).map(Self::Every);
        }
        if let Some(span) = text.strip_prefix("in ") {
            return milliseconds(span).map(Self::Once);
        }
        Cron::parse(text).map(Self::Cron)
    }

    /// The next tick after `now`: the first one for a job just added
    /// (`last` is `None`), else the first after its last tick `last`.
    /// `None` when it never runs again.
    pub(crate) fn next(&self, last: Option<u64>, now: u64) -> Option<u64> {
        match (self, last) {
            (Self::Every(every) | Self::Once(every), None) => now.checked_add(*every),
            (Self::Every(every), Some(last)) => {
                let behind = now.saturating_sub(last) / every;
                last.checked_add(every.checked_mul(behind.checked_add(1)?)?)
            }
            (Self::Once(_), Some(_)) => None,
            (Self::Cron(cron), _) => cron.after(now),
        }
    }
}

/// `30m`, `2 h`, `45s`, `1d`: a positive span.
fn milliseconds(span: &str) -> Result<u64, String> {
    let span = span.trim();
    let digits = span
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(span.len());
    let (count, unit) = span.split_at(digits);
    let count: u64 = count
        .parse()
        .map_err(|_| format!("`{span}` is not a span such as 30m"))?;
    let unit = match unit.trim() {
        "s" | "sec" | "secs" | "second" | "seconds" => 1_000,
        "m" | "min" | "mins" | "minute" | "minutes" => 60_000,
        "h" | "hour" | "hours" => 3_600_000,
        "d" | "day" | "days" => 86_400_000,
        other => return Err(format!("`{other}` is not a unit: use s, m, h or d")),
    };
    count
        .checked_mul(unit)
        .filter(|span| *span > 0)
        .ok_or_else(|| format!("`{span}` must be a positive span"))
}

/// Five cron fields, each a bit set of the values it matches.
pub(crate) struct Cron {
    minutes: u64,
    hours: u64,
    days: u64,
    months: u64,
    weekdays: u64,
    /// Whether day-of-month and day-of-week were both restricted: a day
    /// then matches either, as cron has it.
    either_day: bool,
}

impl Cron {
    fn parse(text: &str) -> Result<Self, String> {
        let fields: Vec<&str> = text.split_whitespace().collect();
        let [minute, hour, day, month, weekday] = fields.as_slice() else {
            return Err(format!(
                "`{text}` is not a schedule: use `every 30m`, `in 1h`, or five cron fields \
                 such as `0 9 * * 1-5`"
            ));
        };
        let weekdays = field(weekday, 0, 7)?;
        Ok(Self {
            minutes: field(minute, 0, 59)?,
            hours: field(hour, 0, 23)?,
            days: field(day, 1, 31)?,
            months: field(month, 1, 12)?,
            // Day 7 is Sunday too.
            weekdays: (weekdays | (weekdays >> 7)) & 0x7f,
            either_day: *day != "*" && *weekday != "*",
        })
    }

    /// The first minute after `now` that the fields match, in local time.
    fn after(&self, now: u64) -> Option<u64> {
        let now = Local
            .timestamp_millis_opt(i64::try_from(now).ok()?)
            .single()?;
        let start = now.naive_local().with_second(0)?.with_nanosecond(0)?;
        let start = start.checked_add_signed(chrono::TimeDelta::minutes(1))?;
        for offset in 0..SEARCH_DAYS {
            let date = start.date().checked_add_days(Days::new(offset))?;
            let (day, weekday) = (date.day(), date.weekday().num_days_from_sunday());
            let by_day = has(self.days, day);
            let by_weekday = has(self.weekdays, weekday);
            let day_matches = match self.either_day {
                true => by_day || by_weekday,
                false => by_day && by_weekday,
            };
            if !has(self.months, date.month()) || !day_matches {
                continue;
            }
            for hour in (0..24).filter(|hour| has(self.hours, *hour)) {
                for minute in (0..60).filter(|minute| has(self.minutes, *minute)) {
                    let Some(at) = date.and_hms_opt(hour, minute, 0) else {
                        continue;
                    };
                    if let Some(at) = local(at).filter(|_| at >= start) {
                        return Some(at);
                    }
                }
            }
        }
        None
    }
}

/// `at` in the local time zone, the earlier of a time a clock change
/// repeats; `None` for a time it skips.
fn local(at: NaiveDateTime) -> Option<u64> {
    let at = Local.from_local_datetime(&at).earliest()?;
    u64::try_from(at.timestamp_millis()).ok()
}

fn has(set: u64, value: u32) -> bool {
    set.checked_shr(value).is_some_and(|bits| bits & 1 == 1)
}

/// One cron field between `low` and `high`: `*`, `5`, `1-5`, `*/15`,
/// `10-50/10`, or a comma list of these.
fn field(text: &str, low: u32, high: u32) -> Result<u64, String> {
    let mut set = 0u64;
    for part in text.split(',') {
        let wrong = || format!("`{part}` is not a cron field between {low} and {high}");
        let (range, step) = match part.split_once('/') {
            Some((range, step)) => (range, step.parse::<usize>().map_err(|_| wrong())?),
            None => (part, 1),
        };
        let number = |text: &str| text.parse::<u32>().map_err(|_| wrong());
        let (from, to) = match range.split_once('-') {
            _ if range == "*" => (low, high),
            Some((from, to)) => (number(from)?, number(to)?),
            None if part.contains('/') => (number(range)?, high),
            None => (number(range)?, number(range)?),
        };
        if step == 0 || from < low || to > high || from > to {
            return Err(wrong());
        }
        for value in (from..=to).step_by(step) {
            set |= 1u64.checked_shl(value).unwrap_or_default();
        }
    }
    Ok(set)
}
