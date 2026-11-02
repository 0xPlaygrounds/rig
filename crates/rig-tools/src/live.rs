//! [`Live`]: where a running tool reports its output as it comes, for a
//! host that shows it before the call ends or keeps it in case the call is
//! stopped; and [`LineFilter`], what each line goes through first.

use std::fmt;
use std::sync::Arc;

/// Where a running tool reports output as it comes, such as
/// [`Shell`](crate::Shell)'s lines. Cloning shares the sink.
///
/// ```
/// use std::sync::{Arc, Mutex};
/// use rig_tools::Live;
///
/// let seen = Arc::new(Mutex::new(String::new()));
/// let live = Live::new({
///     let seen = seen.clone();
///     move |text| seen.lock().map(|mut seen| seen.push_str(text)).unwrap_or_default()
/// });
/// live.line("building\n");
/// assert_eq!(seen.lock().map(|seen| seen.clone()).ok().as_deref(), Some("building\n"));
/// ```
#[derive(Clone)]
pub struct Live(Arc<dyn Fn(&str) + Send + Sync>);

impl Live {
    /// A sink that hands each piece of output to `report`, on the thread the
    /// tool runs on: keep `report` quick.
    pub fn new(report: impl Fn(&str) + Send + Sync + 'static) -> Self {
        Self(Arc::new(report))
    }

    /// Reports `text`, the next piece of output: a line with its `\n`, or
    /// the end of the output without one.
    pub fn line(&self, text: &str) {
        (self.0)(text);
    }
}

impl fmt::Debug for Live {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Live")
    }
}

/// What each line of a tool's output, such as [`Shell`](crate::Shell)'s,
/// goes through before it is reported, spilled or kept, such as a
/// redactor. Clones share its state: give each call its own.
#[derive(Clone)]
pub struct LineFilter(Arc<dyn Fn(&str) -> String + Send + Sync>);

impl LineFilter {
    /// A filter that hands each line to `filter`, in order.
    pub fn new(filter: impl Fn(&str) -> String + Send + Sync + 'static) -> Self {
        Self(Arc::new(filter))
    }

    /// `line`, with its `\n` if it has one, as the filter changes it; empty
    /// leaves it out.
    pub fn apply(&self, line: &str) -> String {
        (self.0)(line)
    }
}

impl fmt::Debug for LineFilter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("LineFilter")
    }
}
