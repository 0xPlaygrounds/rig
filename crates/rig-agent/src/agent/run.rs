//! Run protocol types and request preparation re-exported from [`crate::run`].
//!
//! ```
//! use rig_agent::agent::run::spec::RunSpec;
//! let spec = RunSpec::new();
//! assert_eq!(spec.effective_max_turns(), 1);
//! ```

pub use crate::run::*;

/// Streamed-turn accumulation, re-exported from [`crate::run::streamed`].
///
/// ```
/// use rig_agent::agent::run::streamed::StreamedResolution;
/// let resolution = StreamedResolution::Ignored;
/// ```
pub mod streamed {
    pub use crate::run::streamed::*;
}

/// Structured-output mode, re-exported from [`crate::run::output`].
///
/// ```
/// use rig_agent::agent::run::output_mode::OutputMode;
/// let mode = OutputMode::Native;
/// ```
pub mod output_mode {
    pub use crate::run::output::*;
}
