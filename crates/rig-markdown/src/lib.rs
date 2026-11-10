//! Text for a terminal, shared by the rig coding agent's terminal view and
//! its print mode: markdown drawn as styled lines ([`markdown`]) and those
//! lines word-wrapped to a width ([`wrap`]). It does no terminal I/O; the
//! lines are ratatui's, for a caller to draw or write.
//!
//! ```
//! use rig_markdown::markdown::{Palette, Plain, render};
//! use rig_markdown::wrap::wrap_all;
//!
//! let lines: Vec<_> = render("A *short* answer.", &Palette::default(), Some(40), &Plain)
//!     .into_iter()
//!     .map(|(line, _)| line)
//!     .collect();
//! let rows = wrap_all(&lines, 40);
//! assert_eq!(rows.len(), 1);
//! ```

pub mod code;
pub mod markdown;
pub mod wrap;
