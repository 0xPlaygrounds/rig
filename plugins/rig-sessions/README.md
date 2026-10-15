# rig-sessions

Sessions beyond the one running (`SessionsPlugin`), one of the
[rig coding agent](../../crates/rig-harness)'s default plugins: `/new`,
`/resume` (by id or name, or picked in rig-harness's `SessionPicker`) and
`/name`, and, with the default `tui` feature, the session's name in the status
line. It keeps each session's listing (rig-memory's `SessionMeta`) current,
which every session listing reads.
