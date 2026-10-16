# rig-splits

Splits of a conversation (`SplitsPlugin`), one of the
[rig coding agent](../../crates/rig-harness)'s default plugins: `/split` at any
entry, into a branch of the session or into a new session, `/split move` to
take a range out and leave a pointer behind, `/splits` to list and preview the
lines of any saved session, and `/split switch`, `/split compare` and
`/split merge`. Splits nest to any depth and start from any branch of any saved
session.
