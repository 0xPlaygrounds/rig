# rig-markdown

Text for a terminal, shared by the [rig](https://github.com/0xPlaygrounds/rig) coding agent's
terminal view (`rig-tui`) and its print mode (`rig -p`, in `rig-basics`):

- `markdown`: markdown drawn as styled ratatui lines: headings, emphasis, inline and fenced code,
  lists, quotes and GitHub's alerts, links, rules and tables fit to a width. A caller colors fenced
  code through the `Highlight` trait, or leaves it plain with `Plain`.
- `wrap`: styled lines word-wrapped to a width, a list item's continuation rows under its text.
  Only a word wider than a row is broken.

It does no terminal I/O and has no Bevy dependency.
