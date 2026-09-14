---
name: New model provider
about: Suggest a new model provider to integrate
title: 'feat: Add support for X'
labels: feat, model
assignees: ''

---

# New model provider: <Model Provider Name>

## Description

Please describe the model provider you are adding to the project. Include links to their website and their api documentation.

Fixes # (issue)

## Changelog

<!-- One bullet per user-visible change, in the voice of CHANGELOG.md:
     `- *(scope)* [**breaking**] what changed and why it matters`.
     This is copied into the release notes. Write "None" if nothing is user-visible. -->

## Migration

<!-- Only for breaking or silent-behavior changes. Old form, new form, the
     smallest useful example. This is the raw material for MIGRATING.md.
     Write "None" otherwise. -->

## Testing

Describe the minimal relevant local checks actually run and how to reproduce them. Report committed-head CI status separately; CI may be pending when opening the PR, but required checks and comprehensive acceptance coverage must pass before claiming fully verified or ready to merge. Broad local suites are optional; see [DEVELOPING.md](../../DEVELOPING.md).

- [ ] Test A
- [ ] Test B

## Checklist:

- [ ] My code follows the style guidelines of this project
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have updated READMEs and Rust docs affected by this change
- [ ] My changes generate no new warnings
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] I have completed minimal relevant local checks and reported their results separately from CI status
- [ ] I've reviewed the provider API documentation and implemented the types of response accurately
- [ ] I did not edit `CHANGELOG.md` or `MIGRATING.md` (they are generated at release)

## Notes

Any notes you wish to include about the nature of this PR (implementation details, specific questions, etc.)
