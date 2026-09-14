---
name: General pull request
about: Makes a change to the code base
title: ''
labels: ''
assignees: ''

---

# <Pull Request Title>

## Description

Please include a summary of the changes and the related issue. Please also include relevant motivation and context. List any dependencies that are required for this change.

Fixes # (issue)

## Changelog

<!-- One bullet per user-visible change, in the voice of CHANGELOG.md:
     `- *(scope)* [**breaking**] what changed and why it matters`.
     This is copied into the release notes. Write "None" if nothing is user-visible. -->

## Migration

<!-- Only for breaking or silent-behavior changes. Old form, new form, the
     smallest useful example. This is the raw material for MIGRATING.md.
     Write "None" otherwise. -->

## Type of change

Please delete options that are not relevant.

- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

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
- [ ] I did not edit `CHANGELOG.md` or `MIGRATING.md` (they are generated at release)

## Notes

Any notes you wish to include about the nature of this PR (implementation details, specific questions, etc.)
