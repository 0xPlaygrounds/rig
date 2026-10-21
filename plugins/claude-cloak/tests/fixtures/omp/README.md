# OMP offline request corpus

Captured through `streamAnthropic` from oh-my-pi revision
`d5cd750e41fede277f9eb933455da92a0738915e` using Bun on Linux aarch64.
The models are `anthropic/claude-sonnet-4-6` and `anthropic/claude-sonnet-4-5`.
All credentials, account IDs,
session IDs, installation IDs, messages, tools, and responses are synthetic.
The injected fetch routes every request to a loopback HTTP server. No upstream
authentication or inference was performed.

`fixtures.json` contains the shared caller inputs. `home/.omp/install-id` is the
fixed synthetic installation seed. Each `omp-*.json` is a real captured request,
including headers and parsed body; `.raw` contains the received wire body and
`.decoded` contains its exact uncompressed JSON bytes. Local server addresses
in the captures are incidental. Expected body fields, checksums, header values,
and serialization have not been normalized or hand-authored.

The normal `transport::capture::matches_recorded_omp_requests` test runs the
Rig catalog/provider pipeline against a local fake SSE server. It compares the
original URL, application headers and exact uncompressed JSON bytes against
this corpus. Host, content length and transfer encoding are local transport framing,
excluded from application-header comparison. Content encoding must match. Compressed
wire bytes are retained separately; this test does not claim identical gzip
framing.

The 48 cases cover the default request, tools, adaptive thinking, Unicode
history, absent system prompt, a request above the compression threshold,
explicit auto/any/none/named tool choices, builtin/custom/underscore tool
names, tool-result history, none/short/long cache retention, JSON escaping,
budget-based thinking, numeric schema keys and floats, the yield tool,
sampling parameters with thinking, and five stop sequences.
Additional paired boundaries cover histories of 19/20/21 and 29/30/31 messages
with one or three system blocks, and required/named/disabled tools combined
with adaptive and budget thinking. The latter history lengths cross the
15-conversational-user-turn cache checkpoint.

Compression inputs were calibrated against actual OMP serialized output, not
prompt lengths. ASCII and emoji cases have exactly 1023/1024/1025 UTF-16 code
units; separate `emoji_bytes` cases have exactly 1023/1024/1025 UTF-8 bytes.
`expectedUtf16Length` records and checks the calibration. The source SDK's
attestation wrapper supplies bytes to compression, so the actual OAuth path
compresses at 1024 UTF-8 bytes, even when UTF-16 length is below 1024.
These source-generated expectations intentionally retain any differences from
Rig; they are not rewritten to make the regression pass.

They do not cover every OMP option, model, schema construct or binary content.
