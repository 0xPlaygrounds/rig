"""rig-coder as a Harbor installed agent.

Uploads the Linux binary from `bench/build-linux.sh` into the task container,
runs it on the task instruction in the task's working directory, and keeps the
transcript and effect log in the trial's agent log directory.

    PYTHONPATH=test-support/rig-coder harbor run -p <tasks> \
        -a bench.rig_coder_agent:RigCoderAgent -m gemini/gemini-3.8-flash \
        --ae GEMINI_API_KEY=$GEMINI_API_KEY --force-build -o test-support/rig-coder/runs

Only the selected provider's key reaches the container, and only when passed
with --ae; it is never printed.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from harbor.agents.installed.base import BaseInstalledAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

HERE = Path(__file__).resolve().parent
REMOTE_BIN = "/usr/local/bin/rig-coder"
REMOTE_LOGS = "/logs/agent"
PROVIDER_KEYS = {
    "gemini": "GEMINI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
}
ARCHES = {"x86_64": "x86_64", "amd64": "x86_64", "aarch64": "aarch64", "arm64": "aarch64"}


class RigCoderAgent(BaseInstalledAgent):
    """The prebuilt `rig-coder` binary driven on the task instruction."""

    @staticmethod
    def name() -> str:
        return "rig-coder"

    def version(self) -> str | None:
        receipt = next(HERE.glob("bin/*.build.json"), None)
        if receipt is None:
            return "unbuilt"
        return json.loads(receipt.read_text()).get("source_head", "unknown")[:12]

    async def install(self, environment: BaseEnvironment) -> None:
        # The binary must match the container, which may be emulated.
        machine = await self.exec_as_root(environment, "uname -m")
        arch = ARCHES.get((machine.stdout or "").strip())
        binary = Path(os.environ.get("RIG_CODER_BIN") or HERE / "bin" / f"rig-coder-linux-{arch}")
        if not binary.is_file():
            raise FileNotFoundError(
                f"{binary} is missing for container arch {machine.stdout!r}; "
                f"run ARCH={arch or 'x86_64'} test-support/rig-coder/bench/build-linux.sh"
            )
        await environment.upload_file(binary, REMOTE_BIN)
        await self.exec_as_root(environment, f"chmod 755 {REMOTE_BIN}")
        # The bash tool needs bash, and TLS needs a CA store; bare images lack both.
        await self.exec_as_root(
            environment,
            "(command -v bash >/dev/null && [ -s /etc/ssl/certs/ca-certificates.crt ]) "
            "|| (apt-get update -qq && apt-get install -y -qq bash ca-certificates) "
            "|| (apk add --no-cache bash ca-certificates) || true",
        )

    def _provider_and_model(self) -> tuple[str, str]:
        name = self.model_name or "gemini/gemini-3.8-flash"
        provider, _, model = name.partition("/")
        if not model:
            raise ValueError(f"model {name!r} must be provider/model")
        return provider, model

    async def run(
        self, instruction: str, environment: BaseEnvironment, context: AgentContext
    ) -> None:
        provider, model = self._provider_and_model()
        env = {"RUST_LOG": os.environ.get("RIG_CODER_RUST_LOG", "warn")}
        key = PROVIDER_KEYS.get(provider)
        if key and self._extra_env.get(key):
            env[key] = self._extra_env[key]
        timeout = int(os.environ.get("RIG_CODER_TIMEOUT_SECS", "3600"))
        max_turns = int(os.environ.get("RIG_CODER_MAX_TURNS", "200"))
        # The instruction goes through a file, never argv: it is untrusted text.
        task_file = self.logs_dir / "instruction.md"
        task_file.write_text(instruction)
        await environment.exec(command=f"mkdir -p {REMOTE_LOGS}", user="root")
        await environment.upload_file(task_file, f"{REMOTE_LOGS}/instruction.md")
        if os.environ.get("RIG_CODER_NETWORK_PROBE"):
            await self._probe_network(environment)
        workdir = (await environment.exec(command="pwd")).stdout.strip() or "/app"
        command = (
            f"{REMOTE_BIN} --cwd {workdir} --provider {provider} --model {model} "
            f"--max-turns {max_turns} --timeout-secs {timeout} "
            f"--task-file {REMOTE_LOGS}/instruction.md "
            f"--transcript {REMOTE_LOGS}/transcript.jsonl "
            f"--effect-log {REMOTE_LOGS}/effects.json "
            f"> {REMOTE_LOGS}/rig-coder.txt 2>&1; status=$?; "
            f"echo $status > {REMOTE_LOGS}/exit_code.txt; exit $status"
        )
        # A failed run is still a trial to verify, so keep the exit code instead of raising.
        result = await environment.exec(
            command=command, cwd=workdir, env=env, timeout_sec=timeout + 60, user="root"
        )
        (self.logs_dir / "exit_code.txt").write_text(str(result.return_code))

    async def _probe_network(self, environment: BaseEnvironment) -> None:
        """Records whether arbitrary egress is denied in the agent phase.

        A TCP connect is not evidence, since an egress proxy accepts and then
        drops it; only an HTTP response counts as reached.
        """
        probe = await environment.exec(
            command="python3 - <<'EOF'\n"
            "import urllib.request\n"
            "for url in ('http://93.184.215.14/', 'https://example.com/', 'https://github.com/'):\n"
            "    try:\n"
            "        with urllib.request.urlopen(url, timeout=8) as r:\n"
            "            print(url, 'REACHED', r.status)\n"
            "    except Exception as e:\n"
            "        print(url, 'DENIED', type(e).__name__, str(e)[:80])\n"
            "EOF",
            user="root",
            timeout_sec=60,
        )
        (self.logs_dir / "network-probe.txt").write_text(
            f"stdout={probe.stdout!r}\nstderr={probe.stderr!r}\nreturn_code={probe.return_code}\n"
        )

    def populate_context_post_run(self, context: AgentContext) -> None:
        transcript = self.logs_dir / "transcript.jsonl"
        if not transcript.is_file():
            return
        events = []
        for line in transcript.read_text().splitlines():
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        usage = [e for e in events if e.get("kind") == "usage"]

        def total(field: str) -> int | None:
            values = [e.get(field) for e in usage]
            if not values:
                return None
            return sum(v for v in values if isinstance(v, int))

        context.n_input_tokens = total("input_tokens")
        context.n_output_tokens = total("output_tokens")
        context.n_cache_tokens = total("cached_input_tokens")
        ending = next((e for e in reversed(events) if e.get("kind") in ("settled", "failed")), None)
        context.metadata = {
            "turns": len(usage),
            "tool_calls": sum(1 for e in events if e.get("kind") == "tool_call"),
            "ending": ending.get("kind") if ending else "killed",
            "error": ending.get("error") if ending else None,
        }
