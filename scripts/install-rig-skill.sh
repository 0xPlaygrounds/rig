#!/usr/bin/env bash
set -euo pipefail

# Install the Rig AI coding skill into your project.
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/0xPlaygrounds/rig/main/scripts/install-rig-skill.sh | bash
#
# Or run locally:
#   bash scripts/install-rig-skill.sh
#
# Options:
#   --agent <name>    Install only for a specific agent (claude, cursor, copilot, windsurf, cline, continue, roo, gemini, agents-md)
#   --all             Install for all supported agents
#   --list            List supported agents
#   --dir <path>      Target project directory (default: current directory)

REPO_RAW="https://raw.githubusercontent.com/0xPlaygrounds/rig/main"
TARGET_DIR="."
AGENT_FILTER=""
INSTALL_ALL=false

usage() {
    cat <<'EOF'
Install the Rig AI coding skill into your project.

Usage:
  install-rig-skill.sh [options]

Options:
  --agent <name>    Install for a specific agent only
  --all             Install for all detected agents
  --dir <path>      Target directory (default: .)
  --list            List supported agents
  -h, --help        Show this help

Supported agents:
  claude, cursor, copilot, windsurf, cline, continue, roo, gemini, agents-md

Examples:
  # Auto-detect agents and install
  bash install-rig-skill.sh

  # Install for Claude Code only
  bash install-rig-skill.sh --agent claude

  # Install for all agents
  bash install-rig-skill.sh --all

  # One-liner from GitHub
  curl -fsSL https://raw.githubusercontent.com/0xPlaygrounds/rig/main/scripts/install-rig-skill.sh | bash
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --agent)
            if [[ $# -lt 2 ]] || [[ "$2" == --* ]]; then
                echo "error: --agent requires an argument" >&2
                exit 1
            fi
            AGENT_FILTER="$2"
            shift 2
            ;;
        --all)
            INSTALL_ALL=true
            shift
            ;;
        --dir)
            if [[ $# -lt 2 ]] || [[ "$2" == --* ]]; then
                echo "error: --dir requires an argument" >&2
                exit 1
            fi
            TARGET_DIR="$2"
            shift 2
            ;;
        --list)
            echo "Supported agents: claude, cursor, copilot, windsurf, cline, continue, roo, gemini, agents-md"
            exit 0
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage
            exit 1
            ;;
    esac
done

TARGET_DIR="$(cd "${TARGET_DIR}" && pwd)"

# ── File content ──────────────────────────────────────────────────────
# Check if we're running from the Rig repo (local install) or via curl (remote)

SKILL_SOURCE=""
SKILL_DIR=""
if [[ -f "${SKILL_DIR}/skills/rig/SKILL.md" ]]; then
    SKILL_SOURCE="local"
    SKILL_DIR="$(pwd)"
elif [[ -f "${TARGET_DIR}/skills/rig/SKILL.md" ]]; then
    SKILL_SOURCE="local"
    SKILL_DIR="${TARGET_DIR}"
fi

fetch_file() {
    local rel_path="$1"
    local dest="$2"

    mkdir -p "$(dirname "${dest}")"

    if [[ "${SKILL_SOURCE}" == "local" ]]; then
        if [[ -f "${SKILL_DIR}/${rel_path}" ]]; then
            cp "${SKILL_DIR}/${rel_path}" "${dest}"
            return 0
        fi
    fi

    # Fetch from GitHub
    if curl -fsSL "${REPO_RAW}/${rel_path}" -o "${dest}" 2>/dev/null; then
        return 0
    fi

    echo "  warning: could not fetch ${rel_path}" >&2
    return 1
}

# ── Skill files to install ───────────────────────────────────────────

SKILL_FILES=(
    "skills/rig/SKILL.md"
    "skills/rig/references/tools.md"
    "skills/rig/references/rag.md"
    "skills/rig/references/providers.md"
    "skills/rig/references/patterns.md"
)

# ── Agent detection ──────────────────────────────────────────────────

detect_agents() {
    local agents=()

    # Claude Code: check for .claude/ dir or CLAUDE.md
    if [[ -d "${TARGET_DIR}/.claude" ]] || [[ -f "${TARGET_DIR}/CLAUDE.md" ]]; then
        agents+=("claude")
    fi

    # Cursor: check for .cursor/ dir
    if [[ -d "${TARGET_DIR}/.cursor" ]]; then
        agents+=("cursor")
    fi

    # GitHub Copilot: check for existing copilot config or instructions dir
    if [[ -d "${TARGET_DIR}/.github/instructions" ]] || [[ -f "${TARGET_DIR}/.github/copilot-instructions.md" ]]; then
        agents+=("copilot")
    fi

    # Windsurf: check for .windsurf/ dir or .windsurfrules
    if [[ -d "${TARGET_DIR}/.windsurf" ]] || [[ -f "${TARGET_DIR}/.windsurfrules" ]]; then
        agents+=("windsurf")
    fi

    # Cline: check for .clinerules
    if [[ -d "${TARGET_DIR}/.clinerules" ]] || [[ -f "${TARGET_DIR}/.clinerules" ]]; then
        agents+=("cline")
    fi

    # Continue: check for .continue/ dir
    if [[ -d "${TARGET_DIR}/.continue" ]]; then
        agents+=("continue")
    fi

    # Roo Code: check for .roo/ dir
    if [[ -d "${TARGET_DIR}/.roo" ]]; then
        agents+=("roo")
    fi

    # Gemini CLI: check for GEMINI.md or .gemini/
    if [[ -f "${TARGET_DIR}/GEMINI.md" ]] || [[ -d "${TARGET_DIR}/.gemini" ]]; then
        agents+=("gemini")
    fi

    # If nothing detected, default to claude + agents-md (universal)
    if [[ ${#agents[@]} -eq 0 ]]; then
        agents=("claude" "agents-md")
    fi

    echo "${agents[@]}"
}

# ── Installers per agent ─────────────────────────────────────────────

install_claude() {
    echo "Installing for Claude Code (.claude/skills/rig/)..."
    local dest="${TARGET_DIR}/.claude/skills/rig"
    mkdir -p "${dest}/references"

    for f in "${SKILL_FILES[@]}"; do
        local basename="${f#skills/rig/}"
        fetch_file "${f}" "${dest}/${basename}"
    done

    echo "  Installed: .claude/skills/rig/SKILL.md + 4 reference files"
}

install_cursor() {
    echo "Installing for Cursor (.cursor/rules/rig.mdc)..."
    local dest="${TARGET_DIR}/.cursor/rules"
    mkdir -p "${dest}"

    # Create a single combined rule file for Cursor
    {
        cat <<'HEADER'
---
description: Rig Rust AI framework - architecture patterns, API reference, and coding standards.
globs:
  - "**/*.rs"
alwaysApply: false
---

HEADER
        if [[ "${SKILL_SOURCE}" == "local" ]] && [[ -f "${SKILL_DIR}/skills/rig/SKILL.md" ]]; then
            # Strip YAML frontmatter from SKILL.md
            awk '/^---$/{if(++c==2){skip=0;next}else{skip=1;next}} skip==0{print}' "${SKILL_DIR}/skills/rig/SKILL.md"
        else
            curl -fsSL "${REPO_RAW}/skills/rig/SKILL.md" 2>/dev/null | awk '/^---$/{if(++c==2){skip=0;next}else{skip=1;next}} skip==0{print}'
        fi
    } > "${dest}/rig.mdc"

    echo "  Installed: .cursor/rules/rig.mdc"
}

install_copilot() {
    echo "Installing for GitHub Copilot (.github/instructions/rig.instructions.md)..."
    local dest="${TARGET_DIR}/.github/instructions"
    mkdir -p "${dest}"

    {
        cat <<'HEADER'
---
applyTo: "**/*.rs"
---

HEADER
        if [[ "${SKILL_SOURCE}" == "local" ]] && [[ -f "${SKILL_DIR}/skills/rig/SKILL.md" ]]; then
            awk '/^---$/{if(++c==2){skip=0;next}else{skip=1;next}} skip==0{print}' "${SKILL_DIR}/skills/rig/SKILL.md"
        else
            curl -fsSL "${REPO_RAW}/skills/rig/SKILL.md" 2>/dev/null | awk '/^---$/{if(++c==2){skip=0;next}else{skip=1;next}} skip==0{print}'
        fi
    } > "${dest}/rig.instructions.md"

    echo "  Installed: .github/instructions/rig.instructions.md"
}

install_windsurf() {
    echo "Installing for Windsurf (.windsurf/rules/rig.md)..."
    local dest="${TARGET_DIR}/.windsurf/rules"
    mkdir -p "${dest}"

    if [[ "${SKILL_SOURCE}" == "local" ]] && [[ -f "${SKILL_DIR}/skills/rig/SKILL.md" ]]; then
        awk '/^---$/{if(++c==2){skip=0;next}else{skip=1;next}} skip==0{print}' "${SKILL_DIR}/skills/rig/SKILL.md" > "${dest}/rig.md"
    else
        curl -fsSL "${REPO_RAW}/skills/rig/SKILL.md" 2>/dev/null | awk '/^---$/{if(++c==2){skip=0;next}else{skip=1;next}} skip==0{print}' > "${dest}/rig.md"
    fi

    echo "  Installed: .windsurf/rules/rig.md"
}

install_cline() {
    echo "Installing for Cline (.clinerules/rig.md)..."
    local dest="${TARGET_DIR}/.clinerules"
    mkdir -p "${dest}"

    {
        cat <<'HEADER'
---
paths:
  - "**/*.rs"
---

HEADER
        if [[ "${SKILL_SOURCE}" == "local" ]] && [[ -f "${SKILL_DIR}/skills/rig/SKILL.md" ]]; then
            awk '/^---$/{if(++c==2){skip=0;next}else{skip=1;next}} skip==0{print}' "${SKILL_DIR}/skills/rig/SKILL.md"
        else
            curl -fsSL "${REPO_RAW}/skills/rig/SKILL.md" 2>/dev/null | awk '/^---$/{if(++c==2){skip=0;next}else{skip=1;next}} skip==0{print}'
        fi
    } > "${dest}/rig.md"

    echo "  Installed: .clinerules/rig.md"
}

install_continue() {
    echo "Installing for Continue (.continue/rules/rig.md)..."
    local dest="${TARGET_DIR}/.continue/rules"
    mkdir -p "${dest}"

    {
        cat <<'HEADER'
---
name: Rig AI Framework
globs: "**/*.rs"
alwaysApply: false
---

HEADER
        if [[ "${SKILL_SOURCE}" == "local" ]] && [[ -f "${SKILL_DIR}/skills/rig/SKILL.md" ]]; then
            awk '/^---$/{if(++c==2){skip=0;next}else{skip=1;next}} skip==0{print}' "${SKILL_DIR}/skills/rig/SKILL.md"
        else
            curl -fsSL "${REPO_RAW}/skills/rig/SKILL.md" 2>/dev/null | awk '/^---$/{if(++c==2){skip=0;next}else{skip=1;next}} skip==0{print}'
        fi
    } > "${dest}/rig.md"

    echo "  Installed: .continue/rules/rig.md"
}

install_roo() {
    echo "Installing for Roo Code (.roo/rules/rig.md)..."
    local dest="${TARGET_DIR}/.roo/rules"
    mkdir -p "${dest}"

    if [[ "${SKILL_SOURCE}" == "local" ]] && [[ -f "${SKILL_DIR}/skills/rig/SKILL.md" ]]; then
        awk '/^---$/{if(++c==2){skip=0;next}else{skip=1;next}} skip==0{print}' "${SKILL_DIR}/skills/rig/SKILL.md" > "${dest}/rig.md"
    else
        curl -fsSL "${REPO_RAW}/skills/rig/SKILL.md" 2>/dev/null | awk '/^---$/{if(++c==2){skip=0;next}else{skip=1;next}} skip==0{print}' > "${dest}/rig.md"
    fi

    echo "  Installed: .roo/rules/rig.md"
}

install_gemini() {
    echo "Installing for Gemini CLI (GEMINI.md)..."
    local dest="${TARGET_DIR}/GEMINI.md"

    if [[ -f "${dest}" ]]; then
        if grep -q "Building with Rig" "${dest}" 2>/dev/null; then
            echo "  Rig section already present in GEMINI.md, skipping."
            return
        fi
        echo "  GEMINI.md already exists, appending Rig section..."
        echo "" >> "${dest}"
        echo "---" >> "${dest}"
        echo "" >> "${dest}"
    fi

    if [[ "${SKILL_SOURCE}" == "local" ]] && [[ -f "${SKILL_DIR}/skills/rig/SKILL.md" ]]; then
        awk '/^---$/{if(++c==2){skip=0;next}else{skip=1;next}} skip==0{print}' "${SKILL_DIR}/skills/rig/SKILL.md" >> "${dest}"
    else
        curl -fsSL "${REPO_RAW}/skills/rig/SKILL.md" 2>/dev/null | awk '/^---$/{if(++c==2){skip=0;next}else{skip=1;next}} skip==0{print}' >> "${dest}"
    fi

    echo "  Installed: GEMINI.md"
}

install_agents_md() {
    echo "Installing universal AGENTS.md..."
    local dest="${TARGET_DIR}/AGENTS.md"

    if [[ -f "${dest}" ]]; then
        echo "  AGENTS.md already exists, appending Rig section..."
        # Check if Rig section already present
        if grep -q "Building with Rig" "${dest}" 2>/dev/null; then
            echo "  Rig section already present in AGENTS.md, skipping."
            return
        fi
        echo "" >> "${dest}"
        echo "---" >> "${dest}"
        echo "" >> "${dest}"
    fi

    if [[ "${SKILL_SOURCE}" == "local" ]] && [[ -f "${SKILL_DIR}/skills/rig/SKILL.md" ]]; then
        awk '/^---$/{if(++c==2){skip=0;next}else{skip=1;next}} skip==0{print}' "${SKILL_DIR}/skills/rig/SKILL.md" >> "${dest}"
    else
        curl -fsSL "${REPO_RAW}/skills/rig/SKILL.md" 2>/dev/null | awk '/^---$/{if(++c==2){skip=0;next}else{skip=1;next}} skip==0{print}' >> "${dest}"
    fi

    echo "  Installed: AGENTS.md (works with Codex CLI, Zed, Aider, and more)"
}

# ── Main ──────────────────────────────────────────────────────────────

echo "Rig AI Coding Skill Installer"
echo "=============================="
echo ""

if [[ -n "${AGENT_FILTER}" ]]; then
    agents=("${AGENT_FILTER}")
elif [[ "${INSTALL_ALL}" == true ]]; then
    agents=(claude cursor copilot windsurf cline continue roo gemini agents-md)
else
    echo "Detecting AI coding agents in ${TARGET_DIR}..."
    read -ra agents <<< "$(detect_agents)"
    echo "Detected: ${agents[*]}"
    echo ""
fi

installed=0
for agent in "${agents[@]}"; do
    case "${agent}" in
        claude)    install_claude; installed=$((installed + 1)) ;;
        cursor)    install_cursor; installed=$((installed + 1)) ;;
        copilot)   install_copilot; installed=$((installed + 1)) ;;
        windsurf)  install_windsurf; installed=$((installed + 1)) ;;
        cline)     install_cline; installed=$((installed + 1)) ;;
        continue)  install_continue; installed=$((installed + 1)) ;;
        roo)       install_roo; installed=$((installed + 1)) ;;
        gemini)    install_gemini; installed=$((installed + 1)) ;;
        agents-md) install_agents_md; installed=$((installed + 1)) ;;
        *)
            echo "Unknown agent: ${agent}" >&2
            echo "Supported: claude, cursor, copilot, windsurf, cline, continue, roo, gemini, agents-md" >&2
            ;;
    esac
done

echo ""
echo "Done! Installed Rig skill for ${installed} agent(s)."
echo ""
echo "Your AI coding agent now understands Rig's builder pattern, provider API,"
echo "tool system, RAG pipelines, streaming, and structured extraction."
