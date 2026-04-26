from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_skill_surface_files_exist() -> None:
    assert (ROOT / "SKILL.md").exists()
    assert (ROOT / "HERMES_SETUP.md").exists()
    assert (ROOT / "commands" / "clippos.md").exists()
    assert (ROOT / "commands" / "clippos-config.md").exists()
    assert (ROOT / ".claude-plugin" / "plugin.json").exists()
    assert (ROOT / ".claude-plugin" / "marketplace.json").exists()
    assert (ROOT / ".codex-plugin" / "plugin.json").exists()
    assert (ROOT / ".agents" / "plugins" / "marketplace.json").exists()
    # The universal first-run venv bootstrap script that all three install
    # paths invoke (Claude Code lazy on first /clippos, Codex lazy on first
    # /clippos, Hermes explicitly post-clone).
    bootstrap = ROOT / "scripts" / "bootstrap-venv.sh"
    assert bootstrap.exists()
    assert bootstrap.stat().st_mode & 0o111, "bootstrap-venv.sh must be executable"


def test_env_example_uses_clippos_config_names() -> None:
    env_example = (ROOT / ".env.example").read_text(encoding="utf-8")

    assert "CLIPPOS_OUTPUT_DIR=~/Documents/Clippos" in env_example
    assert "CLIPPOS_RATIOS=9:16,1:1,16:9" in env_example
    assert "CLIPPOS_APPROVE_TOP=5" in env_example
    assert "CLIPPER_" not in env_example
    assert "ClipperTool" not in env_example


def test_bootstrap_resumes_incomplete_venv_installs() -> None:
    bootstrap = (ROOT / "scripts" / "bootstrap-venv.sh").read_text(encoding="utf-8")
    skill = (ROOT / "SKILL.md").read_text(encoding="utf-8")

    assert ".clippos-bootstrap-complete" in bootstrap
    assert "date -u" in bootstrap
    assert "Resuming setup in existing .venv" in bootstrap
    assert "[ -d \"$CLIPPOS_ROOT/.venv\" ] ||" not in skill
    assert 'bash "$CLIPPOS_ROOT/scripts/bootstrap-venv.sh"' in skill


def test_clip_command_invokes_clip_skill() -> None:
    command = (ROOT / "commands" / "clippos.md").read_text(encoding="utf-8")

    assert "Invoke the `clippos` skill" in command
    assert "$ARGUMENTS" in command


def test_skill_instructions_reference_helper_script() -> None:
    skill = (ROOT / "SKILL.md").read_text(encoding="utf-8")

    assert "scripts/clippos_skill.py" in skill
    assert "scoring-response.json" in skill
    assert "approved" in skill


def test_readme_documents_native_install_per_harness() -> None:
    """Each harness installs via its native marketplace / setup flow.
    There is no top-level install.sh — that pattern was deleted in
    favor of per-harness native commands (mirrors mvanhorn/last30days-
    skill). The README must document all three canonical commands."""
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    # Claude Code: native plugin marketplace.
    assert "/plugin marketplace add dylan-buck/Clippos" in readme
    # Codex: native marketplace add (codex-cli >= 0.121).
    assert "codex marketplace add dylan-buck/Clippos" in readme
    # Hermes: clone + bootstrap (no marketplace yet; HERMES_SETUP.md
    # is the canonical Hermes guide).
    assert "git clone https://github.com/dylan-buck/Clippos ~/.hermes/skills/clippos" in readme
    assert "bootstrap-venv.sh" in readme
    assert "HERMES_SETUP.md" in readme


def test_claude_marketplace_json_exposes_clip_plugin() -> None:
    """The .claude-plugin/marketplace.json is what makes
    `/plugin marketplace add dylan-buck/Clippos` work — without it,
    Claude Code returns a 404 from the plugin command. Lock the
    minimum required keys so a future edit can't silently break
    install."""
    import json
    payload = json.loads((ROOT / ".claude-plugin" / "marketplace.json").read_text())
    assert payload.get("name")
    assert isinstance(payload.get("plugins"), list)
    plugin_names = [p.get("name") for p in payload["plugins"]]
    assert "clippos" in plugin_names, (
        "marketplace.json must list a `clippos` plugin entry — that's the "
        "name Claude Code uses when surfacing /clippos:* slash commands."
    )


def test_codex_marketplace_json_exposes_clip_plugin() -> None:
    """The .agents/plugins/marketplace.json is what makes
    `codex marketplace add dylan-buck/Clippos` work — Codex CLI reads
    this file from the cloned repo and registers the plugin in
    ~/.codex/config.toml. Verified against codex-cli 0.121.0."""
    import json
    payload = json.loads((ROOT / ".agents" / "plugins" / "marketplace.json").read_text())
    assert payload.get("name")
    assert isinstance(payload.get("plugins"), list)
    plugin_names = [p.get("name") for p in payload["plugins"]]
    assert "clippos" in plugin_names


def test_skill_is_hermes_first_with_claude_codex_fallback() -> None:
    skill = (ROOT / "SKILL.md").read_text(encoding="utf-8")

    # Hermes is the first-class target…
    assert "${HERMES_SKILL_DIR}" in skill
    assert "/clippos config" in skill
    assert "/clippos package" in skill

    # …but the prologue must fall through to CLAUDE_PLUGIN_ROOT so the skill
    # still works as a Claude Code / Codex plugin, and HERMES_SKILL_DIR must
    # appear earlier in the chain than CLAUDE_PLUGIN_ROOT.
    assert "CLAUDE_PLUGIN_ROOT" in skill
    assert "/clippos-config" in skill
    assert "/clippos-package" in skill
    assert skill.index("${HERMES_SKILL_DIR}") < skill.index("CLAUDE_PLUGIN_ROOT")
    assert 'find "$cache_root" -mindepth 2 -maxdepth 5 -type f' in skill
    assert 'sort -nr' in skill


def test_skill_routes_hermes_flow_through_hermes_clip_helper() -> None:
    skill = (ROOT / "SKILL.md").read_text(encoding="utf-8")

    assert "scripts/hermes_clippos.py" in skill
    assert "advance --source" in skill
    assert "next_action" in skill
    assert (ROOT / "scripts" / "hermes_clippos.py").exists()


def test_skill_documents_messaging_attachment_and_creator_profile_hooks() -> None:
    skill = (ROOT / "SKILL.md").read_text(encoding="utf-8")

    # Attachment URL guidance so Hermes knows to pass Discord/Telegram CDN
    # URLs straight through to the skill.
    assert "Discord" in skill
    assert "Telegram" in skill
    assert "attachment" in skill.lower()

    # Creator-profile memory wiring at both model handoffs.
    assert "Creator Profile" in skill
    assert "creator profile" in skill.lower() or "Creator profile" in skill


def test_skill_documents_feedback_loop_and_creator_patterns() -> None:
    skill = (ROOT / "SKILL.md").read_text(encoding="utf-8")

    # The feedback command must be documented where Hermes can see it.
    assert "Feedback Loop" in skill
    assert "hermes_clippos.py feedback" in skill
    assert "feedback-log.json" in skill
    assert "history.jsonl" in skill

    # creator_patterns must appear at the scoring + packaging handoffs so the
    # harness knows to read it from the advance payload.
    assert "creator_patterns" in skill
    assert skill.count("creator_patterns") >= 2


def test_pyproject_pins_python_and_engine_dep_set() -> None:
    """The engine extras have a known-coexisting version set verified by
    the 2026-04-25 dogfood. Unpinning any of these (especially torch /
    pyannote.audio / transformers / whisperx) breaks the install path
    via the dep-resolution cascade documented in pre-ship-fixes.md.
    Catch a future "let me bump this for flexibility" diff at CI rather
    than at the user's first install.
    """
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

    # Python pin must close the upper bound. bootstrap-venv.sh probes
    # for a Python in [3.12, 3.13) and exits with a clear message
    # otherwise — keep the pyproject pin and the script's range in
    # lockstep.
    assert 'requires-python = ">=3.12,<3.13"' in pyproject

    # Critical pins. Keep the torch / WhisperX / pyannote stack in sync;
    # partial upgrades tend to create resolver conflicts or import failures.
    required_pins = (
        '"torch==2.8.0"',
        '"torchaudio==2.8.0"',
        '"torchvision==0.23.0"',
        '"whisperx==3.8.5"',
        '"transformers>=4.48,<5"',
        '"pyannote.audio>=4.0,<5"',
        '"speechbrain>=1.0,<2"',
        '"matplotlib>=3.7"',
    )
    for pin in required_pins:
        assert pin in pyproject, (
            f"engine extras must pin {pin!r}; dropping this pin reopens the "
            "dep cascade documented in docs/pre-ship-fixes.md (F3)."
        )
