#!/usr/bin/env python3
# CHUNK_META:
#   Purpose: Code-grounded audit of implemented Brain mechanisms with ✅/❌ and code pointers
#   Dependencies: standard library + local Brain modules + yaml
#   API: main()

from __future__ import annotations

from dataclasses import dataclass
import inspect
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


# ANCHOR: CONTRACTS_SOFT_ASSERT_AUDIT - non-fatal contracts helper
# API_PRIVATE
def _soft_assert(condition: bool, message: str) -> bool:
    """Soft contract assertion.

    Description:
        Evaluates a condition as an architectural contract and does not crash production.

    Intent:
        Enforce explicit pre/postconditions during development while allowing
        production execution to continue if a contract is violated.

    Args:
        condition: Condition expected to be True.
        message: Why this condition matters.

    Returns:
        True if condition is True, False otherwise.

    Raises:
        None.
    """
    try:
        assert condition, message
        return True
    except AssertionError:
        return False


# ANCHOR: AUDIT_RESULT_DATACLASS - structured audit output
@dataclass(frozen=True)
class AuditResult:
    """Result of a single mechanism audit.

    Description:
        Captures pass/fail and code references for a mechanism.

    Intent:
        Make audit output machine-readable and consistent.

    Args:
        mechanism_id: Mechanism identifier.
        ok: Whether mechanism is detected and minimally validated.
        details: Short validation details.
        code_refs: List of code reference strings (file:line:symbol).

    Returns:
        None.

    Raises:
        None.
    """

    mechanism_id: str
    ok: bool
    details: str
    code_refs: Tuple[str, ...]


# ANCHOR: REGISTRY_TYPES - typed registry structures
@dataclass(frozen=True)
class RegistryCodeRef:
    """A single code reference entry from mechanism_registry.yml.

    Description:
        Represents one (file, symbol, anchor) tuple that should be verifiable in the codebase.

    Intent:
        Provide a strongly-typed boundary for registry parsing and auditing.

    Args:
        file: Relative file path in repository.
        symbol: Optional python symbol name/path for best-effort validation.
        anchor: Optional anchor string expected to be present in the file.

    Returns:
        None.

    Raises:
        None.
    """

    file: str
    symbol: Optional[str]
    anchor: Optional[str]


# ANCHOR: REGISTRY_TYPES_MECHANISM - typed mechanism entry
@dataclass(frozen=True)
class RegistryMechanism:
    """A single mechanism entry from mechanism_registry.yml.

    Description:
        Holds the mechanism id and its verifiable code references.

    Intent:
        Enable registry↔code consistency checks without relying on ad-hoc dict access.

    Args:
        mechanism_id: Mechanism identifier.
        code: Code references that should exist.

    Returns:
        None.

    Raises:
        None.
    """

    mechanism_id: str
    code: Tuple[RegistryCodeRef, ...]


# ANCHOR: CODE_REF_HELPERS - helpers for stable code pointers
# API_PRIVATE
def _code_ref(obj: Any, symbol: str | None = None) -> str:
    """Build a stable code reference for an object.

    Description:
        Locates the source file and line for a python object.

    Intent:
        Provide explicit code pointers in audit output.

    Args:
        obj: Python object with source.
        symbol: Optional human symbol name.

    Returns:
        Reference string in the form 'path:line:symbol'.

    Raises:
        None.
    """
    try:
        file_path = inspect.getsourcefile(obj) or "<unknown>"
        _, line_no = inspect.getsourcelines(obj)
        sym = symbol or getattr(obj, "__qualname__", getattr(obj, "__name__", "<symbol>"))
        return f"{Path(file_path).name}:{line_no}:{sym}"
    except Exception:
        sym = symbol or getattr(obj, "__qualname__", getattr(obj, "__name__", "<symbol>"))
        return f"<unknown>:?:{sym}"


# ANCHOR: AUDIT_IMPORTS - ensure local imports work
# API_PRIVATE
def _ensure_repo_on_path() -> None:
    """Ensure repository root is on sys.path.

    Description:
        Adds this file's parent directory to sys.path.

    Intent:
        Allow running audit as a script without packaging assumptions.

    Args:
        None.

    Returns:
        None.

    Raises:
        None.
    """
    repo_dir = Path(__file__).resolve().parent
    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))


# ANCHOR: REGISTRY_LOAD - load mechanism_registry.yml
# API_PRIVATE
def _load_mechanism_registry(repo_dir: Path) -> Tuple[bool, str, Tuple[RegistryMechanism, ...]]:
    """Load and parse mechanism_registry.yml.

    Description:
        Reads the registry file and converts it into typed mechanism entries.

    Intent:
        Keep the registry as the single source of truth for implemented mechanisms
        while ensuring its references are machine-verifiable.

    Args:
        repo_dir: Repository directory containing mechanism_registry.yml.

    Returns:
        (ok, details, mechanisms).

    Raises:
        None.
    """
    _soft_assert(repo_dir.exists(), "repo_dir must exist to locate mechanism_registry.yml")

    registry_path = repo_dir / "mechanism_registry.yml"
    if not registry_path.exists():
        _soft_assert(False, "mechanism_registry.yml must exist to audit registry↔code consistency")
        return False, "missing_registry_file", tuple()

    try:
        import yaml  # type: ignore
    except Exception as e:  # pragma: no cover
        return False, f"yaml_unavailable={type(e).__name__}", tuple()

    try:
        raw_text = registry_path.read_text(encoding="utf-8")
        data = yaml.safe_load(raw_text) or {}

        mechs_raw = data.get("mechanisms", [])
        if not isinstance(mechs_raw, list):
            return False, "registry_mechanisms_not_list", tuple()

        mechanisms: List[RegistryMechanism] = []
        for m in mechs_raw:
            if not isinstance(m, dict):
                continue
            mech_id = m.get("id")
            code_raw = m.get("code", [])
            if not isinstance(mech_id, str) or not mech_id:
                continue
            if not isinstance(code_raw, list):
                code_raw = []

            code_refs: List[RegistryCodeRef] = []
            for c in code_raw:
                if not isinstance(c, dict):
                    continue
                file_val = c.get("file")
                symbol_val = c.get("symbol")
                anchor_val = c.get("anchor")
                if not isinstance(file_val, str) or not file_val:
                    continue
                code_refs.append(
                    RegistryCodeRef(
                        file=file_val,
                        symbol=symbol_val if isinstance(symbol_val, str) and symbol_val else None,
                        anchor=anchor_val if isinstance(anchor_val, str) and anchor_val else None,
                    )
                )

            mechanisms.append(RegistryMechanism(mechanism_id=mech_id, code=tuple(code_refs)))

        ok = len(mechanisms) > 0
        _soft_assert(ok, "registry should contain at least one mechanism to be meaningful")
        return ok, f"mechanisms={len(mechanisms)}", tuple(mechanisms)
    except Exception as e:  # pragma: no cover
        return False, f"registry_parse_error={type(e).__name__}: {str(e)[:120]}", tuple()


# ANCHOR: REGISTRY_SYMBOL_CHECK - best-effort python symbol resolver
# API_PRIVATE
def _try_resolve_symbol(repo_dir: Path, file_rel: str, symbol: str) -> bool:
    """Best-effort validation that a python symbol exists for a given file.

    Description:
        Imports the module matching file_rel and traverses dot-separated identifiers.

    Intent:
        Catch stale registry symbol pointers when symbols are simple importable paths.

    Args:
        repo_dir: Repository directory.
        file_rel: Relative file path, expected to end with '.py'.
        symbol: Symbol path like 'Class.method' or 'function_name'.

    Returns:
        True if symbol resolves, False otherwise.

    Raises:
        None.
    """
    _soft_assert(bool(file_rel), "file_rel must be provided to resolve a symbol")
    _soft_assert(bool(symbol), "symbol must be non-empty to be resolvable")

    if not file_rel.endswith(".py"):
        return False

    # Only attempt for identifier-like dotted paths.
    import re

    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*", symbol) is None:
        return False

    module_name = Path(file_rel).stem
    try:
        import importlib

        module = importlib.import_module(module_name)
        obj: Any = module
        for part in symbol.split("."):
            obj = getattr(obj, part)
        return obj is not None
    except Exception:
        return False
    finally:
        # Postcondition
        _soft_assert(True, "symbol resolution must never crash audit execution")


# ANCHOR: REGISTRY_AUDIT - registry↔code consistency check
# API_PRIVATE
def _audit_registry_vs_code() -> List[AuditResult]:
    """Audit that mechanism_registry.yml matches the current codebase.

    Description:
        Validates that every registry mechanism references existing files and anchors.
        When the registry 'symbol' is a resolvable python dotted path, it is validated too.

    Intent:
        Prevent the mechanism registry from becoming stale by making it executable.

    Args:
        None.

    Returns:
        List of AuditResult entries (one per registry mechanism + one summary).

    Raises:
        None.
    """
    repo_dir = Path(__file__).resolve().parent

    ok_loaded, details, mechanisms = _load_mechanism_registry(repo_dir)
    summary_refs = (str((repo_dir / "mechanism_registry.yml").name),)

    if not ok_loaded:
        return [
            AuditResult(
                mechanism_id="registry_mechanisms_loaded",
                ok=False,
                details=details,
                code_refs=summary_refs,
            )
        ]

    file_cache: Dict[Path, str] = {}
    results: List[AuditResult] = []

    ok_count = 0
    for mech in mechanisms:
        # Preconditions
        _soft_assert(bool(mech.mechanism_id), "registry mechanism id must be non-empty")

        missing_files: List[str] = []
        missing_anchors: List[str] = []
        missing_symbols: List[str] = []
        checked = 0

        for ref in mech.code:
            checked += 1
            file_path = repo_dir / ref.file
            if not file_path.exists():
                missing_files.append(ref.file)
                continue

            if ref.anchor:
                if file_path not in file_cache:
                    try:
                        file_cache[file_path] = file_path.read_text(encoding="utf-8")
                    except Exception:
                        file_cache[file_path] = ""
                if ref.anchor not in file_cache[file_path]:
                    missing_anchors.append(f"{ref.file}:{ref.anchor}")

            if ref.symbol:
                resolved = _try_resolve_symbol(repo_dir, ref.file, ref.symbol)
                if ref.symbol and (resolved is False) and ref.file.endswith(".py"):
                    # Only count as missing if it looked resolvable (identifier dotted path)
                    import re

                    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*", ref.symbol):
                        missing_symbols.append(f"{ref.file}:{ref.symbol}")

        ok = len(missing_files) == 0 and len(missing_anchors) == 0 and len(missing_symbols) == 0
        if ok:
            ok_count += 1

        detail_parts = [f"checked={checked}"]
        if missing_files:
            detail_parts.append(f"missing_files={len(missing_files)}")
        if missing_anchors:
            detail_parts.append(f"missing_anchors={len(missing_anchors)}")
        if missing_symbols:
            detail_parts.append(f"missing_symbols={len(missing_symbols)}")

        results.append(
            AuditResult(
                mechanism_id=f"registry_{mech.mechanism_id}",
                ok=ok,
                details=", ".join(detail_parts),
                code_refs=tuple(sorted(set([r.file for r in mech.code]))),
            )
        )

    overall_ok = ok_count == len(mechanisms)
    _soft_assert(isinstance(overall_ok, bool), "registry audit must produce a boolean overall verdict")

    results.insert(
        0,
        AuditResult(
            mechanism_id="registry_mechanisms_loaded",
            ok=True,
            details=f"{details}, ok={ok_count}/{len(mechanisms)}",
            code_refs=summary_refs,
        ),
    )

    # Postcondition
    _soft_assert(len(results) >= 1, "registry audit must return at least a summary result")
    return results


# ANCHOR: AUDIT_MECHANISMS - individual audits
# API_PRIVATE
def _audit_connection_states() -> AuditResult:
    """Audit discrete connection states NEW/USED/MYELINATED/PRUNE.

    Description:
        Checks enums existence and required members.

    Intent:
        Verify that state machine is present as specified.

    Args:
        None.

    Returns:
        AuditResult.

    Raises:
        None.
    """
    from connection import ConnectionState as CS

    required = {"NEW", "USED", "MYELINATED", "PRUNE"}
    present = {m.name for m in CS}

    ok = required.issubset(present)
    _soft_assert(ok, "ConnectionState must include NEW/USED/MYELINATED/PRUNE to support discrete plasticity")

    return AuditResult(
        mechanism_id="states_NEW_USED_MYELINATED_PRUNE",
        ok=ok,
        details=f"ConnectionState members={sorted(present)}",
        code_refs=(_code_ref(CS, "ConnectionState"),),
    )


# API_PRIVATE
# ANCHOR: AUDIT_INHIBITION_WTA - inhibition/WTA/working memory cap
def _audit_inhibition_wta_wm_cap() -> AuditResult:
    """Audit inhibition, WTA, and working-memory capacity cap.

    Description:
        Ensures Activation has inhibition machinery and WM limit constants.

    Intent:
        Verify competition mechanisms used during activation spread.

    Args:
        None.

    Returns:
        AuditResult.

    Raises:
        None.
    """
    import activation as act

    has_inhibition = hasattr(act.Activation, "_apply_inhibition")
    has_wm_limit = hasattr(act, "WORKING_MEMORY_LIMIT")

    ok = bool(has_inhibition and has_wm_limit)
    _soft_assert(ok, "Activation must implement inhibition and a WM cap to match biological constraints")

    refs: List[str] = []
    refs.append(_code_ref(act.Activation, "Activation"))
    if has_inhibition:
        refs.append(_code_ref(act.Activation._apply_inhibition, "Activation._apply_inhibition"))

    return AuditResult(
        mechanism_id="inhibition_WTA_working_memory_cap",
        ok=ok,
        details=f"has_inhibition={has_inhibition}, WORKING_MEMORY_LIMIT={getattr(act, 'WORKING_MEMORY_LIMIT', None)}",
        code_refs=tuple(refs),
    )


# API_PRIVATE
# ANCHOR: AUDIT_HIPPOCAMPUS - hippocampus encode/retrieve/replay
def _audit_hippocampus_encode_retrieve_replay() -> AuditResult:
    """Audit hippocampus encode/retrieve/replay APIs.

    Description:
        Checks Hippocampus has encode/retrieve/sleep and exercises a tiny scene.

    Intent:
        Ensure episodic memory functions exist and are callable.

    Args:
        None.

    Returns:
        AuditResult.

    Raises:
        None.
    """
    from cortex import Cortex
    from hippocampus import Hippocampus

    has_encode = hasattr(Hippocampus, "encode")
    has_retrieve = hasattr(Hippocampus, "retrieve")
    has_sleep = hasattr(Hippocampus, "sleep")

    ok = bool(has_encode and has_retrieve and has_sleep)

    details = f"encode={has_encode}, retrieve={has_retrieve}, sleep={has_sleep}"

    # Minimal functional check
    if ok:
        cortex = Cortex()
        hip = Hippocampus(cortex)
        ep = hip.encode({"dog", "animal", "pet"}, source="audit")
        recalled = hip.retrieve({"dog"})
        stats = hip.sleep(cycles=1, word_to_neuron=None)

        ok = bool(ep is not None and stats is not None)
        details = f"episode={getattr(ep, 'id', None)}, recalled={getattr(recalled, 'id', None) if recalled else None}, sleep_stats={stats}"

    _soft_assert(ok, "Hippocampus must support encode/retrieve/sleep replay for episodic memory")

    refs = (
        _code_ref(Hippocampus, "Hippocampus"),
        _code_ref(Hippocampus.encode, "Hippocampus.encode"),
        _code_ref(Hippocampus.retrieve, "Hippocampus.retrieve"),
        _code_ref(Hippocampus.sleep, "Hippocampus.sleep"),
    )

    return AuditResult(
        mechanism_id="hippocampus_encode_retrieve_replay",
        ok=ok,
        details=details,
        code_refs=refs,
    )


# API_PRIVATE
# ANCHOR: AUDIT_NEUROMODULATORS - neuromodulators influence matrix
def _audit_neuromodulators_matrix() -> Tuple[AuditResult, Dict[str, Dict[str, List[str]]]]:
    """Audit neuromodulators and return an influence matrix.

    Description:
        Detects the neuromodulator system and reports where DA/ACh/NE/5HT enter and what they change.

    Intent:
        Provide explicit, code-grounded modulation points.

    Args:
        None.

    Returns:
        (AuditResult, influence_matrix).

    Raises:
        None.
    """
    import activation as act

    ok = hasattr(act, "Neuromodulator") and hasattr(act, "NeuromodulatorSystem") and hasattr(act.Activation, "release_neuromodulator")

    influence: Dict[str, Dict[str, List[str]]] = {
        "DA": {"enters": [], "changes": []},
        "ACh": {"enters": [], "changes": []},
        "NE": {"enters": [], "changes": []},
        "5HT": {"enters": [], "changes": []},
    }

    if ok:
        influence["DA"]["enters"].append(_code_ref(act.Activation.release_neuromodulator, "Activation.release_neuromodulator"))
        influence["ACh"]["enters"].append(_code_ref(act.Activation.release_neuromodulator, "Activation.release_neuromodulator"))
        influence["NE"]["enters"].append(_code_ref(act.Activation.release_neuromodulator, "Activation.release_neuromodulator"))
        influence["5HT"]["enters"].append(_code_ref(act.Activation.release_neuromodulator, "Activation.release_neuromodulator"))

        influence["DA"]["changes"].append(_code_ref(act.NeuromodulatorSystem.get_learning_rate_modifier, "NeuromodulatorSystem.get_learning_rate_modifier"))
        influence["ACh"]["changes"].append(_code_ref(act.NeuromodulatorSystem.get_learning_rate_modifier, "NeuromodulatorSystem.get_learning_rate_modifier"))
        influence["5HT"]["changes"].append(_code_ref(act.NeuromodulatorSystem.get_learning_rate_modifier, "NeuromodulatorSystem.get_learning_rate_modifier"))

        influence["NE"]["changes"].append(_code_ref(act.NeuromodulatorSystem.get_excitability_modifier, "NeuromodulatorSystem.get_excitability_modifier"))
        influence["5HT"]["changes"].append(_code_ref(act.NeuromodulatorSystem.get_excitability_modifier, "NeuromodulatorSystem.get_excitability_modifier"))

        # Spike pipeline integration point
        influence["DA"]["changes"].append(_code_ref(act.Activation.run_spike_simulation, "Activation.run_spike_simulation"))
        influence["ACh"]["changes"].append(_code_ref(act.Activation.run_spike_simulation, "Activation.run_spike_simulation"))
        influence["NE"]["changes"].append(_code_ref(act.Activation.run_spike_simulation, "Activation.run_spike_simulation"))
        influence["5HT"]["changes"].append(_code_ref(act.Activation.run_spike_simulation, "Activation.run_spike_simulation"))

    _soft_assert(ok, "NeuromodulatorSystem must exist and be wired into Activation for three-factor modulation")

    return (
        AuditResult(
            mechanism_id="neuromodulators_DA_ACh_NE_5HT",
            ok=ok,
            details=f"NeuromodulatorSystem_present={ok}",
            code_refs=(
                _code_ref(act.Neuromodulator, "Neuromodulator"),
                _code_ref(act.NeuromodulatorSystem, "NeuromodulatorSystem"),
                _code_ref(act.Activation.release_neuromodulator, "Activation.release_neuromodulator"),
            )
            if ok
            else tuple(),
        ),
        influence,
    )


# API_PRIVATE
# ANCHOR: AUDIT_STDP - STDP and discrete usage counters
def _audit_stdp_and_usage_counters() -> AuditResult:
    """Audit STDP existence and discrete usage counters.

    Description:
        Checks Connection.apply_stdp and presence of forward_usage/backward_usage.

    Intent:
        Ensure spike-timing learning exists and that discrete counters are available.

    Args:
        None.

    Returns:
        AuditResult.

    Raises:
        None.
    """
    from neuron import Neuron
    from connection import Connection

    n1 = Neuron("audit_pre")
    n2 = Neuron("audit_post")
    conn = Connection(n1, n2)

    has_apply_stdp = hasattr(conn, "apply_stdp") and callable(getattr(conn, "apply_stdp"))
    has_counters = hasattr(conn, "forward_usage") and hasattr(conn, "backward_usage")

    ok = bool(has_apply_stdp and has_counters)
    _soft_assert(ok, "Connection must expose STDP and usage counters to support plasticity audits")

    refs: List[str] = [_code_ref(Connection, "Connection")]
    if has_apply_stdp:
        refs.append(_code_ref(Connection.apply_stdp, "Connection.apply_stdp"))

    return AuditResult(
        mechanism_id="stdp_and_usage_counters",
        ok=ok,
        details=f"apply_stdp={has_apply_stdp}, counters={has_counters}, usage_count={conn.usage_count}",
        code_refs=tuple(refs),
    )


# API_PRIVATE
# ANCHOR: AUDIT_BG_THALAMUS - BG/Thalamus gating and action selection log
def _audit_bg_thalamus_gating() -> AuditResult:
    """Audit BG/Thalamus gating module and decision logging.

    Description:
        Checks that basal_ganglia gating exists and returns a decision trace.

    Intent:
        Provide at least a stub with explicit action selection logging.

    Args:
        None.

    Returns:
        AuditResult.

    Raises:
        None.
    """
    try:
        from basal_ganglia import BasalGangliaThalamusGating

        gating = BasalGangliaThalamusGating()
        decision = gating.select_action(
            ["retrieve", "encode", "sleep"],
            context={"task": "audit"},
            neuromodulators={"DA": 0.4, "ACh": 0.3, "NE": 0.2, "5HT": 0.1},
        )

        ok = bool(getattr(decision, "selected_action", None))
        _soft_assert(ok, "BG/Thalamus gating must return a selected_action for auditability")

        refs = (
            _code_ref(BasalGangliaThalamusGating, "BasalGangliaThalamusGating"),
            _code_ref(BasalGangliaThalamusGating.select_action, "BasalGangliaThalamusGating.select_action"),
        )

        details = f"selected_action={decision.selected_action}, threshold={decision.effective_threshold:.3f}, scores={decision.scores}"

        return AuditResult(
            mechanism_id="bg_thalamus_gating_action_selection",
            ok=ok,
            details=details,
            code_refs=refs,
        )
    except Exception as e:
        return AuditResult(
            mechanism_id="bg_thalamus_gating_action_selection",
            ok=False,
            details=f"missing_or_error={type(e).__name__}: {str(e)[:120]}",
            code_refs=tuple(),
        )


# ANCHOR: OUTPUT_FORMAT - printing helpers
# API_PRIVATE
def _print_result(res: AuditResult) -> None:
    """Print a single audit result.

    Description:
        Formats ✅/❌ status and includes code references.

    Intent:
        Provide human-readable, code-grounded audit output.

    Args:
        res: AuditResult.

    Returns:
        None.

    Raises:
        None.
    """
    status = "✅" if res.ok else "❌"
    print(f"{status} {res.mechanism_id}: {res.details}")
    for ref in res.code_refs:
        print(f"   - {ref}")


# ANCHOR: OUTPUT_MATRIX - neuromodulator influence matrix printing
# API_PRIVATE
def _print_influence_matrix(matrix: Mapping[str, Mapping[str, Sequence[str]]]) -> None:
    """Print neuromodulator influence matrix.

    Description:
        Prints where each neuromodulator enters and which variables/functions it changes.

    Intent:
        Make modulation points explicit and verifiable in code.

    Args:
        matrix: Influence matrix.

    Returns:
        None.

    Raises:
        None.
    """
    print("\nNEUROMODULATOR INFLUENCE MATRIX")
    print("-" * 70)
    for mod in ("DA", "ACh", "NE", "5HT"):
        enters = list(matrix.get(mod, {}).get("enters", []))
        changes = list(matrix.get(mod, {}).get("changes", []))
        print(f"{mod}:")
        print("  enters:")
        for e in enters:
            print(f"    - {e}")
        print("  changes:")
        for c in changes:
            print(f"    - {c}")


# ANCHOR: MAIN - entrypoint
# API_PUBLIC
def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the Brain mechanism audit.

    Description:
        Executes small audit scenes and prints ✅/❌ per mechanism with code pointers.

    Intent:
        Provide a single-command, code-grounded audit independent of documentation.

    Args:
        argv: Optional command-line args.

    Returns:
        Process exit code (0 if all ok, 1 otherwise).

    Raises:
        None.
    """
    _ensure_repo_on_path()

    results: List[AuditResult] = []

    # Registry↔code consistency checks first (prevents stale registry).
    results.extend(_audit_registry_vs_code())

    results.append(_audit_connection_states())
    results.append(_audit_inhibition_wta_wm_cap())
    results.append(_audit_hippocampus_encode_retrieve_replay())

    neuromod_res, matrix = _audit_neuromodulators_matrix()
    results.append(neuromod_res)

    results.append(_audit_stdp_and_usage_counters())
    results.append(_audit_bg_thalamus_gating())

    print("=" * 70)
    print("BRAIN AUDIT")
    print("=" * 70)

    for r in results:
        _print_result(r)

    _print_influence_matrix(matrix)

    all_ok = all(r.ok for r in results)

    # Postcondition
    _soft_assert(isinstance(all_ok, bool), "audit must produce a boolean overall verdict")

    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
