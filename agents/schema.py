"""
Canonical flowsheet JSON schema.
All agents read and write this format. The executor converts it to DWSIM API calls.

Full example:
{
    "compounds": ["Ethanol", "Water"],
    "property_package": "NRTL",
    "streams": [
        {
            "tag": "FEED",
            "T": 298.15, "P": 101325.0, "flow": 2.0,
            "composition": {"Ethanol": 0.6, "Water": 0.4}
        },
        {"tag": "HOT"},
        {"tag": "VAP"},
        {"tag": "LIQ"}
    ],
    "units": [
        {
            "tag": "HT-01", "type": "Heater",
            "T_out": 353.0, "dP": 0.0,
            "property_package": "Raoult's Law"   ← optional per-unit override
        },
        {
            "tag": "V-01", "type": "Vessel",
            "dP": 0.0,
            "property_package": "NRTL"           ← optional per-unit override
        }
    ],
    "connections": [
        ["FEED",  "HT-01", 0, 0],
        ["HT-01", "HOT",   0, 0],
        ["HOT",   "V-01",  0, 0],
        ["V-01",  "VAP",   0, 0],
        ["V-01",  "LIQ",   1, 0]
    ]
}

Unit type parameter reference:
  Heater     : T_out [K], dP [Pa]
  Cooler     : T_out [K], dP [Pa]
  Vessel     : dP [Pa]
  Mixer      : dP [Pa]
  Splitter   : split_fractions {stream_tag: fraction}, dP [Pa]
  Pump       : P_out [Pa], efficiency [0–1]
  Compressor : P_out [Pa], efficiency [0–1]
  Expander   : P_out [Pa], efficiency [0–1]

Connection format: [src_tag, dst_tag, src_port, dst_port]
  src_port 0 = primary / vapour outlet
  src_port 1 = secondary / liquid outlet
  dst_port   = almost always 0

Property package selection guidance (for Thermodynamics Agent):
  Ideal / dilute systems        → Raoult's Law
  Polar non-ideal liquids       → NRTL, UNIQUAC
  Azeotropic distillation       → NRTL, UNIQUAC
  Liquid–liquid extraction      → UNIQUAC, NRTL
  High-pressure gas/vapour      → Peng-Robinson, Soave-Redlich-Kwong
  Light hydrocarbons            → Peng-Robinson, Lee-Kesler-Plöcker
  Electrolytes / salts          → Electrolyte NRTL (future)
  Mixed polar + non-polar       → NRTL (liquid) + PR (vapour)
"""

from __future__ import annotations
import json


SUPPORTED_UNIT_TYPES = {
    "Heater", "Cooler", "Vessel", "Mixer",
    "Splitter", "Pump", "Compressor", "Expander",
}

SUPPORTED_PROPERTY_PACKAGES = [
    "Raoult's Law",
    "NRTL",
    "UNIQUAC",
    "Peng-Robinson",
    "Soave-Redlich-Kwong",
    "Lee-Kesler-Plöcker",
]


def validate(flowsheet: dict) -> list[str]:
    """Return a list of validation errors (empty list = valid)."""
    errors = []

    compounds = flowsheet.get("compounds", [])
    if not compounds:
        errors.append("'compounds' list is empty or missing.")

    default_pp = flowsheet.get("property_package")
    if not default_pp:
        errors.append("'property_package' (default) is missing.")
    elif default_pp not in SUPPORTED_PROPERTY_PACKAGES:
        errors.append(
            f"Default property package '{default_pp}' is not supported. "
            f"Choose from: {SUPPORTED_PROPERTY_PACKAGES}")

    stream_tags = {s["tag"] for s in flowsheet.get("streams", [])}
    unit_tags   = {u["tag"] for u in flowsheet.get("units", [])}
    all_tags    = stream_tags | unit_tags

    for s in flowsheet.get("streams", []):
        if "tag" not in s:
            errors.append("A stream is missing its 'tag'.")
        comp = s.get("composition", {})
        if comp:
            total = sum(comp.values())
            if abs(total - 1.0) > 0.01:
                errors.append(
                    f"Stream '{s.get('tag')}' composition sums to "
                    f"{total:.4f}, not 1.0.")
            for name, frac in comp.items():
                if name not in compounds:
                    errors.append(
                        f"Stream '{s.get('tag')}' references compound "
                        f"'{name}' not in 'compounds' list.")
                if not isinstance(frac, (int, float)) or frac < 0.0:
                    errors.append(
                        f"Stream '{s.get('tag')}' compound '{name}' has "
                        f"invalid mole fraction {frac} — must be ≥ 0.")
        flow = s.get("flow")
        if flow is not None and flow <= 0:
            errors.append(
                f"Stream '{s.get('tag')}' flow={flow} is not positive.")

    for u in flowsheet.get("units", []):
        tag = u.get("tag", "<unnamed>")
        if "tag" not in u:
            errors.append("A unit is missing its 'tag'.")
        if u.get("type") not in SUPPORTED_UNIT_TYPES:
            errors.append(
                f"Unit '{tag}' has unsupported type '{u.get('type')}'. "
                f"Choose from: {sorted(SUPPORTED_UNIT_TYPES)}")
        pp = u.get("property_package")
        if pp and pp not in SUPPORTED_PROPERTY_PACKAGES:
            errors.append(
                f"Unit '{tag}' property_package '{pp}' is not supported.")

    # Track (unit, src_port) pairs to detect duplicate port assignments
    _unit_src_ports: dict[tuple[str, int], str] = {}

    for conn in flowsheet.get("connections", []):
        if len(conn) < 2:
            errors.append(f"Connection {conn} needs at least [src, dst].")
            continue
        src, dst = conn[0], conn[1]
        if src not in all_tags:
            errors.append(f"Connection source '{src}' not in streams or units.")
        if dst not in all_tags:
            errors.append(f"Connection destination '{dst}' not in streams or units.")

        # Duplicate src_port: two outlets from the same unit on the same port
        # causes DWSIM to leave one stream unconnected → ZERO_OUTLET or wrong flow.
        if src in unit_tags and len(conn) >= 3:
            src_port = conn[2]
            key = (src, src_port)
            if key in _unit_src_ports:
                errors.append(
                    f"Unit '{src}' has two connections using src_port={src_port} "
                    f"(to '{_unit_src_ports[key]}' and '{dst}'). "
                    "Each outlet port may only be used once.")
            else:
                _unit_src_ports[key] = dst

    # Acyclicity check — recycle loops cause silent DWSIM solver divergence.
    # Build a directed graph over all nodes and run DFS-based cycle detection.
    connections = flowsheet.get("connections", [])
    if len(connections) >= 2:
        adj: dict[str, list[str]] = {t: [] for t in all_tags}
        for conn in connections:
            if len(conn) >= 2 and conn[0] in all_tags and conn[1] in all_tags:
                adj[conn[0]].append(conn[1])

        visited: set[str] = set()
        in_stack: set[str] = set()

        def _has_cycle(node: str) -> bool:
            visited.add(node)
            in_stack.add(node)
            for neighbour in adj.get(node, []):
                if neighbour not in visited:
                    if _has_cycle(neighbour):
                        return True
                elif neighbour in in_stack:
                    return True
            in_stack.discard(node)
            return False

        for node in all_tags:
            if node not in visited and _has_cycle(node):
                errors.append(
                    "Connection graph contains a cycle (recycle loop). "
                    "Only open-loop (feed-forward, acyclic) flowsheets are supported.")
                break

    # binary_parameters — optional; validate if present
    for i, bp in enumerate(flowsheet.get("binary_parameters", [])):
        tag = f"binary_parameters[{i}]"
        if bp.get("model") not in ("NRTL", "UNIQUAC"):
            errors.append(f"{tag}: 'model' must be 'NRTL' or 'UNIQUAC'.")
        for key in ("compound_a", "compound_b"):
            if bp.get(key) not in compounds:
                errors.append(f"{tag}: '{key}' value '{bp.get(key)}' not in 'compounds'.")
        for fld in ("A12", "A21"):
            if not isinstance(bp.get(fld), (int, float)):
                errors.append(f"{tag}: '{fld}' must be a number.")
        if bp.get("model") == "NRTL" and not isinstance(bp.get("alpha12"), (int, float)):
            errors.append(f"{tag}: 'alpha12' is required for NRTL and must be a number.")
        t_min = bp.get("T_min_K")
        t_max = bp.get("T_max_K")
        if t_min is not None and t_max is not None and t_min >= t_max:
            errors.append(f"{tag}: T_min_K ({t_min}) must be less than T_max_K ({t_max}).")

    return errors


def physics_validate(flowsheet: dict):
    """Run static physics compatibility checks. Returns list of PhysicsIssue."""
    from agents.physics_check import physics_validate as _pv
    return _pv(flowsheet)


def to_json(flowsheet: dict, indent: int = 2) -> str:
    return json.dumps(flowsheet, indent=indent)


def from_json(text: str) -> dict:
    """Parse JSON, stripping markdown code fences if present."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(
            line for line in lines
            if not line.strip().startswith("```")
        )
    return json.loads(text)


def get_unit_property_package(unit: dict, flowsheet: dict) -> str:
    """Return the effective property package for a unit (override or default)."""
    return unit.get("property_package") or flowsheet.get("property_package")
