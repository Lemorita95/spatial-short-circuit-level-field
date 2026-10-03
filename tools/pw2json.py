import argparse
import csv
import json
from pathlib import Path
from typing import Optional

S_BASE_MVA = 100.0
TRANSFORMER_PHASE_SHIFT_DEG = 0.0


def read_pw_csv(path: Path) -> list[dict]:
    """Read a PowerWorld CSV export whose first line contains the object type."""
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        first = f.readline()
        if not first:
            raise ValueError(f"{path} is empty.")
        return list(csv.DictReader(f))


def pw_float(value: str | None) -> Optional[float]:
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    return float(s.replace(",", "."))


def pw_int(value: str) -> int:
    return int(str(value).strip())


def closed(value: str) -> bool:
    return str(value).strip().lower() == "closed"


def pu_power(value: str | None) -> float:
    v = pw_float(value)
    return 0.0 if v is None else v / S_BASE_MVA


def convert(folder: Path) -> tuple[dict, str]:
    buses_csv = read_pw_csv(folder / "buses.csv")
    branches_csv = read_pw_csv(folder / "branches.csv")
    generators_csv = read_pw_csv(folder / "generators.csv")
    loads_csv = read_pw_csv(folder / "loads.csv")
    line_shunts_csv = read_pw_csv(folder / "line_shunts.csv")
    switched_shunts_csv = read_pw_csv(folder / "switched_shunts.csv")

    notes: list[str] = []
    warnings: list[str] = []

    # ------------------------------------------------------------------ buses
    slack_rows = [r for r in buses_csv
                  if r.get("Slack", "").strip().upper() == "YES"
                  or r.get("BusCat", "").strip().lower() == "slack"]
    if len(slack_rows) != 1:
        raise ValueError(
            f"Expected exactly one slack bus from buses.csv; found {len(slack_rows)}."
        )
    slack_id = pw_int(slack_rows[0]["Number"])

    buses = []
    for row in buses_csv:
        bus_id = pw_int(row["Number"])
        item = {
            "id": bus_id,
            "name": row["Name"].strip(),
        }
        # Match the existing executable case schema: only the slack bus needs
        # an explicit type / starting voltage / angle.
        if bus_id == slack_id:
            item.update({
                "bus_type": "Slack",
                "V": pw_float(row["Vpu"]),
                "angle": pw_float(row["Vangle"]),
            })
        buses.append(item)

    # ----------------------------------------------------- lines/transformers
    lines = []
    transformers = []
    omitted_open_branches = []
    active_taps = []

    for row in branches_csv:
        fbus = pw_int(row["BusNumFrom"])
        tbus = pw_int(row["BusNumTo"])
        ckt = row["Circuit"].strip()
        dev = row["BranchDeviceType"].strip()

        if not closed(row["Status"]):
            omitted_open_branches.append((fbus, tbus, ckt, dev))
            continue

        R = pw_float(row["R"]) or 0.0
        X = pw_float(row["X"]) or 0.0
        G = pw_float(row["G"]) or 0.0
        B = pw_float(row["B"]) or 0.0
        tap = pw_float(row["Tap"])
        if tap is None:
            tap = 1.0

        common = {
            "from_bus": fbus,
            "to_bus": tbus,
            "z_series": {"R": R, "X": X},
            "y_shunt": {"G": G, "B": B},
        }

        if row["IsXF"].strip().upper() == "YES":
            item = {
                **common,
                "name": f"Transformer {fbus}-{tbus} ckt {ckt}",
                "m": tap,
            }
            transformers.append(item)
            active_taps.append((fbus, tbus, ckt, tap))
        else:
            item = {
                **common,
                "name": f"Line {fbus}-{tbus} ckt {ckt}",
            }
            lines.append(item)

    # ---------------------------------------------------------------- loads
    loads = []
    omitted_open_loads = []
    for row in loads_csv:
        bus = pw_int(row["BusNum"])
        load_id = row["ID"].strip()
        if not closed(row["Status"]):
            omitted_open_loads.append((bus, load_id))
            continue
        loads.append({
            "from_bus": bus,
            "to_bus": 0,
            "name": f"Load {bus}-{load_id}",
            "P": pu_power(row["MW"]),
            "Q": pu_power(row["Mvar"]),
        })

    # ----------------------------------------------------------- generators
    generators = []
    generator_impedance_details = []

    for row in generators_csv:
        bus = pw_int(row["BusNum"])
        gid = row["ID"].strip()
        mva_base = pw_float(row["MVABase"])
        if not mva_base or mva_base <= 0:
            raise ValueError(f"Invalid MVABase for generator {bus}-{gid}.")

        gen_r_machine = pw_float(row["GenR"]) or 0.0
        gen_x_machine = pw_float(row["GenX"]) or 0.0
        step_r_raw = pw_float(row["StepR"]) or 0.0
        step_x_raw = pw_float(row["StepX"]) or 0.0
        step_tap = pw_float(row["StepTap"])
        if step_tap is None:
            step_tap = 1.0

        # GenR/GenX are on generator MVA base. Convert to system base.
        base_factor = S_BASE_MVA / mva_base
        gen_r_system = gen_r_machine * base_factor
        gen_x_system = gen_x_machine * base_factor

        # For this supplied case, all StepTap values are 1.0 and generator MVA
        # bases equal the 100 MVA system base. Therefore folding StepR/StepX
        # into the series generator impedance is unambiguous numerically.
        #
        # Refuse silent generalization if a future case has non-unit StepTap.
        if abs(step_tap - 1.0) > 1e-12 and (abs(step_r_raw) > 0 or abs(step_x_raw) > 0):
            raise ValueError(
                f"Generator {bus}-{gid} has non-unit StepTap={step_tap} with "
                "nonzero internal step-up impedance. Define the required base/"
                "tap transformation before using this converter for that case."
            )

        # With S_gen = S_system = 100 MVA here, this is a direct sum.
        # If generator MVABase differs in a future case, verify the PowerWorld
        # base convention for StepR/StepX before generalizing this line.
        if abs(mva_base - S_BASE_MVA) > 1e-9 and (abs(step_r_raw) > 0 or abs(step_x_raw) > 0):
            raise ValueError(
                f"Generator {bus}-{gid} has MVABase={mva_base} MVA and nonzero "
                "StepR/StepX. Verify the step-up impedance base before conversion."
            )

        r_eq = gen_r_system + step_r_raw
        x_eq = gen_x_system + step_x_raw

        generators.append({
            "from_bus": bus,
            "to_bus": 0,
            "name": f"G{bus}-{gid}",
            "z_series": {"R": r_eq, "X": x_eq},
            "y_shunt": None,
            "V": pw_float(row["VoltSet"]),
            "P": pu_power(row["MW"]),
            "Q": None,
            "Qg_max": pu_power(row["MvarMax"]),
            "Qg_min": pu_power(row["MvarMin"]),
            "status": 1 if closed(row["Status"]) else 0,
        })

        generator_impedance_details.append({
            "bus": bus,
            "id": gid,
            "MVABase": mva_base,
            "GenR": gen_r_machine,
            "GenX": gen_x_machine,
            "StepR": step_r_raw,
            "StepX": step_x_raw,
            "StepTap": step_tap,
            "R_eq_system": r_eq,
            "X_eq_system": x_eq,
        })

    # --------------------------------------------------------------- shunts
    # The executable's "capacitors" list is structurally a one-terminal shunt
    # admittance, so it is used for fixed bus/line-end/switched shunts.
    capacitors = []

    # End-connected line shunts.
    for row in line_shunts_csv:
        if not closed(row["Status"]):
            continue

        fbus = pw_int(row["BusNumFrom"])
        tbus = pw_int(row["BusNumTo"])
        loc = pw_int(row["BusNumLoc"])
        ckt = row["Circuit"].strip()
        sid = row["ID"].strip()

        capacitors.append({
            "from_bus": loc,
            "to_bus": 0,
            "name": f"LineShunt {fbus}-{tbus} ckt {ckt} ID {sid} @ {loc}",
            "z_series": None,
            "y_shunt": {
                "G": pu_power(row["MWNom"]),
                "B": pu_power(row["MvarNom"]),
            },
            "status": 1,
        })

    # Switched shunts are intentionally frozen at the nominal Mvar of the
    # EXPORTED switching position. We use MvarNom, not instantaneous Mvar
    # (which contains the V^2 effect) and not MvarNomMax.
    fixed_switched_shunts = []
    for row in switched_shunts_csv:
        bus = pw_int(row["BusNum"])
        sid = row["ID"].strip()
        q_nom = pw_float(row["MvarNom"]) or 0.0
        status = 1 if closed(row["Status"]) else 0

        capacitors.append({
            "from_bus": bus,
            "to_bus": 0,
            "name": f"Switched Shunt {bus}-{sid} [FIXED EXPORTED POSITION]",
            "z_series": None,
            "y_shunt": {
                "G": 0.0,
                "B": q_nom / S_BASE_MVA,
            },
            "status": status,
        })

        fixed_switched_shunts.append({
            "bus": bus,
            "id": sid,
            "status": row["Status"].strip(),
            "mode_in_powerworld": row["ShuntMode"].strip(),
            "auto_control_in_powerworld": row["AutoControl"].strip(),
            "MvarNom_exported_position": q_nom,
            "B_pu_fixed_json": q_nom / S_BASE_MVA,
        })

    case = {
        "buses": buses,
        "elements": {
            "lines": lines,
            "transformers": transformers,
            "capacitors": capacitors,
            "generators": generators,
            "loads": loads,
            "full_converters": [],
            "external_grids": [],
        },
    }

    # --------------------------------------------------------------- report
    report = []
    report.append("# PowerWorld → SCC JSON conversion report (v2)\n\n")
    report.append("## Case-level assumptions\n\n")
    report.append(f"- System base: **{S_BASE_MVA:g} MVA**.\n")
    report.append(f"- Slack bus read from CSV: **{slack_id}**.\n")
    report.append(
        f"- Transformer phase shift: **{TRANSFORMER_PHASE_SHIFT_DEG:g}° for all transformers** "
        "(user-verified for this case).\n"
    )
    report.append(
        "- PowerWorld automatic switched-shunt and transformer-tap controls are **not assumed frozen**. "
        "The JSON contains the exported snapshot values. If PowerWorld controls move in another scenario, "
        "the resulting model states should be recorded when comparing results.\n"
    )

    report.append("\n## Schema audit\n\n")
    report.append(
        "**No required field in the current executable JSON schema remains missing for this 37-bus case.**\n\n"
    )
    report.append(
        "- Branch/transformer: R, X, G, B, tap, terminals, circuit and status are available.\n"
    )
    report.append(
        "- Generator: status, MVABase, GenR, GenX, StepR, StepX, StepTap, P, voltage setpoint and Q limits are available.\n"
    )
    report.append(
        "- Bus: explicit Slack/BusCat fields are available; the JSON continues to mark only the slack bus explicitly, "
        "matching the existing 14-bus case structure.\n"
    )
    report.append(
        "- Load: P, Q and status are available.\n"
    )

    report.append("\n## Generator equivalent impedances\n\n")
    report.append(
        "`GenR/GenX` are converted from generator MVA base to the 100-MVA system base, then the internal "
        "step-up `StepR/StepX` is included in the equivalent series impedance for this case. "
        "All generator MVA bases and StepTap values in the supplied case are 100 MVA and 1.0 respectively.\n\n"
    )
    report.append("| Bus-ID | GenR | GenX | StepR | StepX | R_eq | X_eq |\n")
    report.append("|---|---:|---:|---:|---:|---:|---:|\n")
    for d in generator_impedance_details:
        report.append(
            f"| {d['bus']}-{d['id']} | {d['GenR']:.5f} | {d['GenX']:.5f} | "
            f"{d['StepR']:.5f} | {d['StepX']:.5f} | "
            f"{d['R_eq_system']:.5f} | {d['X_eq_system']:.5f} |\n"
        )

    report.append("\n## Transformer exported tap states\n\n")
    report.append(
        "These are the tap ratios contained in the exported snapshot. PowerWorld may move controllable taps "
        "when solving another operating condition.\n\n"
    )
    report.append("| From | To | Ckt | Tap m |\n")
    report.append("|---:|---:|:---:|---:|\n")
    for fbus, tbus, ckt, tap in active_taps:
        report.append(f"| {fbus} | {tbus} | {ckt} | {tap:.5f} |\n")

    report.append("\n## Switched shunts represented as fixed shunts\n\n")
    report.append(
        "The JSON freezes each shunt at the **exported `MvarNom` switching position**:\n\n"
        "\\[\nB_{pu}=\\frac{Q_{nom}}{100\\ \\mathrm{MVA}}\n\\]\n\n"
        "It does not reproduce PowerWorld's switching controller. `Mvar` is not used because it already reflects "
        "the local-voltage squared effect; `MvarNomMax` is not used because it is the available maximum rather than "
        "the exported switching position.\n\n"
    )
    report.append("| Bus-ID | PW mode | PW auto | MvarNom | Fixed B pu |\n")
    report.append("|---|---|---|---:|---:|\n")
    for d in fixed_switched_shunts:
        report.append(
            f"| {d['bus']}-{d['id']} | {d['mode_in_powerworld']} | "
            f"{d['auto_control_in_powerworld']} | {d['MvarNom_exported_position']:.3f} | "
            f"{d['B_pu_fixed_json']:.5f} |\n"
        )

    report.append("\n## Open elements omitted from active topology\n\n")
    if omitted_open_branches:
        for x in omitted_open_branches:
            report.append(f"- Branch: {x[3]} {x[0]} → {x[1]}, circuit {x[2]}.\n")
    else:
        report.append("- No open branches.\n")
    if omitted_open_loads:
        for x in omitted_open_loads:
            report.append(f"- Load: bus {x[0]}, ID {x[1]}.\n")
    else:
        report.append("- No open loads.\n")

    report.append("\n## Comparison caveat\n\n")
    report.append(
        "If PowerWorld automatic tap or switched-shunt controls move after a scenario change while the SCC JSON "
        "continues to use an earlier exported position, the two short-circuit models no longer have identical "
        "network admittances. Report the resulting tap/shunt-state differences alongside the power-flow mismatch; "
        "do not interpret the full SCC discrepancy as solver error alone.\n"
    )

    return case, "".join(report)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", nargs="?", default=".")
    parser.add_argument("-o", "--output", default="37bus_case.json")
    parser.add_argument("--report", default="37bus_conversion_report_v2.md")
    args = parser.parse_args()

    case, report = convert(Path(args.folder))
    Path(args.output).write_text(
        json.dumps(case, indent=4, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    Path(args.report).write_text(report, encoding="utf-8")
    print(f"Wrote: {args.output}")
    print(f"Wrote: {args.report}")


if __name__ == "__main__":
    main()
