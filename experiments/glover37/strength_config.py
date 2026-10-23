"""In-memory Glover 37-bus IBR augmentation (Gate 3).

All branch impedances are on the 100 MVA system base. Transformer per-unit
nominal taps are unity because the 69/34.5/0.69 kV voltage bases follow the
respective bus sides. The existing case file is not modified.
"""
from dataclasses import dataclass

from models.Buses import Bus
from models.FullConverter import FullConverter
from models.Lines import Line
from models.Transformers import Transformer

SYSTEM_BASE_MVA = 100.0
HV_KV, MV_KV, LV_KV = 69.0, 34.5, 0.69
TRANSFORMER_Z_EQUIPMENT_BASE = 0.005 + 0.060j
FEEDER_Z_SYSTEM_BASE = 0.011 + 0.027j
FEEDER_B_SYSTEM_BASE = 0.069
IBR_MVA_MULTIPLIER = 1.1
TRANSFORMER_MVA_MULTIPLIER = 1.2
Q_CAPABILITY_MULTIPLIER = 0.436


@dataclass(frozen=True)
class IBRSite:
    host_bus: int
    pmax_mw: float
    operating_mw: float
    main_mv_bus: int
    padmount_hv_bus: int
    converter_lv_bus: int

    @property
    def poi_bus(self) -> int:
        """IEEE 2800 evaluation point (original Glover HV connection bus)."""
        return self.host_bus

    @property
    def escr_bus(self) -> int:
        """Fingrid-inspired ESCR evaluation point (padmount HV, 34.5 kV)."""
        return self.padmount_hv_bus

    @property
    def sdscr_bus(self) -> int:
        """Wu (2018) evaluation point (converter LV terminal)."""
        return self.converter_lv_bus

    @property
    def converter_mva(self) -> float:
        return IBR_MVA_MULTIPLIER * self.pmax_mw

    @property
    def transformer_mva(self) -> float:
        return TRANSFORMER_MVA_MULTIPLIER * self.converter_mva

    @property
    def q_limit_mvar(self) -> float:
        return Q_CAPABILITY_MULTIPLIER * self.converter_mva

    @property
    def transformer_z_pu(self) -> complex:
        return TRANSFORMER_Z_EQUIPMENT_BASE * (SYSTEM_BASE_MVA / self.transformer_mva)


# Two MV buses + one LV bus at each site. Bus IDs do not overlap Glover IDs.
SITES = (
    IBRSite(21, 60.0, 48.0, 2101, 2102, 2103),
    IBRSite(48, 45.0, 36.0, 4801, 4802, 4803),
    IBRSite(16, 45.0, 36.0, 1601, 1602, 1603),
)


def augment_network(net):
    """Augment an initialized CustomNetwork, rebuild YBus, return IBR sites.

    padmount_hv_bus is the 34.5 kV ESCR evaluation bus.
    IEEE SCR/WSCR use poi_bus; SDSCR uses sdscr_bus.
    This function expects net.create_network() to have run already.
    """
    all_new_ids = [bus for site in SITES for bus in
                   (site.main_mv_bus, site.padmount_hv_bus, site.converter_lv_bus)]
    if len(set(all_new_ids)) != len(all_new_ids):
        raise ValueError("Duplicate IBR bus IDs in configuration")
    if any(site.host_bus not in net.buses for site in SITES):
        raise ValueError("One or more original Glover host buses are missing")
    if any(bus in net.buses for bus in all_new_ids):
        raise ValueError("IBR bus IDs already exist; network may be augmented")

    for site in SITES:
        for bus_id, label, nom_kv in (
            (site.main_mv_bus, "COLLECTOR_MV", MV_KV),
            (site.padmount_hv_bus, "PADMOUNT_HV", MV_KV),
            (site.converter_lv_bus, "CONVERTER_LV", LV_KV),
        ):
            net.buses[bus_id] = Bus(bus_id, f"IBR{site.host_bus}_{label}", "PQ", None, None,
                                    nom_kv=nom_kv)

        main = f"transformer.ibr{site.host_bus}_main"
        feeder = f"line.ibr{site.host_bus}_collector"
        pad = f"transformer.ibr{site.host_bus}_padmount"
        converter = f"full_converter.ibr{site.host_bus}"
        if (main in net.transformers or pad in net.transformers or
                feeder in net.lines or converter in net.full_converters):
            raise ValueError(f"Duplicate equipment name at host {site.host_bus}")

        z = site.transformer_z_pu
        # On corresponding voltage bases, nominal transformer tap ratio is 1 pu.
        # No reg_bus: IBR transformer taps stay at n0 = 0 (m = m0 = 1.0) in this study.
        net.transformers[main] = Transformer(main, site.host_bus, site.main_mv_bus, z)
        net.lines[feeder] = Line(feeder, site.main_mv_bus, site.padmount_hv_bus,
                                 FEEDER_Z_SYSTEM_BASE, Y_shunt=1j * FEEDER_B_SYSTEM_BASE)
        net.transformers[pad] = Transformer(pad, site.padmount_hv_bus, site.converter_lv_bus, z)
        # FullConverter P, Q are per unit; rating is kVA, nominal voltage is kV.
        net.full_converters[converter] = FullConverter(
            name=converter, bus_1=site.converter_lv_bus,
            kVA_nominal=site.converter_mva * 1000.0, cos_phi=1.0, kV_nominal=LV_KV,
            P=site.operating_mw / SYSTEM_BASE_MVA, Q=0.0,
        )

    net.build_ybus()
    return SITES
