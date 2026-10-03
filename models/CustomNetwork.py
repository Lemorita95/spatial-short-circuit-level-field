import json
import numpy as np
from scipy.sparse import coo_matrix

from .Buses import Bus
from .Lines import Line
from .Capacitors import Capacitor
from .Transformers import Transformer
from .Generators import Generator
from .Loads import Load
from .FullConverter import FullConverter
from .ExternalGrid import ExternalGrid

def duplicates_error(key, element_dict):
    if key in element_dict:
        raise ValueError(f"duplicated name for '{key}'")

class CustomNetwork():
    """

    """
    def __init__(self, file):
        self.file = file

    def create_network(self):
        with open(self.file, "r", encoding="utf-8") as f:
            data = json.load(f)

            self.buses = {}
            self.slack_id = None

            # buses
            for bus in data.get("buses", []):

                # only record slack id, PV and PQ buses will be inferred on the fly
                if bus.get('bus_type', "").lower() == "slack":
                    if self.slack_id is not None:
                        raise ValueError("Multiple slack buses defined.")
                    self.slack_id = bus["id"]

                duplicates_error(bus.get("id"), self.buses)

                self.buses[bus.get("id")] = Bus(
                        id = bus.get("id"),
                        name = bus.get("name"),
                        bus_type = bus.get("bus_type"),
                        V = float(bus.get("V")) if bus.get("V") is not None else None,
                        angle = float(bus.get("angle")) if bus.get("angle") is not None else None
                        )
            
            # all elements
            elements = data.get("elements", {})

            # LINE element
            self.lines = {}
            for equipment in elements.get('lines', []):
                name = f'line.{equipment.get("name").lower()}'
                duplicates_error(name, self.lines)
                self.lines[name] = Line(
                    name = name,
                    bus_1 = equipment.get("from_bus"),
                    bus_2 = equipment.get("to_bus"),
                    Z_series = float((equipment.get('z_series') or {}).get('R', 0)) + 1j* float((equipment.get('z_series') or {}).get('X', 0)),
                    Y_shunt = float((equipment.get('y_shunt') or {}).get('G', 0)) + 1j* float((equipment.get('y_shunt') or {}).get('B', 0)),
                    status = bool(equipment.get("status", True)),
                )
            
            # CAPACITOR element
            self.capacitors = {}
            for equipment in elements.get('capacitors', []):
                name = f'capacitor.{equipment.get("name").lower()}'
                duplicates_error(name, self.capacitors)
                self.capacitors[name] = Capacitor(
                    name = name,
                    bus_1 = equipment.get("from_bus"),
                    bus_2 = equipment.get("to_bus", 0), # default value for shunt
                    Z_series = float((equipment.get('z_series') or {}).get('R', 0)) + 1j* float((equipment.get('z_series') or {}).get('X', 0)),
                    Y_shunt = float((equipment.get('y_shunt') or {}).get('G', 0)) + 1j* float((equipment.get('y_shunt') or {}).get('B', 0)),
                    status = bool(equipment.get("status", True)),
                )
            
            # TRANSFORMER element
            self.transformers = {}
            for equipment in elements.get('transformers', []):
                name = f'transformer.{equipment.get("name").lower()}'
                duplicates_error(name, self.transformers)
                self.transformers[name] = Transformer(
                    name = name,
                    bus_1 = equipment.get("from_bus"),
                    bus_2 = equipment.get("to_bus"),
                    Z_series = float((equipment.get('z_series') or {}).get('R', 0)) + 1j* float((equipment.get('z_series') or {}).get('X', 0)),
                    Y_shunt = float((equipment.get('y_shunt') or {}).get('G', 0)) + 1j* float((equipment.get('y_shunt') or {}).get('B', 0)),
                    m = float(equipment.get("m")) if equipment.get("m") is not None else 1.0,
                    status = bool(equipment.get("status", True)),
                )

            # GENERATOR element
            self.generators = {}
            for equipment in elements.get('generators', []):
                # add series element
                r_series = float(equipment['z_series']['R'])
                x_series = float(equipment['z_series']['X'])
                # the ideal voltage source equivalent
                z_series = float("inf") if r_series == float("inf") or x_series == float("inf") else r_series + 1j* x_series
                name = f'generator.{equipment.get("name").lower()}'
                duplicates_error(name, self.generators)
                self.generators[name] = Generator(
                    name = name,
                    bus_1 = equipment.get('from_bus'),
                    Z_series = z_series,
                    V = float(equipment.get("V")) if equipment.get("V") is not None else None,
                    P = float(equipment.get("P")) if equipment.get("P") is not None else None,
                    Q = float(equipment.get("Q")) if equipment.get("Q") is not None else None,
                    Q_max = float(equipment.get("Qg_max")) if equipment.get("Qg_max") is not None else None,
                    Q_min = float(equipment.get("Qg_min")) if equipment.get("Qg_min") is not None else None,
                    status = bool(equipment.get("status", True))
                )

            # LOAD element
            self.loads = {}
            for equipment in elements.get('loads', []):
                name = f'load.{equipment.get("name").lower()}'
                duplicates_error(name, self.loads)
                self.loads[name] = Load(
                    name = name,
                    bus_1 = equipment.get('from_bus'),
                    bus_2 = equipment.get('to_bus', 0), # default value for shunt
                    P = 0.0 if equipment.get("P") is None else float(equipment.get("P")),
                    Q = 0.0 if equipment.get("Q") is None else float(equipment.get("Q")),
                    status = bool(equipment.get("status", True))
                )

            # FULL CONVERTER element
            self.full_converters = {}
            for equipment in elements.get('full_converters', []):
                name = f'full_converter.{equipment.get("name").lower()}'
                duplicates_error(name, self.full_converters)
                self.full_converters[name] = FullConverter(
                    name = name,
                    bus_1 = equipment.get('from_bus'),
                    P = None if equipment.get("P") is None else float(equipment.get("P")),
                    Q = None if equipment.get("Q") is None else float(equipment.get("Q")),
                    kVA_nominal = float(equipment.get("kVA_nominal")) if equipment.get("kVA_nominal") is not None else None,
                    cos_phi = float(equipment.get("cos_phi")) if equipment.get("cos_phi") is not None else None,
                    kV_nominal = float(equipment.get("kV_nominal")) if equipment.get("kV_nominal") is not None else None,
                    K_scc = 1.2 if equipment.get("K_scc") is None else float(equipment.get("K_scc")),
                    V_min = 0.15 if equipment.get("V_min") is None else float(equipment.get("V_min")),
                    K_factor = 3.0 if equipment.get("K_factor") is None else float(equipment.get("K_factor")),
                    status = bool(equipment.get("status", True))
                )

            # EXTERNAL GRID element
            self.external_grids = {}
            for equipment in elements.get('external_grids', []):
                # add series element
                r_series, x_series = equipment.get('Z_SC_ohms',{}).get('R'), equipment.get('Z_SC_ohms',{}).get('X')
                name = f'external_grid.{equipment.get("name").lower()}'
                duplicates_error(name, self.external_grids)
                self.external_grids[name] = ExternalGrid(
                    name = name,
                    bus_1 = equipment.get('from_bus'),
                    kV_nom = equipment.get('kV_nom'),
                    Z_SC_ohms = (float(r_series) + 1j * float(x_series)) if r_series is not None and x_series is not None else None,
                    S_SC_MVA = equipment.get('S_SC_MVA'),
                    I_SC_kamps = equipment.get('I_SC_kamps'),
                    status = bool(equipment.get("status", True))
                )
                    
    def buses_type(self):
        PV_id = set()

        for gen in self.generators.values():
            # if generator is active and not slack
            if gen.status and gen.bus_1 != self.slack_id:
                PV_id.add(gen.bus_1)

        PQ_id = set(self.buses.keys()) - {self.slack_id} - PV_id

        return PQ_id, PV_id
                    
    def compute_bus_net_power(self):
        # initialize buses
        for bus_values in self.buses.values():
            bus_values.P = 0
            bus_values.Q = 0
            bus_values.Qg_min = None
            bus_values.Qg_max = None

        # add generator contributions
        for gen in self.generators.values():
            if not gen.status:
                continue
            bus = self.buses[gen.bus_1]
            if gen.P is not None:
                bus.P += gen.P
            if gen.Q is not None:
                bus.Q += gen.Q
            if gen.Q_min is not None:
                bus.Qg_min = (bus.Qg_min or 0) + gen.Q_min
            if gen.Q_max is not None:
                bus.Qg_max = (bus.Qg_max or 0) + gen.Q_max

        # subtract load contributions
        for load in self.loads.values():
            if not load.status:
                continue
            bus = self.buses[load.bus_1]
            if load.P is not None:
                bus.P -= load.P
            if load.Q is not None:
                bus.Q -= load.Q

        # add converter contributions (negative load)
        for conv in self.full_converters.values():
            if not conv.status:
                continue
            bus = self.buses[conv.bus_1]
            if conv.P is not None:
                bus.P += conv.P
            if conv.Q is not None:
                bus.Q += conv.Q

    def build_ybus(self):
        '''
            build Y bus matrix from the edges (loop each element only once)
        '''
        self.idx = {b: i for i, b in enumerate(self.buses.keys())} # bus number -> python index
        self.id = {i: b for i, b in enumerate(self.buses.keys())} # python index <-> bus number
        n = len(self.buses)
        
        rows = []
        cols = []
        Ydata = []

        '''
            the following elements are added individually to control each ones 
            i want to add to the base Y matrix (power flow)
        '''
        # add each line element
        for line in self.lines.values():
            
            # if status is False (element is off)
            if not line.status:
                continue

            r, c, d = line.get_Y()

            rows.extend(r)
            cols.extend(c)
            Ydata.extend(d)

        # add each capacitor element
        for cap in self.capacitors.values():
            
            # if status is False (element is off)
            if not cap.status:
                continue

            r, c, d = cap.get_Y()

            rows.extend(r)
            cols.extend(c)
            Ydata.extend(d)

        # add each trafo element
        for trafo in self.transformers.values():
            
            # if status is False (element is off)
            if not trafo.status:
                continue
            
            r, c, d = trafo.get_Y()

            rows.extend(r)
            cols.extend(c)
            Ydata.extend(d)

        rows = [self.idx[id] for id in rows] # python index
        cols = [self.idx[id] for id in cols] # python index
        
        self.YBus = coo_matrix((Ydata, (rows, cols)), shape=(n, n)).tocsr()

    def get_active_elements(self, V_complex, active_element_flow, element_status):
        '''
            duplicate keys are already handle in create_network
            (name, bus_idx): (element, v_pre, s_pre)
        '''
        elements = self.loads | self.generators | self.full_converters | self.external_grids

        return {
            (full_name, self.idx[element.bus_1]): (element, V_complex[self.idx[element.bus_1]], active_element_flow[(full_name, self.idx[element.bus_1])])
            for full_name, element in elements.items() 
            if element_status[full_name]}
    
    def change_element_status(self, full_name, status: np.bool):
        '''
            change one generator active power setpoint
        '''
        assert isinstance(status, (np.bool, bool)), "status must be a boolean"

        elements = self.capacitors | self.lines | self.transformers | \
            self.loads | self.generators | self.full_converters | self.external_grids

        elements[full_name.lower()].status = status

