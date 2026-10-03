import numpy as np
from scipy.sparse import coo_matrix, csc_matrix
from scipy.sparse.linalg import splu
from scipy.sparse import diags

class PowerFlow():
    '''
        run the newton-raphson algorithm to solve the non-linear power flow.

        The power flow solution is not necessary for the SSC methodology, however it gives
        the flexibility of changing the bus voltage other then assuming values.

        it uses the CustomNetwork object to represent the topology of the system
        and the JacobianBuilder object to manage the jacobian values update at each iteration.

        - solve_power_flow() method returns the computed complex voltages (module and angle) when convergence
        - PQ() method compute the bus power injections given the network topology and bus complex voltages
        - ...
    '''

    def PQ(self, YBus, ThetaV):
        n = len(ThetaV) // 2
        theta = ThetaV[:n]
        V = ThetaV[n:]

        P = np.zeros(n)
        Q = np.zeros(n)

        YBus = YBus.tocsr()

        for k in range(n):
            row_start = YBus.indptr[k]
            row_end = YBus.indptr[k + 1]

            # get non-zero entries of k-th row
            cols = YBus.indices[row_start:row_end]
            Y_vals = YBus.data[row_start:row_end]

            G = Y_vals.real
            B = Y_vals.imag

            theta_diff = theta[k] - theta[cols]

            P[k] = V[k] * np.sum(V[cols] * (G * np.cos(theta_diff) + B * np.sin(theta_diff)))
            Q[k] = V[k] * np.sum(V[cols] * (G * np.sin(theta_diff) - B * np.cos(theta_diff)))

        return P, Q

    def newton_raphson(self, ThetaV0, PQVec, YBus, slack_idx, PV_idx, Q_load, Qmin, Qmax, max_iter=100, tol=1e-4):
        '''
            Newton-Raphson method for solving nonlinear equations
            solve for dVTheta at a:  J . dVTheta = mismatch (P_target - P_calculated)
        '''
        print('\t\tSolving Power Flow using Newton-Raphson method...')
        convergence = False
        n_bus = YBus.shape[0]
        ThetaV = ThetaV0.copy()
        PV_active = np.ones(len(PV_idx), dtype=bool)  # active PV buses

        # unknown indices
        theta_unknown = np.delete(np.arange(n_bus), slack_idx) # exclude slack
        exclude = np.concatenate(([slack_idx], PV_idx))
        V_unknown = np.delete(np.arange(n_bus), exclude) # exclude slack and PV
        unknown_indices = np.concatenate([theta_unknown, n_bus + V_unknown])

        iteration = 0
        while iteration <= max_iter:
            iteration += 1
            P_calc, Q_calc = self.PQ(YBus, ThetaV)

            mismatch = PQVec[unknown_indices] - np.concatenate((P_calc, Q_calc))[unknown_indices]

            # convergence check
            if np.all(np.abs(mismatch) < tol):
                # print(f"\t\t\t> converged in {iteration} iterations")
                convergence = True
                break

            # solve linear through LU decomposition
            ''' 
                similar to 
                    new_x = x + np.linalg.inv(J) @ delta_y
                or
                    new_x = np.linalg.solve(J, delta_y) + x
            '''
            # remove slack bus
            J_full = build_jacobian(YBus, ThetaV)
            J_reduced = J_full[np.ix_(unknown_indices, unknown_indices)]

            # convert to sparse matrix for solving
            dThetaV = splu(csc_matrix(J_reduced)).solve(mismatch)
            ThetaV[unknown_indices] += dThetaV

            # reset PV voltages
            if PV_active.any():
                ThetaV[n_bus + PV_idx[PV_active]] = ThetaV0[n_bus + PV_idx[PV_active]]
            else:
                continue

            '''
                make PV-PQ change a little better: to accept an arbitrary number of generators
                with limits. today: all or none must have limits
            '''

            # PV bus Q range control
            if (Qmin is None) and (Qmax is None):
                continue

            # active PV bus indices
            active_idx = PV_idx[PV_active]  # absolute bus indices

            # compute Q at buses (net injection)
            Q_calc = self.PQ(YBus, ThetaV)[1]

            # find Q_g at active PV buses (net injection + load)
            Q_g = Q_calc[active_idx] + Q_load[active_idx] # generator Q at PV buses

            # relative indices inside active PV mask
            rel_idx = np.arange(len(active_idx))

            below_min = Q_g < Qmin[PV_active]  # mask relative to active PV
            above_max = Q_g > Qmax[PV_active]

            exceeded = below_min | above_max
            if exceeded.any():
                print(f'\t\t\t... generators at PV bus index {active_idx[exceeded]} exceeded Q limits, enforcing as PQ buses')
                # buses that exceed limits
                exceed_rel = rel_idx[exceeded]          # relative indices in active PV
                exceed_abs = active_idx[exceeded]       # absolute bus indices in ThetaV

                # enforce reactive power limits
                PQVec[n_bus + exceed_abs[below_min[exceeded]]] = Qmin[PV_active][exceed_rel[below_min[exceeded]]] - Q_load[exceed_abs[below_min[exceeded]]]
                PQVec[n_bus + exceed_abs[above_max[exceeded]]] = Qmax[PV_active][exceed_rel[above_max[exceeded]]] - Q_load[exceed_abs[above_max[exceeded]]]

                # mark these buses as now PQ
                PV_active[PV_active] = ~exceeded

                # update unknown_indices for voltages
                V_unknown = np.delete(np.arange(n_bus), slack_idx)
                V_unknown = np.setdiff1d(V_unknown, PV_idx[PV_active])
                unknown_indices = np.concatenate([theta_unknown, n_bus + V_unknown])

        if not convergence:
            print(f'\t\t\t> power flow did not converged in {max_iter} iterations')
            return None
        
        return ThetaV[:len(ThetaV)//2], ThetaV[len(ThetaV)//2:]
    
    def solve_power_flow(self, CustomNetwork):
        '''
            set up the network and the input data for problem solving
            - create the initial values vector (for theta and V) - power flow objective
            - create the target P and Q values vector (`true values`)
            - solve using newton-raphson
        '''
        # i might not need the folowing line, just use the CustomNetwork passed as argument
        CN = CustomNetwork # an instance of a built network with passive YBus calculated
        CN.compute_bus_net_power()
        PQ_id, PV_id = CN.buses_type()

        '''
            if i remove a generator, i need to get the PV bus right -> PQ
        '''

        bus_idx = CN.idx # bus id <> python index mapping
        num_bus = len(bus_idx)
        
        # map network bus ID to numpy matrix index
        slack_idx = bus_idx[CN.slack_id] # python index, always a single value as per assertion
        PV_idx = np.array([bus_idx[n] for n in PV_id], dtype=int) # python index
        PQ_idx = np.array([bus_idx[n] for n in PQ_id], dtype=int) # python index

        # create Theta, V vector
        slack_angle = np.deg2rad(CN.buses[CN.slack_id].angle)
        Theta0 = np.full(num_bus, slack_angle)
        V0 = np.ones(num_bus)
        V0[slack_idx] = CN.buses[CN.slack_id].V # set slack bus voltage magnitude from network data

        # set fixed bus voltages from generators
        for gen in CN.generators.values():
            V0[bus_idx[gen.bus_1]] = gen.V

        # create P, Q vector
        P = np.zeros(num_bus)
        Q = np.zeros(num_bus)

        # create Q load vector Qg_min/max calculation
        Q_load = np.zeros(num_bus)

        for load in CN.loads.values():
            bus_n = load.bus_1
            Q_load[bus_idx[bus_n]] += load.Q

        # PV buses
        if len(PV_id) != 0:
            
            P[PV_idx] = np.array([CN.buses[n].P for n in PV_id])

            # Q limits
            Qmin = np.array([CN.buses[n].Qg_min for n in PV_id])
            Qmax = np.array([CN.buses[n].Qg_max for n in PV_id])

            # handle Q limits None
            mask_min = Qmin != None
            mask_max = Qmax != None

            Qmin = Qmin[mask_min] if np.any(mask_min) else None
            Qmax = Qmax[mask_max] if np.any(mask_max) else None

        else:
            Qmin = Qmax = None

        # PQ buses
        if len(PQ_id) != 0:
            P[PQ_idx] = np.array([CN.buses[n].P for n in PQ_id])
            Q[PQ_idx] = np.array([CN.buses[n].Q for n in PQ_id])
            
        # unify vectors
        ThetaV0 = np.concatenate((Theta0, V0))
        PQ = np.concatenate((P, Q))

        # call NR solver, return ThetaV if converged or None otherwise
        return self.newton_raphson(ThetaV0, PQ, CN.YBus, slack_idx, PV_idx, Q_load, Qmin, Qmax, tol=1e-9)
    
    def bus_flow(self, CustomNetwork, ThetaV):
        '''
            ThetaV = [Theta, V]
            compute flow from shunt elements, Gen, Loads and Cap/Reactors
            do not compute the flow from line shunt branches, right now...
        '''
        n = len(CustomNetwork.idx)

        # compute net injections
        PQ_net = np.array(self.PQ(CustomNetwork.YBus, ThetaV))

        # load power, loads are lumped at bus
        PQ_load = np.zeros_like(PQ_net)
        for load in CustomNetwork.loads.values():
            if not load.status:
                continue
            bus_n = load.bus_1
            PQ_load[0][CustomNetwork.idx[bus_n]] += load.P
            PQ_load[1][CustomNetwork.idx[bus_n]] += load.Q

        # converter power, converters are lumped at bus
        PQ_converter = np.zeros_like(PQ_net)
        for conv in CustomNetwork.full_converters.values():
            if not conv.status:
                continue
            bus_n = conv.bus_1
            PQ_converter[0][CustomNetwork.idx[bus_n]] += conv.P
            PQ_converter[1][CustomNetwork.idx[bus_n]] += conv.Q

        PQ_gen = PQ_net + PQ_load - PQ_converter

        # for shunt elements (cap/react)
        Ysh_diag = np.zeros(n, dtype=complex)

        for shunt in CustomNetwork.capacitors.values():
            if not shunt.status:
                continue

            rows, cols, data = shunt.get_Y()

            for r, c, y in zip(rows, cols, data):
                Ysh_diag[CustomNetwork.idx[r]] += y

        # Build sparse diagonal YBus for shunts only
        YBus_shunt = diags(Ysh_diag, 0, format='csr')

        # Compute PQ contribution for all shunts at once
        PQ_shunt = np.array(self.PQ(YBus_shunt, ThetaV))

        return (PQ_net, PQ_gen, PQ_converter, PQ_load, PQ_shunt)
    
    def element_flow(self, CustomNetwork, ThetaV):
        '''
            series element power flow
        '''
        n = len(CustomNetwork.idx)
        V_complex = ThetaV[n:] * np.exp(1j*ThetaV[:n])
        element_flow = {}

        element_flow['Lines'] = []
        for name, element in CustomNetwork.lines.items():
            bus_1_idx = CustomNetwork.idx[element.bus_1] # i
            bus_2_idx = CustomNetwork.idx[element.bus_2] # j

            I = np.zeros(n, dtype=complex)

            rows, cols, data = element.get_Y()

            for r, c, y in zip(rows, cols, data):
                I[CustomNetwork.idx[r]] += y * V_complex[CustomNetwork.idx[c]]

            S = V_complex * np.conj(I)

            element_flow['Lines'].append({
                'name': name,
                'i': element.bus_1,
                'j': element.bus_2,
                'Sij': S[bus_1_idx],
                'Sji': S[bus_2_idx]
            })

        element_flow['Capacitors'] = []
        for name, element in CustomNetwork.capacitors.items():
            bus_1_idx = CustomNetwork.idx[element.bus_1] # i

            I = np.zeros(n, dtype=complex)

            rows, cols, data = element.get_Y()

            for r, c, y in zip(rows, cols, data):
                I[CustomNetwork.idx[r]] += y * V_complex[CustomNetwork.idx[c]]

            S = V_complex * np.conj(I)

            element_flow['Capacitors'].append({
                'name': name,
                'i': element.bus_1,
                'j': element.bus_2,
                'Sij': S[bus_1_idx],
                'Sji': -S[bus_1_idx]
            })

        element_flow['Transformers'] = []
        for name, element in CustomNetwork.transformers.items():
            bus_1_idx = CustomNetwork.idx[element.bus_1] # i
            bus_2_idx = CustomNetwork.idx[element.bus_2] # j

            I = np.zeros(n, dtype=complex)

            rows, cols, data = element.get_Y()

            for r, c, y in zip(rows, cols, data):
                I[CustomNetwork.idx[r]] += y * V_complex[CustomNetwork.idx[c]]

            S = V_complex * np.conj(I)

            element_flow['Transformers'].append({
                'name': name,
                'i': element.bus_1,
                'j': element.bus_2,
                'Sij': S[bus_1_idx],
                'Sji': S[bus_2_idx]
            })
        
        return element_flow
    
    def power_flow_summary(self, CustomNetwork, ThetaV):

        net_flow, gen_flow, converter_flow, load_flow, shunt_flow = self.bus_flow(CustomNetwork, ThetaV)

        Theta, V = ThetaV[:len(ThetaV)//2], ThetaV[len(ThetaV)//2:]

        rows = []

        for i, (theta, v, Pnet, Qnet, Pg, Qg, Pconv, Qconv, Pl, Ql, Psh, Qsh) in enumerate(
            zip(Theta, V,
                net_flow[0], net_flow[1],
                gen_flow[0], gen_flow[1],
                converter_flow[0], converter_flow[1],
                load_flow[0], load_flow[1],
                shunt_flow[0], shunt_flow[1])
        ):

            rows.append([
                CustomNetwork.id[i],
                v,
                np.degrees(theta),
                Pnet, Qnet,
                Pg, Qg,
                Pconv, Qconv,
                Pl, Ql,
                Psh, Qsh
            ])

        # --- system totals ---
        total_gen   = sum(r[5]  for r in rows)   # Pg
        total_conv  = sum(r[7]  for r in rows)   # Pconv
        total_load  = sum(r[9] for r in rows)   # Pl
        total_shunt = sum(r[11] for r in rows)   # Psh
        total_losses = total_gen + total_conv - total_load - total_shunt

        totals = {
            "total_gen_MW":    total_gen * 100, # PU > MW
            "total_load_MW":   total_load  * 100, # PU > MW
            "total_shunt_MW":  total_shunt * 100, # PU > MW
            "total_conv_MW":   total_conv * 100, # PU > MW
            "total_losses_MW": total_losses * 100, # PU > MW
            "loss_pct":        abs(total_losses / (total_gen+total_conv)) * 100 if total_gen != 0 else 0,
        }
        
        return rows, totals

    def generator_P_dispatch(self, P_bus, gen_bus):
        '''
            assign bus P flow from power flow to individual generators
            proportional to their rated power. Sync condensers do not get any P
        '''
        # if only one element, no need for dispatch calculation
        if len(gen_bus) == 1:
            gen_bus[0]["P_dispatch"] = P_bus
            return

        total_P = sum(gen["P_weight"] for gen in gen_bus)
        for gen in gen_bus:
            ratio_P = gen["P_weight"] / total_P if total_P > 0 else 0 # if for all synchronous condenser bus
            gen["P_dispatch"] = ratio_P * P_bus

    def generator_Q_dispatch(self, Q_bus, gen_bus, episilon=1e-6):
        '''
            assign bus Q flow from power flow to individual generators
            inversely proportional to their impedance
        '''
        # if only one element, no need for dispatch calculation
        if len(gen_bus) == 1:
            gen_bus[0]["Q_dispatch"] = Q_bus
            return

        total_Y = sum(gen["Q_weight"] for gen in gen_bus)
        for gen in gen_bus:
            # initial alocation
            ratio_Q = gen["Q_weight"] / total_Y
            gen["Q_dispatch"] = ratio_Q * Q_bus

        # if only one generator in bus, no need to make limits check, already
        # enforced in power flow
        if len(gen_bus) == 1:
            return

        # loop through each generator in bus and collect residuals
        for _ in range(len(gen_bus)):
            residual = 0.0
            free = []

            for gen in gen_bus:
                
                # if its already over limit
                if gen["fixed"]:
                    continue

                if gen["Q_dispatch"] < gen["Q_min"]:
                    residual += gen["Q_dispatch"] - gen["Q_min"]
                    gen["Q_dispatch"], gen["fixed"] = gen["Q_min"], True
                elif gen["Q_dispatch"] > gen["Q_max"]:
                    residual += gen["Q_dispatch"] - gen["Q_max"]
                    gen["Q_dispatch"], gen["fixed"] = gen["Q_max"], True
                else:
                    free.append(gen)

            if abs(residual) < episilon or not free:
                break

            free_total = sum(gen["Q_weight"] for gen in free)
            for g in free:
                g["Q_dispatch"] += (g["Q_weight"] / free_total) * residual

    def generators_flow(self, CustomNetwork, gen_bus_flow):
        '''
            compute individual ACTIVE generators power flow
            in case generators are not lumped at bus
            if there is more then one generator at the same bus, make the division between them
            P: proportional to generator P rated
            Q: inversely proportional to |Z|, limited to Q limits
        '''
        state = {
            (full_name, CustomNetwork.idx[gen.bus_1]): {
                "P_weight": gen.P,
                "Q_weight": 1/abs(gen.Z_series),
                "Q_min": gen.Q_min if gen.Q_min is not None else -float("inf"),
                "Q_max": gen.Q_max if gen.Q_max is not None else +float("inf"),
                "P_dispatch": 0,
                "Q_dispatch": 0,
                "fixed": False,
                "breaker_on": gen.status
            }
            for full_name, gen in CustomNetwork.generators.items()
        }

        for bus_idx, (P_bus, Q_bus) in enumerate(zip(gen_bus_flow[0], gen_bus_flow[1])):

            gen_bus = [g for k, g in state.items() if k[1] == bus_idx and g["breaker_on"]]

            # skip dispatch calculation for buses without generators
            if not gen_bus:
                continue

            # make changes directly on state
            self.generator_P_dispatch(P_bus, gen_bus)
            self.generator_Q_dispatch(Q_bus, gen_bus) 

        # gen_flow[(gen.name, bus)] = gen_bus_flow[0][bus] * ratio_P + 1j* gen_Q
        return {k: v["P_dispatch"]+1j*v["Q_dispatch"] for k, v in state.items()}
    
    def loads_flow(self, CustomNetwork, load_bus_flow):
        '''
            compute individual ACTIVE loads power flow
            in case loads are not lumped at bus
            if there is more then one load at the same bus, make the division between them
            P/Q: proportional to load P/Q
        '''
        state = {
            (full_name, CustomNetwork.idx[element.bus_1]): {
                "P_weight": element.P,
                "Q_weight": element.Q,
                "P_dispatch": 0,
                "Q_dispatch": 0,
                "breaker_on": element.status
            }
            for full_name, element in CustomNetwork.loads.items()
        }

        sum_P = np.zeros(load_bus_flow.shape[1])
        sum_Q = np.zeros(load_bus_flow.shape[1])
        for (_, bus_idx), element in state.items():
            if not element["breaker_on"]:
                continue # dont share unconnected elements
            sum_P[bus_idx] += element["P_weight"]
            sum_Q[bus_idx] += element["Q_weight"]

        scale_P =  np.divide(load_bus_flow[0], sum_P, out=np.ones_like(load_bus_flow[0], dtype=float), where=sum_P != 0)
        scale_Q =  np.divide(load_bus_flow[1], sum_Q, out=np.ones_like(load_bus_flow[1], dtype=float), where=sum_Q != 0)

        for (_, bus_idx), element in state.items():
            if not element["breaker_on"]:
                continue # unconnected elements stays zeroed

            element["P_dispatch"] = element["P_weight"] * scale_P[bus_idx]
            element["Q_dispatch"] = element["Q_weight"] * scale_Q[bus_idx]

        return {k: v["P_dispatch"]+1j*v["Q_dispatch"] for k, v in state.items()}
    
    def converters_flow(self, CustomNetwork, converter_bus_flow):
        '''
            compute individual ACTIVE converters power flow
            in case converters are not lumped at bus
            if there is more then one converter at the same bus, make the division between them
            P/Q: proportional to converter P/Q
        '''
        state = {
            (full_name, CustomNetwork.idx[element.bus_1]): {
                "P_weight": element.P,
                "Q_weight": element.Q,
                "P_dispatch": 0,
                "Q_dispatch": 0,
                "breaker_on": element.status
            }
            for full_name, element in CustomNetwork.full_converters.items()
        }

        sum_P = np.zeros(converter_bus_flow.shape[1])
        sum_Q = np.zeros(converter_bus_flow.shape[1])
        for (_, bus_idx), element in state.items():
            if not element["breaker_on"]:
                continue # dont share unconnected elements
            sum_P[bus_idx] += element["P_weight"]
            sum_Q[bus_idx] += element["Q_weight"]

        scale_P =  np.divide(converter_bus_flow[0], sum_P, out=np.ones_like(converter_bus_flow[0], dtype=float), where=sum_P != 0)
        scale_Q =  np.divide(converter_bus_flow[1], sum_Q, out=np.ones_like(converter_bus_flow[1], dtype=float), where=sum_Q != 0)

        for (_, bus_idx), element in state.items():
            if not element["breaker_on"]:
                continue # unconnected elements stays zeroed

            element["P_dispatch"] = element["P_weight"] * scale_P[bus_idx]
            element["Q_dispatch"] = element["Q_weight"] * scale_Q[bus_idx]

        return {k: v["P_dispatch"]+1j*v["Q_dispatch"] for k, v in state.items()}

    def external_grids_flow(self, CustomNetwork, external_grid_bus_flow):
        '''
            only one external grid in a bus
            this function just format the bus flow into the right format
            for feeding into SCC NR and to serve as the SCADA interface
            right now the external grids have no flow exchange to the internal network
        '''

        state = {
            (full_name, CustomNetwork.idx[element.bus_1]): {
                "P_dispatch": 0,
                "Q_dispatch": 0,
                "breaker_on": element.status
            }
            for full_name, element in CustomNetwork.external_grids.items()
        }

        # current implementation: external grids does not exchange power
        if external_grid_bus_flow is not None:
            for (_, bus_idx), element in state.items():
                if not element["breaker_on"]:
                    continue # unconnected elements stays zeroed
                element["P_dispatch"] = external_grid_bus_flow[0][bus_idx]
                element["Q_dispatch"] = external_grid_bus_flow[1][bus_idx]

        return {k: v["P_dispatch"]+1j*v["Q_dispatch"] for k, v in state.items()}

    def get_active_elements_flow(self, CustomNetwork, ThetaV):
        '''
            ThetaV = [Theta, V]
            get the complex power for each active element for instantiation
            it returns even the disconnected elements flow to have compatibility with 
                SCADA endpoints
        '''
        _, gen_flow, converter_flow, load_flow, _ = self.bus_flow(CustomNetwork, ThetaV)

        # get the power for each element as power flow solution output bus values
        load_S = self.loads_flow(CustomNetwork, load_flow)
        gen_S = self.generators_flow(CustomNetwork, gen_flow)
        converter_S = self.converters_flow(CustomNetwork, converter_flow)
        external_grid_S = self.external_grids_flow(CustomNetwork, None) # no power from power flow

        return load_S | gen_S | converter_S | external_grid_S

    def get_active_element_status(self, CustomNetwork):
        ''''
            acquire the status of production facilities
            emulate SCADA data aquisition of these states
        '''
        active_element_status = {}

        # generators
        for full_name, element in CustomNetwork.generators.items():
            active_element_status[full_name] = element.status

        # converters
        for full_name, element in CustomNetwork.full_converters.items():
            active_element_status[full_name] = element.status

        # loads
        for full_name, element in CustomNetwork.loads.items():
            active_element_status[full_name] = element.status

        # external grids
        for full_name, element in CustomNetwork.external_grids.items():
            active_element_status[full_name] = element.status

        return active_element_status


def build_jacobian(YBus, ThetaV):
    '''
        auxiliary function to compute the jacobian to solve the power flow problem
        with newton-raphson

        use explicit formulas for the partial derivatives of P and Q w.r.t. Theta and V
        J1: dP/dTheta
        J2: dP/dV
        J3: dQ/dTheta
        J4: dQ/dV

        the class does not hold any attribute, instead is the skeleton that will update
        the value of the jacobian given a change in the VTheta vector since YBus is fixed.
        
        - VTheta in the newton-raphson is updates at each iteration until convergence.
    '''
    num_bus = YBus.shape[0]
    theta = ThetaV[:num_bus]
    V = ThetaV[num_bus:]

    rows = []
    cols = []
    data = []

    YBus = YBus.tocsr()

    for k in range(num_bus):
        V_k = V[k]
        theta_k = theta[k]

        # for the k-th row of the YBus matrix, get non-zero entries
        row_start = YBus.indptr[k]
        row_end = YBus.indptr[k + 1]

        indices = YBus.indices[row_start:row_end]
        Y_vals = YBus.data[row_start:row_end]

        G = Y_vals.real
        B = Y_vals.imag

        # Diagonal placeholders
        G_kk = 0.0
        B_kk = 0.0

        sum_J1 = 0.0
        sum_J2 = 0.0
        sum_J3 = 0.0
        sum_J4 = 0.0

        for idx, n in enumerate(indices):
            G_kn = G[idx]
            B_kn = B[idx]
            d_theta = theta_k - theta[n]

            if n == k:
                G_kk = G_kn
                B_kk = B_kn
                continue

            # off-diagonal entries

            # J1 (dP/dTheta)
            rows.append(k)
            cols.append(n)
            data.append(V_k * V[n] * (G_kn * np.sin(d_theta) - B_kn * np.cos(d_theta)))

            # J2 (dP/dV)
            rows.append(k)
            cols.append(n + num_bus)
            data.append(V_k * (G_kn * np.cos(d_theta) + B_kn * np.sin(d_theta)))

            # J3 (dQ/dTheta)
            rows.append(k + num_bus)
            cols.append(n)
            data.append(-V_k * V[n] * (G_kn * np.cos(d_theta) + B_kn * np.sin(d_theta)))

            # J4 (dQ/dV)
            rows.append(k + num_bus)
            cols.append(n + num_bus)
            data.append(V_k * (G_kn * np.sin(d_theta) - B_kn * np.cos(d_theta)))

            # accumulate diagonal sums
            sum_J1 += V[n] * (-G_kn * np.sin(d_theta) + B_kn * np.cos(d_theta))
            sum_J2 += V[n] * (G_kn * np.cos(d_theta) + B_kn * np.sin(d_theta))
            sum_J3 += V[n] * (G_kn * np.cos(d_theta) + B_kn * np.sin(d_theta))
            sum_J4 += V[n] * (G_kn * np.sin(d_theta) - B_kn * np.cos(d_theta))

        # --- Diagonal entries ---
        # J1_kk
        rows.append(k)
        cols.append(k)
        data.append(V_k * sum_J1)

        # J2_kk
        rows.append(k)
        cols.append(k + num_bus)
        data.append(2 * V_k * G_kk + sum_J2)

        # J3_kk
        rows.append(k + num_bus)
        cols.append(k)
        data.append(V_k * sum_J3)

        # J4_kk
        rows.append(k + num_bus)
        cols.append(k + num_bus)
        data.append(-2 * V_k * B_kk + sum_J4)

    # Build sparse COO matrix and convert to CSR
    J = coo_matrix((data, (rows, cols)), shape=(2*num_bus, 2*num_bus))
    return J.tocsr()