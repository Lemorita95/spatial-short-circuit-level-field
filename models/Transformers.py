import numpy as np

class Transformer:
    def __init__(self, name, bus_1, bus_2, Z_series, Y_shunt, m0, m_min, m_max, m_step, n0, status):
        ''' n0: initial tap position, integer, limited to [n_min, n_max] '''
        self.name = name
        self.bus_1 = bus_1
        self.bus_2 = bus_2
        self.Z_series = Z_series
        self.Y_shunt = Y_shunt
        self.status = True if status is None else status

        # Tap grid: m = m0 + n * m_step, limited to [m_min, m_max]
        self.m0 = m0                    # reference (neutral) ratio, system pu
        self.m_min = m_min
        self.m_max = m_max
        self.m_step = m_step
        self.n_min = int(np.ceil((m_min - m0) / m_step - 1e-9))
        self.n_max = int(np.floor((m_max - m0) / m_step + 1e-9))
        self.n = int(np.clip(n0, self.n_min, self.n_max))   # current tap position

    # ---- tap state -------------------------------------------------
    @property
    def m(self):
        """Current tap ratio (read-only, derived from n)."""
        return self.m0 + self.n * self.m_step

    @property
    def tap_ratio(self):
        m = self.m
        return [1 / m**2, 1 / m, 1 / m, 1]

    def can_step(self, direction):
        """direction = +1 / -1 in m."""
        return self.status and self.n_min <= self.n + direction <= self.n_max

    def step(self, direction):
        if not self.can_step(direction):
            raise ValueError(f"{self.name}: tap limit reached or out of service")
        self.n += direction

    def reset_tap(self, n0=0):
        self.n = int(np.clip(n0, self.n_min, self.n_max))

    # ---- admittance ------------------------------------------------
    def get_Y(self):
        '''
            Y11 = (y + ysh) / m²
            Y12 = -y / m
            Y21 = -y / m
            Y22 =  y + ysh
        '''
        rows = [self.bus_1, self.bus_1, self.bus_2, self.bus_2]
        cols = [self.bus_1, self.bus_2, self.bus_1, self.bus_2]
        if not self.status:
            return rows, cols, [0, 0, 0, 0]

        y = 1 / self.Z_series
        ysh = self.Y_shunt / 2
        base = [y + ysh, -y, -y, y + ysh]
        Ydata = [b * r for b, r in zip(base, self.tap_ratio)]
        return rows, cols, Ydata

    def get_dY_dm(self):
        """d[Y11, Y12, Y21, Y22]/dm, for the voltage sensitivities."""
        y = 1 / self.Z_series
        ysh = self.Y_shunt / 2
        m = self.m
        return [-2 * (y + ysh) / m**3, y / m**2, y / m**2, 0]