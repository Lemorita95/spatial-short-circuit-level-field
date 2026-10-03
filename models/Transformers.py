class Transformer:
    def __init__(self, name, bus_1, bus_2, Z_series, Y_shunt, m, status):
        self.name = name
        self.bus_1 = bus_1
        self.bus_2 = bus_2
        self.Z_series = Z_series
        self.Y_shunt = Y_shunt
        self.m = m # fixed tap ratio
        self.status = True if status is None else status
        self.tap_ratio = [1/(m**2), 1/m, 1/m, 1]


    def get_Y(self):
        rows = [self.bus_1, self.bus_1, self.bus_2, self.bus_2]
        cols = [self.bus_1, self.bus_2, self.bus_1, self.bus_2]
        
        data = [1/self.Z_series + self.Y_shunt/2 if i == j else -1/self.Z_series for i, j in zip(rows,cols)]

        Ydata = [y*ratio for y, ratio in zip(data, self.tap_ratio)]

        return (rows, cols, Ydata)