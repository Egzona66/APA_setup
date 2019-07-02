import matplotlib.pyplot as plt


class Plotter:
    def __init__(self):
        pass

    def initialise_sensors_plot(self):
        plt.ion()

        self.sensors_live_figure, self.sensors_live_ax = plt.subplots(figsize=(12, 8))
        self.sensors_live_ax.set(title="Sensors data", ylabel="Volts (0-1)", xlabel="time - frames", facecolor=[.2, .2, .2])

        self.sensors_live_data = {
            sensor:[] for sensor in self.arduino_config["sensors"]
        }


        plt.pause(0.000001)

    def append_sensors_data(self, states):
        for ch, data in self.sensors_live_data.items():
            data.append(states[ch])

    def update_sensors_plot(self):
        for ch, color in self.analysis_config["plot_colors"].items():
            self.sensors_live_ax.plot(self.sensors_live_data[ch], color=color)
        plt.pause(0.00001)


if __name__ == "__main__":
    p = Plotter()