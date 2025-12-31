import matplotlib
matplotlib.use("MacOSX")  # macOS backend (sem tkinter)
import matplotlib.pyplot as plt

class ACSLiveViewer:
    def __init__(self, problem, pause=0.05):
        self.problem = problem
        self.best_costs = []
        self.pause = pause

        plt.ion()
        self.fig = plt.figure(figsize=(10, 4))
        self.ax_cost = self.fig.add_subplot(1, 2, 1)
        self.ax_map  = self.fig.add_subplot(1, 2, 2)

        coords = problem.coords
        self.x = coords[:, 0]
        self.y = coords[:, 1]

        self.fig.show()

    def __call__(self, event):
        if event.get("type") != "iteration_end":
            return

        it = event["iteration"]
        best_cost = event["global_best_cost"]
        sol = event["global_best_solution"]
        self.best_costs.append(best_cost)

        self.ax_cost.clear()
        self.ax_cost.plot(self.best_costs)
        self.ax_cost.set_title("Global Best Cost")
        self.ax_cost.set_xlabel("Iteration")
        self.ax_cost.set_ylabel("Cost")
        self.ax_cost.grid(True)

        self.ax_map.clear()
        self.ax_map.scatter(self.x, self.y)
        self.ax_map.scatter([self.x[0]], [self.y[0]], marker="s")

        for route in sol.routes:
            nodes = route.nodes
            self.ax_map.plot(self.x[nodes], self.y[nodes], linewidth=1)

        self.ax_map.set_title(f"Routes (it={it+1}, best={best_cost:.2f})")
        self.ax_map.axis("equal")
        self.ax_map.grid(True)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(self.pause)
