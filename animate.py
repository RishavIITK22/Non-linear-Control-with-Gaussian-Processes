import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def animate(q_log, l1=2.0, l2=1.0, interval=10):
    fig, ax = plt.subplots()
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(- (l1+l2+0.5), l1+l2+0.5)
    ax.set_ylim(- (l1+l2+0.5), l1+l2+0.5)
    line, = ax.plot([], [], 'o-', lw=2)

    def update(i):
        q1, q2 = q_log[i]
        x1 = l1*np.cos(q1); y1 = l1*np.sin(q1)
        x2 = x1 + l2*np.cos(q1+q2); y2 = y1 + l2*np.sin(q1+q2)
        line.set_data([0,x1,x2],[0,y1,y2])
        return line,
    ani = FuncAnimation(fig, update, frames=len(q_log), interval=interval, blit=True)
    plt.show()

if __name__ == '__main__':
    from sim import run_sim
    log = run_sim(T=6.0, dt=0.005, uncert=0.2, controller='robust_learning')
    animate(log['q'], l1=2.0, l2=1.0)
