import numpy as np
import random

random.seed(42)
np.random.seed(42)
num_jobs = 10
num_machines = 3
job_times = [random.randint(2, 15) for _ in range(num_jobs)]

print("Job times:", job_times)


num_particles = 40
max_iter = 200
w = 0.7
c1 = 1.4
c2 = 1.4
def schedule_from_position(pos):
    sched = np.rint(pos).astype(int)
    sched = np.clip(sched, 0, num_machines-1)
    return sched

def fitness(schedule):
    loads = [0] * num_machines
    for job, machine in enumerate(schedule):
        loads[machine] += job_times[job]
    return max(loads)

positions = np.random.uniform(0, num_machines-1, size=(num_particles, num_jobs))
velocities = np.random.uniform(-1, 1, size=(num_particles, num_jobs)) * 0.1

pbest_pos = positions.copy()
pbest_schedule = np.array([schedule_from_position(p) for p in pbest_pos])
pbest_fitness = np.array([fitness(s) for s in pbest_schedule])

gbest_idx = int(np.argmin(pbest_fitness))
gbest_pos = pbest_pos[gbest_idx].copy()
gbest_schedule = pbest_schedule[gbest_idx].copy()
gbest_value = float(pbest_fitness[gbest_idx])

for it in range(max_iter):
    for i in range(num_particles):
        r1 = np.random.rand(num_jobs)
        r2 = np.random.rand(num_jobs)

        velocities[i] = (w * velocities[i]
                         + c1 * r1 * (pbest_pos[i] - positions[i])
                         + c2 * r2 * (gbest_pos - positions[i]))
        positions[i] = positions[i] + velocities[i]

        positions[i] = np.clip(positions[i], 0.0, float(num_machines - 1))

        sched = schedule_from_position(positions[i])
        fit = fitness(sched)

        if fit < pbest_fitness[i]:
            pbest_fitness[i] = fit
            pbest_pos[i] = positions[i].copy()
            pbest_schedule[i] = sched.copy()

            if fit < gbest_value:
                gbest_value = fit
                gbest_pos = positions[i].copy()
                gbest_schedule = sched.copy()

print("\nBest schedule (machine assignment per job):", gbest_schedule.tolist())
for m in range(num_machines):
    jobs_on_m = [j for j in range(num_jobs) if gbest_schedule[j] == m]
    load = sum(job_times[j] for j in jobs_on_m)
    print(f"Machine {m}: jobs {jobs_on_m}, load = {load}")

print("Final makespan:", gbest_value)
