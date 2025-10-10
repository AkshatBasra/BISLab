import numpy as np

np.random.seed(42)
true_center = (5, 5)
true_radius = 3
num_points = 50
noise = 0.5

angles = np.linspace(0, 2*np.pi, num_points)
points = np.array([
    true_center[0] + true_radius * np.cos(angles) + np.random.randn(num_points)*noise,
    true_center[1] + true_radius * np.sin(angles) + np.random.randn(num_points)*noise
]).T


def fitness(circle_params, points):
    a, b, R = circle_params
    distances = np.sqrt((points[:,0]-a)**2 + (points[:,1]-b)**2)
    return np.sum((distances - R)**2)

num_particles = 30
num_iterations = 100
w = 0.5
c1 = 1
c2 = 1

x_min, x_max = points[:,0].min()-5, points[:,0].max()+5
y_min, y_max = points[:,1].min()-5, points[:,1].max()+5
r_min, r_max = 1, 20

particles = np.random.rand(num_particles, 3)
particles[:,0] = x_min + particles[:,0]*(x_max - x_min)
particles[:,1] = y_min + particles[:,1]*(y_max - y_min)
particles[:,2] = r_min + particles[:,2]*(r_max - r_min)

velocities = np.random.rand(num_particles, 3)*0.5 - 0.25

pbest = particles.copy()
pbest_fitness = np.array([fitness(p, points) for p in particles])

gbest_idx = np.argmin(pbest_fitness)
gbest = pbest[gbest_idx].copy()
gbest_fit = pbest_fitness[gbest_idx]

for t in range(num_iterations):
    for i in range(num_particles):
        r1, r2 = np.random.rand(2)
        velocities[i] = (w*velocities[i] 
                         + c1*r1*(pbest[i]-particles[i]) 
                         + c2*r2*(gbest - particles[i]))
        particles[i] += velocities[i]
        if particles[i,2] < 0.1:
            particles[i,2] = 0.1
        f = fitness(particles[i], points)
        if f < pbest_fitness[i]:
            pbest[i] = particles[i].copy()
            pbest_fitness[i] = f
            if f < gbest_fit:
                gbest = particles[i].copy()
                gbest_fit = f


print("Estimated center: ({:.3f}, {:.3f})".format(gbest[0], gbest[1]))
print("Estimated radius: {:.3f}".format(gbest[2]))
print("Error in center: ({:.3f}, {:.3f})".format(gbest[0]-true_center[0], gbest[1]-true_center[1]))
