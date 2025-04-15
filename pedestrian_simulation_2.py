import numpy as np
from vedo import *

class PedestrianSimulation:
    def __init__(self):
        # Simulation parameters
        self.num_group1 = 4
        self.num_group2 = 4
        self.num_people = self.num_group1 + self.num_group2
        self.dt = 0.25
        self.max_steps = 1000
        self.person_radius = 0.5
        self.obstacle_radius = 2.0
        self.social_radius = 2.0
        self.goal_attraction = 2.0
        self.obstacle_repulsion = 3.0
        self.person_repulsion = 2.0
        
        # Scene
        self.corridor_width = 8
        self.corridor_length = 30
        self.junction_pos = 15

        # Range of positions
        self.positions = np.zeros((self.num_people, 3))
        
        # Plotting first group of pedestrians
        self.positions[:self.num_group1, 0] = np.linspace(2, 8, self.num_group1)
        self.positions[:self.num_group1, 1] = np.random.uniform(
            self.junction_pos + 2, self.junction_pos + 4, self.num_group1)
        
        # Plotting second group of pedestrians
        self.positions[self.num_group1:, 1] = np.linspace(2, 8, self.num_group2)
        self.positions[self.num_group1:, 0] = np.random.uniform(
            self.junction_pos - 2, self.junction_pos + 2, self.num_group2)
        
        # Plot end goal
        self.goal = np.array([self.corridor_length + 10, self.junction_pos + 3, 0])
        
        # Barriers
        self.walls = []
        
        self.walls.append(np.array([[self.junction_pos - self.corridor_width/2, 0, 0],
                             [self.junction_pos - self.corridor_width/2, self.junction_pos, 0]]))
        
        self.walls.append(np.array([[self.junction_pos + self.corridor_width/2, 0, 0],
                             [self.junction_pos + self.corridor_width/2, self.junction_pos, 0]]))
        
        self.walls.append(np.array([[0, self.junction_pos, 0], [self.junction_pos - self.corridor_width/2, self.junction_pos, 0]]))

        self.walls.append(np.array([[self.junction_pos + self.corridor_width/2, self.junction_pos, 0],
                             [self.corridor_length, self.junction_pos, 0]]))
        
        # Initialize plotter
        self.plt = Plotter(interactive=False, axes=1, size=(1200, 800))
        self.people_viz = None
        self.group1_viz = None
        self.group2_viz = None
        self.walls_viz = None
        self.goal_viz = None
    
    def penalty_field(self, d, R):
        # Same obstacle avoidance penalty function
        safe_d = np.maximum(d, 1e-10)
        f = np.where(d <= R, np.log(R / safe_d), 0)
        return np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0)
    
    def distance_to_walls(self, point):
        # Calculating minimum distance from pedestrian to walls
        min_dist = np.inf
        for wall in self.walls:
            v = wall[1] - wall[0]
            w = point - wall[0]
            
            c1 = np.dot(w, v)
            if c1 <= 0:
                dist = np.linalg.norm(point - wall[0])
            else:
                c2 = np.dot(v, v)
                if c2 <= c1:
                    dist = np.linalg.norm(point - wall[1])
                else:
                    b = c1 / (c2 + 1e-10)
                    pb = wall[0] + b * v
                    dist = np.linalg.norm(point - pb)
            
            if dist < min_dist:
                min_dist = dist
        return min_dist
    
    def calculate_cost(self, positions):
        # Total cost for all pedestrians
        goal_distances = np.linalg.norm(positions - self.goal, axis=1)
        goal_cost = self.goal_attraction * goal_distances
        
        wall_distances = np.array([self.distance_to_walls(p) for p in positions])
        wall_cost = self.obstacle_repulsion * self.penalty_field(wall_distances, self.obstacle_radius)
        
        person_cost = np.zeros(self.num_people)
        for i in range(self.num_people):
            others = np.delete(positions, i, axis=0)
            distances = np.linalg.norm(others - positions[i], axis=1)
            cost = self.person_repulsion * np.sum(self.penalty_field(distances, self.social_radius))
            person_cost[i] = cost
        
        total_cost = goal_cost + wall_cost + person_cost
        return np.nan_to_num(total_cost, nan=1e10, posinf=1e10, neginf=1e10)
    
    def calculate_gradient(self, positions, epsilon=1e-4):
        # Gradient method
        grad = np.zeros_like(positions)
        for i in range(self.num_people):
            for j in range(3):
                pos_plus = positions.copy()
                pos_minus = positions.copy()
                pos_plus[i, j] += epsilon
                pos_minus[i, j] -= epsilon
                
                cost_plus = self.calculate_cost(pos_plus)[i]
                cost_minus = self.calculate_cost(pos_minus)[i]
                
                if not np.isfinite(cost_plus - cost_minus):
                    grad[i, j] = 0
                else:
                    grad[i, j] = (cost_plus - cost_minus) / (2 * epsilon)
        
        return -grad
    
    def update_positions(self):
        # Updating pedestrian positions
        grad = self.calculate_gradient(self.positions)
        self.positions += self.dt * grad
    
    def visualize(self):
        # Method for changing scene
        if not self.people_viz:
            self.group1_viz = [Sphere(pos, r=self.person_radius).c("blue") 
                             for pos in self.positions[:self.num_group1]]
            self.group2_viz = [Sphere(pos, r=self.person_radius).c("green") 
                             for pos in self.positions[self.num_group1:]]
            self.people_viz = self.group1_viz + self.group2_viz
            
            wall_objects = []
            for wall in self.walls[:2]:
                barrier = Box(pos=wall[0],  length=1, width=31, height=2).lw(5).c("black")
                wall_objects.append(barrier)
            for wall in self.walls[2:3]:
                barrier = Box(pos=wall[0],  length=21, width=1, height=2).lw(5).c("black")
                wall_objects.append(barrier)
            for wall in self.walls[3:]:
                barrier = Box(pos=wall[1],  length=21, width=1, height=2).lw(5).c("black")
                wall_objects.append(barrier)
            self.walls_viz = wall_objects
            
            self.goal_viz = Box(pos=self.goal, length=2, width=2, height=2).c("red").alpha(0.5)
            
            self.plt.camera.SetPosition(-50, -50, 50)
            self.plt.camera.SetViewUp(0, 0, 1)

            self.plt += self.people_viz + self.walls_viz + [self.goal_viz]
            self.plt.show(interactive=False)
        else:
            for i, viz in enumerate(self.people_viz):
                viz.pos(self.positions[i])
            self.plt.render()
    
    def run_simulation(self):
        # Simulate pedestrian simulation
        for step in range(self.max_steps):
            self.update_positions()
            self.visualize()
            
            if np.all(self.positions[:, 0] > self.corridor_length + 8):
                break
        
        self.plt.interactive().close()

# Main
sim = PedestrianSimulation()
sim.run_simulation()