import numpy as np
from vedo import *

class PedestrianSimulation:
    def __init__(self):
        # Simulation parameters
        self.num_people = 6
        self.dt = 0.25
        self.max_steps = 1000
        self.person_radius = 0.5
        self.obstacle_radius = 3.0
        self.social_radius = 2.0
        self.goal_attraction = 2.0
        self.obstacle_repulsion = 4.0
        self.person_repulsion = 2.0
        
        # Scene
        self.room_width = 30
        self.room_height = 20
        self.door_width = 6
        self.obstacle_pos = np.array([20, 10, 0])
        
        # Plotting pedestrians
        self.positions = np.zeros((self.num_people, 3))
        self.positions[:, 0] = np.random.uniform(2, 8, self.num_people)
        self.positions[:, 1] = np.linspace(2, self.room_height-2, self.num_people)
        np.random.shuffle(self.positions[:, 1])
        
        # Setting goal
        self.goal = np.array([self.room_width + 5, self.room_height/2, 0])
        
        # Setting walls
        self.walls = [
            np.array([[self.room_width, 0, 0], [self.room_width, (self.room_height-self.door_width)/2, 0]]),
            np.array([[self.room_width, self.room_height, 0], [self.room_width, (self.room_height+self.door_width)/2, 0]]),
        ]
        
        # Initialize plotter
        self.plt = Plotter(interactive=False, axes=1)
        self.people_viz = None
        self.obstacle_viz = None
        self.walls_viz = None
        self.goal_viz = None
        
    def penalty_field(self, d, R):
        # Obstacle avoidance penalty function
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
        
        obstacle_distances = np.linalg.norm(positions - self.obstacle_pos, axis=1)
        obstacle_cost = self.obstacle_repulsion * self.penalty_field(obstacle_distances, self.obstacle_radius)
        
        wall_distances = np.array([self.distance_to_walls(p) for p in positions])
        wall_cost = self.obstacle_repulsion * self.penalty_field(wall_distances, self.obstacle_radius)
        
        person_cost = np.zeros(self.num_people)
        for i in range(self.num_people):
            others = np.delete(positions, i, axis=0)
            distances = np.linalg.norm(others - positions[i], axis=1)
            cost = self.person_repulsion * np.sum(self.penalty_field(distances, self.social_radius))
            person_cost[i] = cost
        
        total_cost = goal_cost + obstacle_cost + wall_cost + person_cost
        return np.nan_to_num(total_cost, nan=1e10, posinf=1e10, neginf=1e10)
    
    def calculate_gradient(self, positions, epsilon=1e-4):
        # Numerical gradient method
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
            self.people_viz = [Sphere(pos, r=self.person_radius).c("blue") for pos in self.positions]
            self.obstacle_viz = Cylinder(pos=self.obstacle_pos, r=1, height=2, axis=(0,0,1)).c("red")
            
            wall_objects = []
            for wall in self.walls:
                barrier = Box(pos=wall[0],  length=1, width=15, height=2).lw(5).c("black")
                wall_objects.append(barrier)
            self.walls_viz = wall_objects
            
            self.goal_viz = Box(pos=self.goal, length=2, width=2, height=2).c("green").alpha(0.5)
            
            self.plt.camera.SetPosition(-50, -50, 50)
            self.plt.camera.SetViewUp(0, 0, 1)

            self.plt += self.people_viz + [self.obstacle_viz] + self.walls_viz + [self.goal_viz]
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
            
            if np.all(self.positions[:, 0] > self.room_width):
                break
            
        
        self.plt.interactive().close()

# Main
sim = PedestrianSimulation()
sim.run_simulation()