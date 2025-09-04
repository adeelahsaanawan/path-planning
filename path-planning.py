# Author: Adeel Ahsan
# Website: https://www.aeronautyy.com
# License: MIT
# Copyright (c) 2025 Adeel Ahsan

import time
import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.widgets import Button, RadioButtons, Slider, CheckButtons
from scipy.spatial import Delaunay
from scipy.ndimage import maximum_filter


# -----------------------------
# Obstacle generation
# -----------------------------

def triangle_circumradius(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Circumradius of triangle ABC."""
    # side lengths
    A = np.linalg.norm(b - c)
    B = np.linalg.norm(a - c)
    C = np.linalg.norm(a - b)
    s = 0.5 * (A + B + C)
    area_sq = max(s * (s - A) * (s - B) * (s - C), 1e-16)
    area = math.sqrt(area_sq)
    R = (A * B * C) / (4.0 * area)
    return R


def gen_obstacles(seed: Optional[int] = None,
                  n_points: int = 200,
                  alpha: float = 25.0,
                  limits: Tuple[Tuple[float, float], Tuple[float, float]] = ((0, 1), (0, 1))) -> List[np.ndarray]:
    """
    Generate triangular obstacles similarly to the IRIS 2D script:
    - Sample random points in [0,1]^2
    - Delaunay triangulation
    - Keep triangles with circumradius < 1/alpha (alpha-shape like)
    - Add bounding triangles around the square workspace
    Returns list of triangles as (3,2) numpy arrays.
    """
    if seed is not None:
        np.random.seed(seed)

    pts = np.random.random(size=(n_points, 2))
    tri = Delaunay(pts)
    keep = []
    thresh = 1.0 / alpha
    for simplex in tri.simplices:
        a, b, c = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
        R = triangle_circumradius(a, b, c)
        if R < thresh:
            keep.append(np.vstack([a, b, c]))

    # Add 4 outer triangles surrounding the square like the original script
    keep.append(np.array([[0, 0], [1, 0], [0.5, -0.5]]))
    keep.append(np.array([[1, 0], [1, 1], [1.5, 0.5]]))
    keep.append(np.array([[1, 1], [0, 1], [0.5, 1.5]]))
    keep.append(np.array([[0, 1], [0, 0], [-0.5, 0.5]]))

    return keep


# -----------------------------
# Geometry utilities
# -----------------------------

def point_in_triangle(p: np.ndarray, tri: np.ndarray) -> bool:
    """Barycentric coordinate test for point in triangle."""
    a, b, c = tri
    v0 = c - a
    v1 = b - a
    v2 = p - a
    dot00 = np.dot(v0, v0)
    dot01 = np.dot(v0, v1)
    dot02 = np.dot(v0, v2)
    dot11 = np.dot(v1, v1)
    dot12 = np.dot(v1, v2)
    denom = (dot00 * dot11 - dot01 * dot01)
    if abs(denom) < 1e-12:
        return False
    inv = 1.0 / denom
    u = (dot11 * dot02 - dot01 * dot12) * inv
    v = (dot00 * dot12 - dot01 * dot02) * inv
    return (u >= 0) and (v >= 0) and (u + v <= 1)


def segment_intersects_segment(p: np.ndarray, p2: np.ndarray, q: np.ndarray, q2: np.ndarray) -> bool:
    """Check if segments p-p2 and q-q2 intersect (including colinear overlap)."""
    def orient(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    def on_seg(a, b, c):  # c on segment ab
        return min(a[0], b[0]) - 1e-12 <= c[0] <= max(a[0], b[0]) + 1e-12 and \
               min(a[1], b[1]) - 1e-12 <= c[1] <= max(a[1], b[1]) + 1e-12

    o1 = orient(p, p2, q)
    o2 = orient(p, p2, q2)
    o3 = orient(q, q2, p)
    o4 = orient(q, q2, p2)

    if (o1 * o2 < 0) and (o3 * o4 < 0):
        return True
    # handle colinear cases
    if abs(o1) < 1e-12 and on_seg(p, p2, q):
        return True
    if abs(o2) < 1e-12 and on_seg(p, p2, q2):
        return True
    if abs(o3) < 1e-12 and on_seg(q, q2, p):
        return True
    if abs(o4) < 1e-12 and on_seg(q, q2, p2):
        return True
    return False


def segment_intersects_triangles(p: np.ndarray, q: np.ndarray, tris: List[np.ndarray]) -> bool:
    for tri in tris:
        # If both endpoints inside any triangle, it's in collision
        if point_in_triangle(p, tri) and point_in_triangle(q, tri):
            return True
        # Check against triangle edges
        for i in range(3):
            a = tri[i]
            b = tri[(i + 1) % 3]
            if segment_intersects_segment(p, q, a, b):
                return True
    return False


def point_in_collision(p: np.ndarray, tris: List[np.ndarray]) -> bool:
    for tri in tris:
        if point_in_triangle(p, tri):
            return True
    return False


# -----------------------------
# RRT family
# -----------------------------

@dataclass
class Node:
    pos: np.ndarray
    parent: Optional[int] = None
    cost: float = 0.0  # for RRT*


def steer(from_pt: np.ndarray, to_pt: np.ndarray, step: float) -> np.ndarray:
    vec = to_pt - from_pt
    dist = np.linalg.norm(vec)
    if dist <= step:
        return to_pt.copy()
    if dist < 1e-12:
        return from_pt.copy()
    return from_pt + (vec / dist) * step


def nearest(nodes: List[Node], pt: np.ndarray) -> int:
    dists = [np.linalg.norm(n.pos - pt) for n in nodes]
    return int(np.argmin(dists))


def rrt(start: np.ndarray, goal: np.ndarray, tris: List[np.ndarray],
        step_size: float, goal_sample_rate: float, max_iters: int,
        ax, animate: bool = True) -> Tuple[List[np.ndarray], Dict[str, float]]:
    nodes = [Node(start)]
    t0 = time.time()
    for it in range(max_iters):
        if random.random() < goal_sample_rate:
            rnd = goal
        else:
            rnd = np.array([random.random(), random.random()])
        idx = nearest(nodes, rnd)
        new_pos = steer(nodes[idx].pos, rnd, step_size)
        if segment_intersects_triangles(nodes[idx].pos, new_pos, tris) or point_in_collision(new_pos, tris):
            if animate and it % 50 == 0:
                plt.pause(0.001)
            continue
        nodes.append(Node(new_pos, parent=idx, cost=nodes[idx].cost + np.linalg.norm(new_pos - nodes[idx].pos)))

        # draw edge
        if animate:
            ax.plot([nodes[idx].pos[0], new_pos[0]], [nodes[idx].pos[1], new_pos[1]], color='skyblue', linewidth=0.8, alpha=0.7)
            if it % 10 == 0:
                plt.pause(0.001)

        if np.linalg.norm(new_pos - goal) <= step_size:
            # connect to goal
            if not segment_intersects_triangles(new_pos, goal, tris):
                nodes.append(Node(goal, parent=len(nodes) - 1, cost=0))
                break
    # reconstruct path
    path = []
    cur = len(nodes) - 1
    while cur is not None and cur >= 0:
        path.append(nodes[cur].pos)
        cur = nodes[cur].parent if nodes[cur].parent is not None else None
    path = list(reversed(path))
    t1 = time.time()
    stats = {
        'nodes': len(nodes),
        'time_s': t1 - t0,
        'path_len': float(sum(np.linalg.norm(path[i+1] - path[i]) for i in range(len(path)-1))) if len(path) > 1 else float('inf')
    }
    return path, stats


def rrt_star(start: np.ndarray, goal: np.ndarray, tris: List[np.ndarray],
             step_size: float, goal_sample_rate: float, max_iters: int,
             ax, animate: bool = True) -> Tuple[List[np.ndarray], Dict[str, float]]:
    nodes: List[Node] = [Node(start, parent=None, cost=0.0)]
    t0 = time.time()
    found_goal_idx: Optional[int] = None
    for it in range(max_iters):
        rnd = goal if random.random() < goal_sample_rate else np.array([random.random(), random.random()])
        idx = nearest(nodes, rnd)
        new_pos = steer(nodes[idx].pos, rnd, step_size)
        if segment_intersects_triangles(nodes[idx].pos, new_pos, tris) or point_in_collision(new_pos, tris):
            if animate and it % 50 == 0:
                plt.pause(0.001)
            continue

        # choose parent among neighbors
        new_node = Node(new_pos, parent=idx, cost=nodes[idx].cost + np.linalg.norm(new_pos - nodes[idx].pos))
        nodes.append(new_node)
        new_idx = len(nodes) - 1

        # neighbor radius (RRT*): gamma * (log(n)/n)^(1/d)
        n = len(nodes)
        radius = min(0.25, 2.0 * math.sqrt(math.log(max(n, 2)) / max(n, 2)))
        neigh_idxs = [i for i, nd in enumerate(nodes[:-1]) if np.linalg.norm(nd.pos - new_pos) <= radius]
        # choose best parent
        best_parent = new_node.parent
        best_cost = new_node.cost
        for i in neigh_idxs:
            cand_cost = nodes[i].cost + np.linalg.norm(nodes[i].pos - new_pos)
            if cand_cost + 1e-9 < best_cost and not segment_intersects_triangles(nodes[i].pos, new_pos, tris):
                best_parent = i
                best_cost = cand_cost
        if best_parent != new_node.parent:
            nodes[new_idx].parent = best_parent
            nodes[new_idx].cost = best_cost

        # rewire neighbors
        for i in neigh_idxs:
            new_cost = nodes[new_idx].cost + np.linalg.norm(nodes[i].pos - new_pos)
            if new_cost + 1e-9 < nodes[i].cost and not segment_intersects_triangles(nodes[i].pos, new_pos, tris):
                nodes[i].parent = new_idx
                nodes[i].cost = new_cost

        if animate:
            pidx = nodes[new_idx].parent
            if pidx is not None:
                ax.plot([nodes[pidx].pos[0], new_pos[0]], [nodes[pidx].pos[1], new_pos[1]], color='orchid', linewidth=0.8, alpha=0.6)
            if it % 10 == 0:
                plt.pause(0.001)

        if np.linalg.norm(new_pos - goal) <= step_size and not segment_intersects_triangles(new_pos, goal, tris):
            nodes.append(Node(goal, parent=new_idx, cost=nodes[new_idx].cost + np.linalg.norm(goal - new_pos)))
            found_goal_idx = len(nodes) - 1
            break

    # reconstruct path
    path = []
    if found_goal_idx is None and len(nodes) > 1:
        # pick closest to goal
        found_goal_idx = int(np.argmin([np.linalg.norm(nd.pos - goal) for nd in nodes]))
    cur = found_goal_idx
    while cur is not None:
        path.append(nodes[cur].pos)
        cur = nodes[cur].parent
    path = list(reversed(path))
    t1 = time.time()
    stats = {
        'nodes': len(nodes),
        'time_s': t1 - t0,
        'path_len': float(sum(np.linalg.norm(path[i+1] - path[i]) for i in range(len(path)-1))) if len(path) > 1 else float('inf')
    }
    return path, stats


def birrt(start: np.ndarray, goal: np.ndarray, tris: List[np.ndarray],
          step_size: float, max_iters: int, ax, animate: bool = True) -> Tuple[List[np.ndarray], Dict[str, float]]:
    tree_a: List[Node] = [Node(start, parent=None)]
    tree_b: List[Node] = [Node(goal, parent=None)]

    def extend(tree: List[Node], target: np.ndarray) -> Optional[int]:
        idx = nearest(tree, target)
        new_pos = steer(tree[idx].pos, target, step_size)
        if segment_intersects_triangles(tree[idx].pos, new_pos, tris) or point_in_collision(new_pos, tris):
            return None
        tree.append(Node(new_pos, parent=idx))
        new_idx = len(tree) - 1
        if animate:
            ax.plot([tree[idx].pos[0], new_pos[0]], [tree[idx].pos[1], new_pos[1]], color='forestgreen', linewidth=0.9, alpha=0.7)
        return new_idx

    t0 = time.time()
    for it in range(max_iters):
        # sample random point
        rnd = np.array([random.random(), random.random()])
        # extend A towards rnd
        a_idx = extend(tree_a, rnd)
        if a_idx is not None:
            # try connect B towards new node
            while True:
                b_idx = extend(tree_b, tree_a[a_idx].pos)
                if b_idx is None:
                    break
                if np.linalg.norm(tree_b[b_idx].pos - tree_a[a_idx].pos) <= step_size and not segment_intersects_triangles(tree_b[b_idx].pos, tree_a[a_idx].pos, tris):
                    # connected
                    if animate:
                        plt.pause(0.001)
                    # build path
                    path_a = []
                    cur = a_idx
                    while cur is not None:
                        path_a.append(tree_a[cur].pos)
                        cur = tree_a[cur].parent
                    path_a = list(reversed(path_a))
                    path_b = []
                    cur = b_idx
                    while cur is not None:
                        path_b.append(tree_b[cur].pos)
                        cur = tree_b[cur].parent
                    path = path_a + path_b
                    t1 = time.time()
                    stats = {
                        'nodes': len(tree_a) + len(tree_b),
                        'time_s': t1 - t0,
                        'path_len': float(sum(np.linalg.norm(path[i+1] - path[i]) for i in range(len(path)-1)))
                    }
                    return path, stats
        # swap trees
        tree_a, tree_b = tree_b, tree_a
        if animate and it % 10 == 0:
            plt.pause(0.001)

    # failed to connect; return best partial
    # choose closest pair
    minpair = (None, None, float('inf'))
    for i, na in enumerate(tree_a):
        for j, nb in enumerate(tree_b):
            d = np.linalg.norm(na.pos - nb.pos)
            if d < minpair[2]:
                minpair = (i, j, d)
    i, j, _ = minpair
    path = []
    if i is not None:
        cur = i
        while cur is not None:
            path.append(tree_a[cur].pos)
            cur = tree_a[cur].parent
        path = list(reversed(path))
    if j is not None:
        cur = j
        while cur is not None:
            path.append(tree_b[cur].pos)
            cur = tree_b[cur].parent
    t1 = time.time()
    stats = {'nodes': len(tree_a) + len(tree_b), 'time_s': t1 - t0, 'path_len': float('inf')}
    return path, stats


# -----------------------------
# A* on grid
# -----------------------------

def rasterize_obstacles(tris: List[np.ndarray], grid_n: int) -> np.ndarray:
    """Rasterize only interior obstacles (ignore the 4 outer boundary triangles)."""
    # Filter to triangles fully inside [0,1]^2 so outer boundary triangles don't block A*
    inner_tris = [t for t in tris if np.all(t >= 0.0) and np.all(t <= 1.0)]

    occ = np.zeros((grid_n, grid_n), dtype=bool)
    xs = np.linspace(0, 1, grid_n, endpoint=False) + 0.5 / grid_n
    ys = np.linspace(0, 1, grid_n, endpoint=False) + 0.5 / grid_n
    for ix, x in enumerate(xs):
        for iy, y in enumerate(ys):
            # Sample 5 points per cell (center + 4 offsets) for robustness
            samples = [
                np.array([x, y]),
                np.array([min(max(x + 0.25 / grid_n, 0.0), 1.0), y]),
                np.array([min(max(x - 0.25 / grid_n, 0.0), 1.0), y]),
                np.array([x, min(max(y + 0.25 / grid_n, 0.0), 1.0)]),
                np.array([x, min(max(y - 0.25 / grid_n, 0.0), 1.0)]),
            ]
            # Exclude exact boundary samples from triangle tests (avoid marking edges as blocked)
            def is_inside_workspace(p):
                return (0.0 < p[0] < 1.0) and (0.0 < p[1] < 1.0)
            samples = [p for p in samples if is_inside_workspace(p)]
            for tri in inner_tris:
                if any(point_in_triangle(p, tri) for p in samples):
                    occ[iy, ix] = True  # row=y, col=x
                    break
    return occ


def smooth_path(path: List[np.ndarray], tris: List[np.ndarray]) -> List[np.ndarray]:
    """Smooth path by removing unnecessary waypoints using line-of-sight checks."""
    if len(path) <= 2:
        return path

    smoothed = [path[0]]
    i = 0
    while i < len(path) - 1:
        # Find the farthest point we can reach directly
        farthest = i + 1
        for j in range(i + 2, len(path)):
            if not segment_intersects_triangles(path[i], path[j], tris):
                farthest = j
            else:
                break
        smoothed.append(path[farthest])
        i = farthest

    return smoothed


def a_star(start: np.ndarray, goal: np.ndarray, tris: List[np.ndarray], grid_n: int, ax, animate: bool = True, inflate: int = 0) -> Tuple[List[np.ndarray], Dict[str, float]]:
    occ = rasterize_obstacles(tris, grid_n)
    if inflate and inflate > 0:
        occ = (maximum_filter(occ.astype(np.uint8), size=(2*inflate+1, 2*inflate+1)) > 0)

    def to_idx(p: np.ndarray) -> Tuple[int, int]:
        x = min(max(int(p[0] * grid_n), 0), grid_n - 1)
        y = min(max(int(p[1] * grid_n), 0), grid_n - 1)
        return y, x  # row, col

    def to_coord(i: int, j: int) -> np.ndarray:
        return np.array([(j + 0.5) / grid_n, (i + 0.5) / grid_n])

    s = to_idx(start)
    g = to_idx(goal)

    # If start/goal are in occupied cells, snap to nearest free
    def nearest_free(idx: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        from collections import deque
        if not occ[idx]:
            return idx
        q = deque([idx])
        seen = {idx}
        nbrs4 = [(-1,0),(1,0),(0,-1),(0,1)]
        while q:
            ci, cj = q.popleft()
            for di, dj in nbrs4:
                ni, nj = ci + di, cj + dj
                if ni < 0 or nj < 0 or ni >= grid_n or nj >= grid_n:
                    continue
                if (ni, nj) in seen:
                    continue
                if not occ[ni, nj]:
                    return (ni, nj)
                seen.add((ni, nj))
                q.append((ni, nj))
        return None

    s = nearest_free(s)
    g = nearest_free(g)
    if s is None or g is None:
        return [], {'nodes': 0, 'time_s': 0.0, 'path_len': float('inf')}

    import heapq
    open_heap = []
    heapq.heappush(open_heap, (0.0, s))
    came: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {s: None}
    gscore = {s: 0.0}
    closed_set = set()

    # 8-connected for smoother paths
    nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

    def h(i, j):
        # Proper Euclidean distance heuristic
        return math.sqrt((g[0] - i)**2 + (g[1] - j)**2)

    t0 = time.time()
    visited = 0
    max_iterations = grid_n * grid_n  # Prevent infinite loops

    while open_heap and visited < max_iterations:
        _, cur = heapq.heappop(open_heap)

        if cur in closed_set:
            continue

        closed_set.add(cur)
        visited += 1

        if cur == g:
            break

        ci, cj = cur
        for di, dj in nbrs:
            ni, nj = ci + di, cj + dj
            if ni < 0 or nj < 0 or ni >= grid_n or nj >= grid_n:
                continue
            if occ[ni, nj] or (ni, nj) in closed_set:
                continue

            # Diagonal moves cost sqrt(2), orthogonal moves cost 1
            step_cost = math.sqrt(2) if abs(di) + abs(dj) == 2 else 1.0
            tentative = gscore[cur] + step_cost

            if (ni, nj) not in gscore or tentative < gscore[(ni, nj)]:
                gscore[(ni, nj)] = tentative
                came[(ni, nj)] = cur
                f = tentative + h(ni, nj)
                heapq.heappush(open_heap, (f, (ni, nj)))

        # Progress indication - update status periodically
        if animate and visited % 500 == 0:
            progress = min(100, int(100 * visited / (grid_n * grid_n * 0.1)))  # Estimate progress
            ax.set_title(f'A* Progress: {progress}% ({visited} nodes explored)')
            plt.pause(0.001)

    # Reconstruct path in grid indices
    idx_path: List[Tuple[int, int]] = []
    cur = g if g in came else None
    while cur is not None:
        idx_path.append(cur)
        cur = came[cur]
    idx_path = list(reversed(idx_path))

    # Convert to coordinates
    path: List[np.ndarray] = []
    if idx_path:
        for i, j in idx_path:
            path.append(to_coord(i, j))

    # Smooth the path to reduce oscillations
    if len(path) > 2:
        path = smooth_path(path, tris)

    # Ensure start and goal are exact
    if path and len(path) > 0:
        path[0] = start.copy()
        path[-1] = goal.copy()

    t1 = time.time()
    stats = {
        'nodes': visited,
        'time_s': t1 - t0,
        'path_len': float(sum(np.linalg.norm(path[i+1] - path[i]) for i in range(len(path)-1))) if len(path) > 1 else float('inf')
    }
    return path, stats


# -----------------------------
# UI and app
# -----------------------------

class PlannerApp:
    def __init__(self):
        self.limits = ((0, 1), (0, 1))
        self.tris: List[np.ndarray] = gen_obstacles()
        self.start: Optional[np.ndarray] = None
        self.goal: Optional[np.ndarray] = None

        # Larger figure with right-side control panel
        self.fig, self.ax = plt.subplots(figsize=(12, 7))
        plt.subplots_adjust(left=0.06, right=0.74, bottom=0.07, top=0.95)
        self.ax.set_xlim(*self.limits[0])
        self.ax.set_ylim(*self.limits[1])
        self.ax.set_aspect('equal')
        self.ax.set_title('Path Planning (Click to set Start → Goal)')

        # Convenience function to create panel axes by row index
        # Panel area: x from 0.76 to 0.98; 12 rows
        self._panel_left = 0.76
        self._panel_right = 0.98
        self._panel_top = 0.95
        self._panel_bottom = 0.07
        self._rows = 14

        def panel_axes(row: int, rowspan: int = 1, height_pad: float = 0.0):
            total_h = self._panel_top - self._panel_bottom
            rh = total_h / self._rows
            h = rh * rowspan - height_pad
            y1 = self._panel_top - rh * (row + rowspan)
            return self.fig.add_axes([self._panel_left, y1, self._panel_right - self._panel_left, h])

        # UI widgets with improved spacing
        ax_algo = panel_axes(0, rowspan=4)
        self.radio_algo = RadioButtons(ax_algo, ("RRT", "RRT*", "BiRRT", "A*"), active=0)
        self.radio_algo.on_clicked(lambda _: self.update_status("Algorithm changed"))

        ax_plan = panel_axes(4)
        self.btn_plan = Button(ax_plan, 'Plan')
        self.btn_plan.on_clicked(self.on_plan)

        ax_reset = panel_axes(5)
        self.btn_reset = Button(ax_reset, 'Reset')
        self.btn_reset.on_clicked(self.on_reset)

        ax_regen = panel_axes(6)
        self.btn_regen = Button(ax_regen, 'New Obstacles')
        self.btn_regen.on_clicked(self.on_regen)

        # Algorithm-specific sliders with better ranges and descriptions
        ax_step = panel_axes(7)
        self.sld_step = Slider(ax_step, 'Step Size', 0.01, 0.1, valinit=0.05)
        self.sld_step.label.set_text('Step Size (RRT/RRT*/BiRRT)')

        ax_goalrate = panel_axes(8)
        self.sld_goalrate = Slider(ax_goalrate, 'Goal Bias', 0.0, 0.3, valinit=0.1)
        self.sld_goalrate.label.set_text('Goal Bias % (RRT/RRT*)')

        ax_maxit = panel_axes(9)
        self.sld_maxit = Slider(ax_maxit, 'Max Iters', 500, 10000, valinit=3000, valstep=100)
        self.sld_maxit.label.set_text('Max Iterations (RRT family)')

        ax_grid = panel_axes(10)
        self.sld_grid = Slider(ax_grid, 'Grid Size', 50, 200, valinit=100, valstep=10)
        self.sld_grid.label.set_text('Grid Resolution (A*)')

        ax_infl = panel_axes(11)
        self.sld_inflate = Slider(ax_infl, 'Inflation', 0, 3, valinit=1, valstep=1)
        self.sld_inflate.label.set_text('Obstacle Inflation (A*)')

        # Add visualization options
        ax_vis = panel_axes(12)
        self.chk_vis = CheckButtons(ax_vis, ["Show Tree", "Show Grid"], [True, False])

        # Status and metrics
        self.status_text = self.fig.text(0.06, 0.965, '', va='top', fontsize=10)
        self.metrics_text = self.fig.text(0.06, 0.03, '', va='bottom', fontsize=10)

        self.cid_click = self.fig.canvas.mpl_connect('button_press_event', self.on_click)

        self.draw_scene()

    def update_status(self, msg: str):
        self.status_text.set_text(msg)
        self.fig.canvas.draw_idle()

    def draw_scene(self):
        self.ax.cla()
        self.ax.set_xlim(*self.limits[0])
        self.ax.set_ylim(*self.limits[1])
        self.ax.set_aspect('equal')
        self.ax.set_title('Path Planning (Click to set Start → Goal)')
        # obstacles
        for tri in self.tris:
            self.ax.add_patch(Polygon(tri, color='red', alpha=0.8, label='_ob'))
        # start/goal
        handles = []
        labels = []
        if self.start is not None:
            h1 = self.ax.scatter([self.start[0]], [self.start[1]], c='lime', s=70, label='Start', edgecolors='black', zorder=5)
            handles.append(h1)
            labels.append('Start')
        if self.goal is not None:
            h2 = self.ax.scatter([self.goal[0]], [self.goal[1]], c='gold', s=70, label='Goal', edgecolors='black', zorder=5)
            handles.append(h2)
            labels.append('Goal')
        if handles:
            self.ax.legend(handles, labels, loc='upper right')
        self.fig.canvas.draw_idle()

    def on_click(self, event):
        if event.inaxes != self.ax:
            return
        p = np.array([event.xdata, event.ydata])
        if self.start is None:
            if point_in_collision(p, self.tris):
                self.update_status('Start inside obstacle. Pick another point.')
                return
            self.start = p
            self.update_status('Start set. Click to set goal.')
        elif self.goal is None:
            if point_in_collision(p, self.tris):
                self.update_status('Goal inside obstacle. Pick another point.')
                return
            self.goal = p
            self.update_status('Goal set. Press Plan.')
        else:
            # reset cycle: start -> goal
            self.start = p
            self.goal = None
            self.update_status('Start updated. Click to set goal.')
        self.draw_scene()

    def on_reset(self, _):
        self.start = None
        self.goal = None
        self.metrics_text.set_text('')
        self.update_status('Reset. Click to set start.')
        self.draw_scene()

    def on_regen(self, _):
        self.tris = gen_obstacles()
        self.on_reset(_)

    def on_plan(self, _):
        if self.start is None or self.goal is None:
            self.update_status('Pick start and goal first.')
            return

        algo = self.radio_algo.value_selected
        step = float(self.sld_step.val)
        goal_rate = float(self.sld_goalrate.val)
        max_it = int(self.sld_maxit.val)
        grid_n = int(self.sld_grid.val)
        show_tree, show_grid = self.chk_vis.get_status()

        self.draw_scene()
        self.update_status(f'Planning with {algo}...')

        # Force UI update before starting computation
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        t0 = time.time()
        if algo == 'RRT':
            path, stats = rrt(self.start, self.goal, self.tris, step, goal_rate, max_it, self.ax, show_tree)
        elif algo == 'RRT*':
            path, stats = rrt_star(self.start, self.goal, self.tris, step, goal_rate, max_it, self.ax, show_tree)
        elif algo == 'BiRRT':
            path, stats = birrt(self.start, self.goal, self.tris, step, max_it, self.ax, show_tree)
        else:  # A*
            inflate = int(self.sld_inflate.val)
            path, stats = a_star(self.start, self.goal, self.tris, grid_n, self.ax, show_grid, inflate)

        # Reset title after A* progress updates
        self.ax.set_title('Path Planning (Click to set Start → Goal)')

        if len(path) >= 2 and stats['path_len'] < float('inf'):
            xs = [p[0] for p in path]
            ys = [p[1] for p in path]
            self.ax.plot(xs, ys, color='deepskyblue', linewidth=3.0, alpha=0.9, zorder=10)
            self.update_status(f'{algo} completed successfully!')
        else:
            self.update_status(f'{algo} failed to find a path.')

        self.fig.canvas.draw_idle()

        # Display comprehensive metrics
        took = stats.get('time_s', time.time() - t0)
        extra = ''
        if algo == 'A*':
            extra = f" | Grid: {grid_n}×{grid_n} | Inflation: {int(self.sld_inflate.val)}"
        elif algo in ['RRT', 'RRT*']:
            extra = f" | Step: {step:.3f} | Goal Bias: {goal_rate:.1%}"
        elif algo == 'BiRRT':
            extra = f" | Step: {step:.3f}"

        self.metrics_text.set_text(
            f"Algorithm: {algo}{extra} | Nodes Explored: {stats.get('nodes', 0)} | "
            f"Time: {took:.3f}s | Path Length: {stats.get('path_len', float('inf')):.3f}"
        )


def main():
    app = PlannerApp()
    plt.show()


if __name__ == '__main__':
    main()
