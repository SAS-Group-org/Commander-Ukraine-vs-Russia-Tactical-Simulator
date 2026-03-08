# spatial.py — ECS-style 1D Spatial Hashing Grid

import math
import numpy as np
from numba import njit

@njit(cache=True)
def get_candidate_cells(lat: float, lon: float, max_range_km: float, cell_size_deg: float) -> np.ndarray:
    """
    DATA-ORIENTED MATH: Instantly calculates the bounding box of a sensor sweep 
    and translates it to 1D array indices using compiled C-level execution.
    """
    lat_rad = max(0.0001, math.cos(math.radians(lat)))
    deg_range_lat = max_range_km / 111.32
    deg_range_lon = max_range_km / (111.32 * lat_rad)

    # Shift lat/lon to absolute positive coordinates (0 to 360, 0 to 180)
    min_cx = int((lon + 180.0 - deg_range_lon) / cell_size_deg)
    max_cx = int((lon + 180.0 + deg_range_lon) / cell_size_deg)
    min_cy = int((lat + 90.0 - deg_range_lat) / cell_size_deg)
    max_cy = int((lat + 90.0 + deg_range_lat) / cell_size_deg)

    W = int(360.0 / cell_size_deg) + 1

    # Pre-calculate array size and fill it
    count = max(0, (max_cx - min_cx + 1) * (max_cy - min_cy + 1))
    cells = np.empty(count, dtype=np.int32)
    
    idx = 0
    for x in range(min_cx, max_cx + 1):
        for y in range(min_cy, max_cy + 1):
            cells[idx] = y * W + x
            idx += 1
            
    return cells

class SpatialHashGrid:
    def __init__(self, cell_size_deg=0.5):
        # 0.5 degrees is roughly 55km. A great balance for planes and ground units.
        self.cell_size = cell_size_deg
        
        self.W = int(360.0 / self.cell_size) + 1
        self.H = int(180.0 / self.cell_size) + 1
        
        # DOD OPTIMIZATION: Pre-allocate a 1D array of lists to eliminate dictionary hashing overhead.
        # At 0.5 deg, this is ~259,200 cells taking negligible memory.
        self.cells = [[] for _ in range(self.W * self.H)]
        
        # Track which cells are actually populated this tick for lightning-fast resets
        self.active_cells = set()

    def clear(self):
        """Reset only the cells that were populated this logic tick (Sparse Set clear)."""
        for idx in self.active_cells:
            self.cells[idx].clear()
        self.active_cells.clear()

    def insert(self, unit):
        """Hash a unit directly into the 1D array using pure math offsets."""
        if not unit.alive: 
            return
            
        cx = int((unit.lon + 180.0) / self.cell_size)
        cy = int((unit.lat + 90.0) / self.cell_size)
        
        # Clamp to prevent out-of-bounds errors near the poles or dateline
        cx = max(0, min(self.W - 1, cx))
        cy = max(0, min(self.H - 1, cy))
        
        idx = cy * self.W + cx
        
        self.cells[idx].append(unit)
        self.active_cells.add(idx)

    def get_candidates(self, lat, lon, max_range_km):
        """Fetch units using the JIT-compiled grid math."""
        
        # Use the Numba compute function to instantly grab all candidate cell indices
        cell_indices = get_candidate_cells(lat, lon, max_range_km, self.cell_size)
        
        candidates = []
        max_idx = self.W * self.H - 1
        
        # A simple iteration over a 1D array is drastically faster than dictionary lookups
        for idx in cell_indices:
            if 0 <= idx <= max_idx:
                # Fast truthiness check: only extend if the cell actually has units
                if self.cells[idx]:
                    candidates.extend(self.cells[idx])
                    
        return candidates