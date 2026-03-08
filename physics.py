# physics.py — Data-Oriented Kinematics and Movement using SIMD Compute Pipelines

import math
import numpy as np
from numba import njit, prange
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from scenario import Unit, Missile

# =============================================================================
# DATA-ORIENTED KERNELS
# Parallel execution for kinematics (Runs across all CPU cores or GPU)
# =============================================================================

@njit(parallel=True, fastmath=True, cache=True)
def _air_kinematics_kernel(
    N: int, dt: float,
    lat: np.ndarray, lon: np.ndarray, alt: np.ndarray,
    heading: np.ndarray, speed: np.ndarray, fuel: np.ndarray, g_load: np.ndarray,
    tgt_heading: np.ndarray, tgt_alt: np.ndarray, tgt_speed: np.ndarray,
    max_g: np.ndarray, perf_mult: np.ndarray, burn_rate: np.ndarray
):
    """Computes flight dynamics for N aircraft simultaneously."""
    for i in prange(N):
        # 1. Altitude Interpolation
        if alt[i] != tgt_alt[i]:
            climb_rate = 166.67 * perf_mult[i]
            diff = tgt_alt[i] - alt[i]
            step = climb_rate * dt
            if abs(diff) <= step:
                alt[i] = tgt_alt[i]
            else:
                alt[i] += math.copysign(step, diff)
        
        # 2. Heading & Turn Rate (G-Load constrained)
        spd_mps = max(10.0, speed[i] / 3.6)
        max_turn = ((max_g[i] * 9.81) / spd_mps * 57.2957795) * dt * perf_mult[i] # 57.29 is 180/pi
        
        diff_h = (tgt_heading[i] - heading[i] + 360.0) % 360.0
        if diff_h > 180.0:
            diff_h -= 360.0
            
        if abs(diff_h) <= max_turn:
            heading[i] = tgt_heading[i]
            g_load[i] = 1.0
        else:
            heading[i] = (heading[i] + math.copysign(max_turn, diff_h)) % 360.0
            g_load[i] = max_g[i] * perf_mult[i]
            
        # 3. Speed & Acceleration
        if g_load[i] > 2.0:
            # Bleed speed in hard turns
            speed[i] = max(250.0, speed[i] - (g_load[i] * 12.0) * dt)
        else:
            if speed[i] < tgt_speed[i]:
                speed[i] = min(tgt_speed[i], speed[i] + 60.0 * dt)
            elif speed[i] > tgt_speed[i]:
                speed[i] = max(tgt_speed[i], speed[i] - 60.0 * dt)
                
        # 4. Geographic Position Update
        move_dist = (speed[i] / 3600.0) * dt
        lat_rad = max(0.0001, math.cos(lat[i] * 0.0174532925)) # pi/180
        lat[i] += (math.cos(heading[i] * 0.0174532925) * move_dist) / 111.32
        lon[i] += (math.sin(heading[i] * 0.0174532925) * move_dist) / (111.32 * lat_rad)
        
        # 5. Fuel Burn
        if fuel[i] > 0.0:
            fuel[i] = max(0.0, fuel[i] - burn_rate[i] * dt)


@njit(parallel=True, fastmath=True, cache=True)
def _missile_kinematics_kernel(
    M: int, dt: float,
    lat: np.ndarray, lon: np.ndarray, alt: np.ndarray, speed: np.ndarray, 
    status: np.ndarray, fraction_travelled: np.ndarray,
    t_lat: np.ndarray, t_lon: np.ndarray, t_alt: np.ndarray, 
    t_speed: np.ndarray, t_gload: np.ndarray,
    flight_profile: np.ndarray, is_gun: np.ndarray, 
    launch_dist: np.ndarray, start_alt: np.ndarray, motor_burnout: np.ndarray
):
    """Computes kinematics and dynamic drag profiles for M missiles simultaneously."""
    for j in prange(M):
        if status[j] != 1: 
            continue # 1 = IN_FLIGHT
        
        # True 3D Slant Range to Target
        dlat = t_lat[j] - lat[j]
        dlon = (t_lon[j] - lon[j]) * math.cos((lat[j] + t_lat[j]) * 0.5 * 0.0174532925)
        dist_2d = math.sqrt(dlat*dlat + dlon*dlon) * 111.32
        dist_vert = (alt[j] - t_alt[j]) * 0.0003048
        dist = math.sqrt(dist_2d*dist_2d + dist_vert*dist_vert)
        
        ld = max(1.0, launch_dist[j])
        frac = 1.0 - max(0.0, min(1.0, dist / ld))
        fraction_travelled[j] = frac
        
        # Energy Bleed & Motor Burnout
        if not is_gun[j] and frac > motor_burnout[j]:
            drag = 250.0 if alt[j] > 20000.0 else 600.0
            if t_gload[j] > 3.0: drag *= 1.5
            speed[j] = max(800.0, speed[j] - drag * dt)
            
            # Kinematic defeat
            if speed[j] < t_speed[j] + 50.0 and dist > 2.0:
                status[j] = 0 # 0 = LOST ENERGY
                continue
                
        move_dist = (speed[j] / 3600.0) * dt
        
        # Specific Flight Profiles
        fp = flight_profile[j]
        if fp == 1: # lofted
            if frac < 0.5: alt[j] = start_alt[j] + (60000.0 - start_alt[j]) * (frac * 2.0)
            else: alt[j] = 60000.0 - (60000.0 - t_alt[j]) * ((frac - 0.5) * 2.0)
        elif fp == 2: # sea_skimming
            if frac > 0.05: alt[j] = max(30.0, t_alt[j])
        elif fp == 3: # terrain_following
            if frac > 0.10: alt[j] = max(200.0, t_alt[j])
        elif fp == 4: # ballistic
            peak = ld * 1000.0
            alt[j] = math.sin(frac * math.pi) * peak + t_alt[j]
            
        # Intercept Logic
        if dist <= move_dist:
            status[j] = 2 # 2 = READY TO DETONATE
            lat[j] = t_lat[j]
            lon[j] = t_lon[j]
            alt[j] = t_alt[j]
        else:
            ratio = move_dist / dist
            lat[j] += (t_lat[j] - lat[j]) * ratio
            lon[j] += (t_lon[j] - lon[j]) * ratio
            alt[j] += (t_alt[j] - alt[j]) * ratio


# =============================================================================
# PIPELINE MANAGER
# =============================================================================

class KinematicsComputePipeline:
    def __init__(self):
        self._fp_map = {
            "direct": 0, "lofted": 1, "sea_skimming": 2, 
            "terrain_following": 3, "ballistic": 4
        }
        
        # Persistent buffers for Air Units
        self._air_cap = 0
        self.a_lat = None
        self.a_lon = None
        self.a_alt = None
        self.a_heading = None
        self.a_speed = None
        self.a_fuel = None
        self.a_g_load = None
        self.a_tgt_heading = None
        self.a_tgt_alt = None
        self.a_tgt_speed = None
        self.a_max_g = None
        self.a_perf_mult = None
        self.a_burn_rate = None

        # Persistent buffers for Missiles
        self._msl_cap = 0
        self.m_lat = None
        self.m_lon = None
        self.m_alt = None
        self.m_speed = None
        self.m_status = None
        self.m_frac = None
        self.m_t_lat = None
        self.m_t_lon = None
        self.m_t_alt = None
        self.m_t_speed = None
        self.m_t_gload = None
        self.m_fp = None
        self.m_is_gun = None
        self.m_launch_dist = None
        self.m_start_alt = None
        self.m_burnout = None

    def _ensure_air_capacity(self, required_size: int):
        if required_size <= self._air_cap:
            return
        # Exponential growth to minimize reallocations
        new_cap = max(1024, self._air_cap * 2)
        while new_cap < required_size:
            new_cap *= 2
            
        self.a_lat = np.zeros(new_cap, dtype=np.float64)
        self.a_lon = np.zeros(new_cap, dtype=np.float64)
        self.a_alt = np.zeros(new_cap, dtype=np.float64)
        self.a_heading = np.zeros(new_cap, dtype=np.float64)
        self.a_speed = np.zeros(new_cap, dtype=np.float64)
        self.a_fuel = np.zeros(new_cap, dtype=np.float64)
        self.a_g_load = np.zeros(new_cap, dtype=np.float64)
        
        self.a_tgt_heading = np.zeros(new_cap, dtype=np.float64)
        self.a_tgt_alt = np.zeros(new_cap, dtype=np.float64)
        self.a_tgt_speed = np.zeros(new_cap, dtype=np.float64)
        
        self.a_max_g = np.zeros(new_cap, dtype=np.float64)
        self.a_perf_mult = np.zeros(new_cap, dtype=np.float64)
        self.a_burn_rate = np.zeros(new_cap, dtype=np.float64)
        
        self._air_cap = new_cap

    def _ensure_missile_capacity(self, required_size: int):
        if required_size <= self._msl_cap:
            return
        
        new_cap = max(512, self._msl_cap * 2)
        while new_cap < required_size:
            new_cap *= 2

        self.m_lat = np.zeros(new_cap, dtype=np.float64)
        self.m_lon = np.zeros(new_cap, dtype=np.float64)
        self.m_alt = np.zeros(new_cap, dtype=np.float64)
        self.m_speed = np.zeros(new_cap, dtype=np.float64)
        self.m_status = np.zeros(new_cap, dtype=np.int8)
        self.m_frac = np.zeros(new_cap, dtype=np.float64)
        
        self.m_t_lat = np.zeros(new_cap, dtype=np.float64)
        self.m_t_lon = np.zeros(new_cap, dtype=np.float64)
        self.m_t_alt = np.zeros(new_cap, dtype=np.float64)
        self.m_t_speed = np.zeros(new_cap, dtype=np.float64)
        self.m_t_gload = np.zeros(new_cap, dtype=np.float64)
        
        self.m_fp = np.zeros(new_cap, dtype=np.int8)
        self.m_is_gun = np.zeros(new_cap, dtype=np.int8)
        self.m_launch_dist = np.zeros(new_cap, dtype=np.float64)
        self.m_start_alt = np.zeros(new_cap, dtype=np.float64)
        self.m_burnout = np.zeros(new_cap, dtype=np.float64)
        
        self._msl_cap = new_cap

    def step_air_units(self, air_units: List['Unit'], dt: float):
        N = len(air_units)
        if N == 0: return
        
        self._ensure_air_capacity(N)
        
        # 1. Pack Arrays into persistent buffers
        for i, u in enumerate(air_units):
            self.a_lat[i] = u.lat
            self.a_lon[i] = u.lon
            self.a_alt[i] = u.altitude_ft
            self.a_heading[i] = u.heading
            self.a_speed[i] = u.current_speed_kmh
            self.a_fuel[i] = u.fuel_kg
            self.a_g_load[i] = getattr(u, 'current_g_load', 1.0)
            
            self.a_tgt_heading[i] = u.target_heading
            self.a_tgt_alt[i] = u.target_altitude_ft
            
            # Resolve Target Speed and Fuel Burn dynamically before the kernel
            throttle_spd_mult = {"LOITER": 0.6, "CRUISE": 0.8, "FLANK": 1.1}.get(u.throttle_state, 0.8)
            t_spd = u.platform.speed_kmh * u.performance_mult * throttle_spd_mult
            
            # Overrides
            if getattr(u, 'is_evading', False): t_spd = u.platform.speed_kmh * u.performance_mult * 1.1 
            elif getattr(u, '_cached_tot_speed', 0.0) > 0: t_spd = getattr(u, '_cached_tot_speed')
            elif getattr(u, 'formation_target_speed', 0.0) > 0 and getattr(u, 'leader_uid', ""): t_spd = getattr(u, 'formation_target_speed')
            
            self.a_tgt_speed[i] = t_spd
            
            self.a_max_g[i] = u.platform.max_g
            self.a_perf_mult[i] = u.performance_mult
            
            throttle_fuel_mult = {"LOITER": 0.5, "CRUISE": 1.0, "FLANK": 3.0}.get(u.throttle_state, 1.0)
            if getattr(u, 'is_evading', False): throttle_fuel_mult = 3.0
            self.a_burn_rate[i] = (u.platform.fuel_burn_rate_kg_h / 3600.0) * throttle_fuel_mult * (1.0 + getattr(u, 'inefficiency_penalty', 0.0))

        # 2. Execute Hardware Kernel (Passing only the active slice [:N])
        _air_kinematics_kernel(
            N, dt, self.a_lat[:N], self.a_lon[:N], self.a_alt[:N], 
            self.a_heading[:N], self.a_speed[:N], self.a_fuel[:N], self.a_g_load[:N], 
            self.a_tgt_heading[:N], self.a_tgt_alt[:N], self.a_tgt_speed[:N], 
            self.a_max_g[:N], self.a_perf_mult[:N], self.a_burn_rate[:N]
        )

        # 3. Unpack Arrays back to OOP State
        for i, u in enumerate(air_units):
            u.lat = self.a_lat[i]
            u.lon = self.a_lon[i]
            u.altitude_ft = self.a_alt[i]
            u.heading = self.a_heading[i]
            u.current_speed_kmh = self.a_speed[i]
            u.fuel_kg = self.a_fuel[i]
            u.current_g_load = self.a_g_load[i]
            if u.fuel_kg == 0.0:
                u.take_damage(999.0) # Out of fuel crash

    def step_missiles(self, missiles: List['Missile'], dt: float):
        M = len(missiles)
        if M == 0: return
        
        self._ensure_missile_capacity(M)
        
        # 1. Pack Arrays into persistent buffers
        for j, m in enumerate(missiles):
            self.m_lat[j] = m.lat
            self.m_lon[j] = m.lon
            self.m_alt[j] = m.altitude_ft
            self.m_speed[j] = m.eff_speed_kmh
            self.m_status[j] = 1 if m.active and m.status == "IN_FLIGHT" else 0
            
            self.m_t_lat[j] = m.impact_lat if m.is_ballistic else m.target.lat
            self.m_t_lon[j] = m.impact_lon if m.is_ballistic else m.target.lon
            self.m_t_alt[j] = m.impact_alt_ft if m.is_ballistic else m.target.altitude_ft
            self.m_t_speed[j] = getattr(m.target, 'current_speed_kmh', 0.0)
            self.m_t_gload[j] = getattr(m.target, 'current_g_load', 1.0)
            
            self.m_fp[j] = self._fp_map.get(m.wdef.flight_profile, 0)
            self.m_is_gun[j] = 1 if m.wdef.is_gun else 0
            self.m_launch_dist[j] = m.launch_dist
            self.m_start_alt[j] = m.shooter.altitude_ft
            self.m_burnout[j] = m.motor_burnout_fraction
            
            # Save trail directly in Python
            if self.m_status[j] == 1:
                m.trail.append((m.lat, m.lon))

        # 2. Execute Hardware Kernel (Passing only the active slice [:M])
        _missile_kinematics_kernel(
            M, dt, self.m_lat[:M], self.m_lon[:M], self.m_alt[:M], self.m_speed[:M], 
            self.m_status[:M], self.m_frac[:M],
            self.m_t_lat[:M], self.m_t_lon[:M], self.m_t_alt[:M], self.m_t_speed[:M], self.m_t_gload[:M], 
            self.m_fp[:M], self.m_is_gun[:M], self.m_launch_dist[:M], self.m_start_alt[:M], self.m_burnout[:M]
        )

        # 3. Unpack Arrays and Trigger Resolvers
        for j, m in enumerate(missiles):
            if not m.active: continue
            
            m.lat = self.m_lat[j]
            m.lon = self.m_lon[j]
            m.altitude_ft = self.m_alt[j]
            m.eff_speed_kmh = self.m_speed[j]
            
            if self.m_status[j] == 0:
                m.active = False
                m.status = "LOST ENERGY"
            elif self.m_status[j] == 2:
                # Trigger terminal phase resolution flag
                m._terminal_phase_triggered = True