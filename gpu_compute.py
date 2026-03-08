# gpu_compute.py — Data-Oriented broad-phase processing for sensors and Line of Sight

import math
import numpy as np
from numba import njit, prange

# =============================================================================
# DATA-ORIENTED KERNEL
# =============================================================================
# To run this strictly on an Nvidia GPU, you would replace @njit with @cuda.jit
# and replace the prange loops with threadIdx.x and blockIdx.x coordinate logic.
# For maximum compatibility out-of-the-box, we use parallel SIMD execution, 
# which acts as a CPU-side "compute shader".
# =============================================================================

@njit(parallel=True, fastmath=True, cache=True)
def _broadphase_sensor_kernel(
    # --- Sensor Arrays (Length N) ---
    s_lat: np.ndarray, s_lon: np.ndarray, s_alt: np.ndarray, 
    s_radar_km: np.ndarray, s_esm_km: np.ndarray, s_ir_km: np.ndarray, 
    s_perf: np.ndarray, s_ew_band: np.ndarray, s_det_mod: np.ndarray,
    
    # --- Target Arrays (Length M) ---
    t_lat: np.ndarray, t_lon: np.ndarray, t_alt: np.ndarray, 
    t_rcs: np.ndarray, t_emitting: np.ndarray, t_jamming: np.ndarray, 
    t_ecm: np.ndarray, t_heading: np.ndarray,
    
    # --- Environment Globals ---
    weather_ir_mod: float, weather_radar_mod: float, cloud_ceiling_ft: float,
    
    # --- Output Matrices (N x M) ---
    out_cls: np.ndarray, out_sen: np.ndarray, out_dist: np.ndarray
):
    """
    Computes detection classification and sensor type for an N x M grid of units.
    Writes to pre-allocated N x M matrices:
    out_cls: 0=NONE, 1=FAINT, 2=PROBABLE, 3=CONFIRMED
    out_sen: 0=NONE, 1=ESM, 2=IR, 3=RADAR
    out_dist: Slant range in km
    """
    N = s_lat.shape[0]
    M = t_lat.shape[0]

    # Parallel loop: Every sensor processes its targets simultaneously
    for i in prange(N):
        for j in range(M):
            # Fast Pythagorean Distance (Flat Earth approx for speed)
            dlat = t_lat[j] - s_lat[i]
            dlon = (t_lon[j] - s_lon[i]) * math.cos((s_lat[i] + t_lat[j]) * 0.5 * 0.0174532925)
            dist_km = math.sqrt(dlat*dlat + dlon*dlon) * 111.32
            
            # Record distance for the matrix output
            out_dist[i, j] = dist_km
            
            max_sensor_range = max(s_radar_km[i], s_esm_km[i], s_ir_km[i])
            if dist_km > max_sensor_range * 1.2:
                continue # Broad-phase cull: Target is out of all possible sensor ranges

            # Inline Line-of-Sight (LoS) Raycast
            has_los = -1
            
            best_cls = 0
            best_sen = 0

            # 1. ESM (Passive radar sniffing - Unaffected by weather)
            if t_emitting[j] and s_esm_km[i] > 0:
                esm_range = s_esm_km[i] * s_det_mod[i]
                if dist_km <= esm_range:
                    if has_los == -1:
                        # Earth Curvature Horizon Check
                        alt1_m = max(s_alt[i] * 0.3048, 5.0)
                        alt2_m = max(t_alt[j] * 0.3048, 5.0)
                        horizon_km = 4.12 * (math.sqrt(alt1_m) + math.sqrt(alt2_m))
                        
                        if dist_km > horizon_km:
                            has_los = 0
                        else:
                            # Procedural DEM Midpoint Raycast
                            mid_lat = (s_lat[i] + t_lat[j]) * 0.5
                            mid_lon = (s_lon[i] + t_lon[j]) * 0.5
                            lat_r, lon_r = round(mid_lat, 3), round(mid_lon, 3)
                            
                            elev = 0.0
                            if lat_r < 46.5 and lon_r < 36.0 and lat_r < (46.5 - (36.0 - lon_r)*0.5):
                                elev = 0.0
                            else:
                                hills = (math.sin(lat_r * 15.0) * math.cos(lon_r * 15.0)) * 400.0
                                if lon_r < 26.0:
                                    elev = max(0.0, 1000.0 + hills + (math.sin(lat_r * 20.0) * math.cos(lon_r * 20.0)) * 3500.0)
                                else:
                                    elev = max(0.0, 300.0 + hills)
                                    
                            has_los = 1 if ((s_alt[i] + t_alt[j]) * 0.5) >= elev else 0

                    if has_los == 1:
                        best_cls = 2
                        best_sen = 1

            # 2. IR / FLIR / Optical (Heavily impacted by Weather)
            ir_range = s_ir_km[i] * s_det_mod[i] * weather_ir_mod
            if dist_km <= ir_range and weather_ir_mod > 0.0:
                if not ((s_alt[i] > cloud_ceiling_ft and t_alt[j] < cloud_ceiling_ft) or 
                        (s_alt[i] < cloud_ceiling_ft and t_alt[j] > cloud_ceiling_ft)):
                    if best_cls < 3:
                        if has_los == -1:
                            alt1_m = max(s_alt[i] * 0.3048, 5.0)
                            alt2_m = max(t_alt[j] * 0.3048, 5.0)
                            horizon_km = 4.12 * (math.sqrt(alt1_m) + math.sqrt(alt2_m))
                            if dist_km > horizon_km: has_los = 0
                            else:
                                mid_lat = (s_lat[i] + t_lat[j]) * 0.5
                                mid_lon = (s_lon[i] + t_lon[j]) * 0.5
                                lat_r, lon_r = round(mid_lat, 3), round(mid_lon, 3)
                                elev = 0.0
                                if not (lat_r < 46.5 and lon_r < 36.0 and lat_r < (46.5 - (36.0 - lon_r)*0.5)):
                                    hills = (math.sin(lat_r * 15.0) * math.cos(lon_r * 15.0)) * 400.0
                                    elev = max(0.0, 1000.0 + hills + (math.sin(lat_r * 20.0) * math.cos(lon_r * 20.0)) * 3500.0) if lon_r < 26.0 else max(0.0, 300.0 + hills)
                                has_los = 1 if ((s_alt[i] + t_alt[j]) * 0.5) >= elev else 0

                        if has_los == 1:
                            best_cls = 3
                            best_sen = 2

            # 3. Active Radar (Band & Aspect-dependent)
            if s_radar_km[i] > 0:
                target_rcs = t_rcs[j]
                
                # Aspect Calculation
                dlon_rad = (t_lon[j] - s_lon[i]) * 0.0174532925
                lat1r = s_lat[i] * 0.0174532925
                lat2r = t_lat[j] * 0.0174532925
                x = math.sin(dlon_rad) * math.cos(lat2r)
                y = math.cos(lat1r) * math.sin(lat2r) - math.sin(lat1r) * math.cos(lat2r) * math.cos(dlon_rad)
                brg_to_tgt = (math.degrees(math.atan2(x, y)) + 360.0) % 360.0
                aspect = abs(t_heading[j] - brg_to_tgt) % 360.0
                
                if (60 <= aspect <= 120) or (240 <= aspect <= 300):
                    target_rcs *= 1.5
                elif (aspect <= 30) or (150 <= aspect <= 210) or (aspect >= 330):
                    target_rcs *= 0.7
                    
                if s_ew_band[i] == 1: target_rcs = max(target_rcs, 1.0)
                
                rcs_ratio = max(target_rcs, 0.01) / 5.0
                R_rcs = (s_radar_km[i] * s_perf[i]) * (rcs_ratio ** 0.25)
                
                ecm_penalty = 0.0
                if t_jamming[j] and dist_km > 25.0:
                    jam_factor = 1.0 - (25.0 / dist_km)**2
                    ecm_penalty = t_ecm[j] * 0.60 * jam_factor
                    
                R_effective = R_rcs * max(0.0, 1.0 - ecm_penalty) * s_det_mod[i] * weather_radar_mod
                
                if R_effective > 0.0 and dist_km <= R_effective:
                    if has_los == -1:
                        alt1_m = max(s_alt[i] * 0.3048, 5.0)
                        alt2_m = max(t_alt[j] * 0.3048, 5.0)
                        horizon_km = 4.12 * (math.sqrt(alt1_m) + math.sqrt(alt2_m))
                        if dist_km > horizon_km: has_los = 0
                        else:
                            mid_lat = (s_lat[i] + t_lat[j]) * 0.5
                            mid_lon = (s_lon[i] + t_lon[j]) * 0.5
                            lat_r, lon_r = round(mid_lat, 3), round(mid_lon, 3)
                            elev = 0.0
                            if not (lat_r < 46.5 and lon_r < 36.0 and lat_r < (46.5 - (36.0 - lon_r)*0.5)):
                                hills = (math.sin(lat_r * 15.0) * math.cos(lon_r * 15.0)) * 400.0
                                elev = max(0.0, 1000.0 + hills + (math.sin(lat_r * 20.0) * math.cos(lon_r * 20.0)) * 3500.0) if lon_r < 26.0 else max(0.0, 300.0 + hills)
                            has_los = 1 if ((s_alt[i] + t_alt[j]) * 0.5) >= elev else 0

                    if has_los == 1:
                        fraction = dist_km / R_effective
                        if fraction <= 0.50: cls = 3
                        elif fraction <= 0.75: cls = 2
                        else: cls = 1
                        
                        if s_ew_band[i] == 1 and cls == 3: cls = 2
                        
                        if cls > best_cls:
                            best_cls = cls
                            best_sen = 3

            out_cls[i, j] = best_cls
            out_sen[i, j] = best_sen

# =============================================================================
# DATA PIPELINE MANAGER
# =============================================================================
class SensorComputePipeline:
    def __init__(self):
        self.sensor_uids = []
        self.target_uids = []
        
        # Capacity trackers
        self._s_cap = 0
        self._t_cap = 0
        
        # Sensor persistent buffers
        self.s_lat = None
        self.s_lon = None
        self.s_alt = None
        self.s_radar_km = None
        self.s_esm_km = None
        self.s_ir_km = None
        self.s_perf = None
        self.s_ew_band = None
        self.s_det_mod = None
        
        # Target persistent buffers
        self.t_lat = None
        self.t_lon = None
        self.t_alt = None
        self.t_rcs = None
        self.t_emitting = None
        self.t_jamming = None
        self.t_ecm = None
        self.t_heading = None
        
        # Output Matrices
        self.out_cls = None
        self.out_sen = None
        self.out_dist = None

    def _ensure_capacities(self, req_s: int, req_t: int):
        resize_s = req_s > self._s_cap
        resize_t = req_t > self._t_cap
        
        if not (resize_s or resize_t):
            return
            
        if resize_s:
            new_s = max(128, self._s_cap * 2)
            while new_s < req_s: 
                new_s *= 2
                
            self.s_lat = np.zeros(new_s, dtype=np.float32)
            self.s_lon = np.zeros(new_s, dtype=np.float32)
            self.s_alt = np.zeros(new_s, dtype=np.float32)
            self.s_radar_km = np.zeros(new_s, dtype=np.float32)
            self.s_esm_km = np.zeros(new_s, dtype=np.float32)
            self.s_ir_km = np.zeros(new_s, dtype=np.float32)
            self.s_perf = np.zeros(new_s, dtype=np.float32)
            self.s_ew_band = np.zeros(new_s, dtype=np.int8)
            self.s_det_mod = np.zeros(new_s, dtype=np.float32)
            self._s_cap = new_s
            
        if resize_t:
            new_t = max(128, self._t_cap * 2)
            while new_t < req_t: 
                new_t *= 2
                
            self.t_lat = np.zeros(new_t, dtype=np.float32)
            self.t_lon = np.zeros(new_t, dtype=np.float32)
            self.t_alt = np.zeros(new_t, dtype=np.float32)
            self.t_rcs = np.zeros(new_t, dtype=np.float32)
            self.t_emitting = np.zeros(new_t, dtype=np.int8)
            self.t_jamming = np.zeros(new_t, dtype=np.int8)
            self.t_ecm = np.zeros(new_t, dtype=np.float32)
            self.t_heading = np.zeros(new_t, dtype=np.float32)
            self._t_cap = new_t
            
        # If either dimension grew, reallocate the matrices
        self.out_cls = np.zeros((self._s_cap, self._t_cap), dtype=np.int8)
        self.out_sen = np.zeros((self._s_cap, self._t_cap), dtype=np.int8)
        self.out_dist = np.full((self._s_cap, self._t_cap), 9999.0, dtype=np.float32)

    def _pack_sensor_arrays(self, sensors):
        self.sensor_uids.clear()
        for i, s in enumerate(sensors):
            self.sensor_uids.append(s.uid)
            self.s_lat[i] = s.lat
            self.s_lon[i] = s.lon
            self.s_alt[i] = s.altitude_ft
            self.s_radar_km[i] = s.platform.radar_range_km if getattr(s, 'search_radar_active', True) else 0.0
            self.s_esm_km[i] = s.platform.esm_range_km
            self.s_ir_km[i] = s.platform.ir_range_km
            self.s_perf[i] = s.performance_mult
            self.s_ew_band[i] = 1 if getattr(s.platform, 'radar_band', 'fire_control') in ("early_warning", "volume_search") else 0
            self.s_det_mod[i] = 1.0 - getattr(s, 'inefficiency_penalty', 0.0)

    def _pack_target_arrays(self, targets):
        self.target_uids.clear()
        for j, t in enumerate(targets):
            self.target_uids.append(t.uid)
            self.t_lat[j] = t.lat
            self.t_lon[j] = t.lon
            self.t_alt[j] = t.altitude_ft
            self.t_rcs[j] = t.platform.rcs_m2
            self.t_emitting[j] = 1 if (getattr(t, 'search_radar_active', False) or getattr(t, 'fc_radar_active', False)) else 0
            self.t_jamming[j] = 1 if t.is_jamming else 0
            self.t_ecm[j] = t.platform.ecm_rating
            self.t_heading[j] = t.heading

    def run_sweep(self, sensors: list, targets: list, weather: str = "CLEAR", time_of_day: str = "DAY") -> dict:
        """
        Executes the SIMD Kernel using persistent memory buffers.
        Returns: dict of target_uid -> (best_cls_str, best_sen_str, min_detect_dist_km)
        """
        if not sensors or not targets:
            return {}

        N = len(sensors)
        M = len(targets)
        
        # 1. Expand arrays if necessary
        self._ensure_capacities(N, M)
        
        # 2. Pack Python objects into flat C-arrays
        self._pack_sensor_arrays(sensors)
        self._pack_target_arrays(targets)

        # 3. Wipe ONLY the active slice of our output matrices
        self.out_cls[:N, :M].fill(0)
        self.out_sen[:N, :M].fill(0)
        self.out_dist[:N, :M].fill(9999.0)

        # 4. Determine Global Modifiers
        ir_mod = 1.0
        if time_of_day == "NIGHT": ir_mod *= 0.8
        if weather == "RAIN": ir_mod *= 0.4
        elif weather == "STORM": ir_mod *= 0.1
        elif weather == "OVERCAST": ir_mod *= 0.7
        
        cloud_ceiling_ft = {"CLEAR": 50000.0, "OVERCAST": 10000.0, "RAIN": 5000.0, "STORM": 3000.0}.get(weather, 50000.0)
        radar_mod = 0.8 if weather == "STORM" else 1.0

        # 5. Fire off the SIMD Kernel
        _broadphase_sensor_kernel(
            self.s_lat[:N], self.s_lon[:N], self.s_alt[:N], 
            self.s_radar_km[:N], self.s_esm_km[:N], self.s_ir_km[:N], 
            self.s_perf[:N], self.s_ew_band[:N], self.s_det_mod[:N],
            self.t_lat[:M], self.t_lon[:M], self.t_alt[:M], 
            self.t_rcs[:M], self.t_emitting[:M], self.t_jamming[:M], 
            self.t_ecm[:M], self.t_heading[:M],
            ir_mod, radar_mod, cloud_ceiling_ft,
            self.out_cls[:N, :M], self.out_sen[:N, :M], self.out_dist[:N, :M]
        )

        _CLS_MAP = {0: "NONE", 1: "FAINT", 2: "PROBABLE", 3: "CONFIRMED"}
        _SEN_MAP = {0: "NONE", 1: "ESM", 2: "IR", 3: "RADAR"}
        
        results = {}
        
        # 6. Extract winning detections
        for j in range(M):
            best_cls_int = 0
            best_sen_int = 0
            min_det_dist = 9999.0
            
            for i in range(N):
                if self.out_cls[i, j] > 0:
                    if self.out_dist[i, j] < min_det_dist:
                        min_det_dist = self.out_dist[i, j]
                
                if self.out_cls[i, j] > best_cls_int:
                    best_cls_int = self.out_cls[i, j]
                    best_sen_int = self.out_sen[i, j]
                    
            if best_cls_int > 0:
                results[self.target_uids[j]] = (
                    _CLS_MAP[best_cls_int], 
                    _SEN_MAP[best_sen_int],
                    min_det_dist 
                )
                
        return results