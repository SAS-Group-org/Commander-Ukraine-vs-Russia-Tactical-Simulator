# sensor.py — physics-based multi-spectrum sensor model, optimized via Data-Oriented Design

from __future__ import annotations

import math
import random
from typing import Optional, TYPE_CHECKING
from numba import njit

from constants import BURNTHROUGH_RANGE_KM
from geo import haversine, fast_dist_km, bearing, check_line_of_sight

if TYPE_CHECKING:
    from scenario import PlatformDef, Unit

# ── Physical constants ────────────────────────────────────────────────────────
RCS_REFERENCE_M2: float = 5.0
GROUND_SENSOR_HEIGHT_M: float = 5.0
ECM_SCALE: float = 0.60

FAINT_BAND:    float = 1.00
PROBABLE_BAND: float = 0.75
CONFIRM_BAND:  float = 0.50
CONTACT_TIMEOUT_S: float = 30.0

class Contact:
    """
    OPTIMIZATION: __slots__ tells Python not to use a dynamic dictionary for this object,
    drastically reducing the RAM footprint when tracking thousands of radar contacts.
    """
    __slots__ = (
        'uid', 'est_lat', 'est_lon', 'altitude_ft', 'classification',
        'unit_type', 'perceived_side', 'last_update', 'sensor_type',
        'pos_error_km', 'error_angle', 'base_pos_error_km', 'is_ghost'
    )

    def __init__(self, uid: str, est_lat: float, est_lon: float, altitude_ft: float,
                 classification: str, unit_type: Optional[str], perceived_side: str,
                 last_update: float, sensor_type: str = "NONE", pos_error_km: float = 0.0,
                 error_angle: float = 0.0, base_pos_error_km: float = 0.0, is_ghost: bool = False):
        self.uid = uid
        self.est_lat = est_lat
        self.est_lon = est_lon
        self.altitude_ft = altitude_ft
        self.classification = classification
        self.unit_type = unit_type
        self.perceived_side = perceived_side
        self.last_update = last_update
        self.sensor_type = sensor_type
        self.pos_error_km = pos_error_km
        self.error_angle = error_angle
        self.base_pos_error_km = base_pos_error_km
        self.is_ghost = is_ghost


@njit(cache=True)
def _fast_classify_math(dist_km: float, 
                        s_lat: float, s_lon: float, s_alt: float, 
                        s_radar_km: float, s_esm_km: float, s_ir_km: float, 
                        s_perf: float, s_ew_band: bool, s_det_mod: float,
                        t_lat: float, t_lon: float, t_alt: float, 
                        t_rcs: float, t_emitting: bool, t_jamming: bool, 
                        t_ecm: float, t_heading: float,
                        weather_ir_mod: float, weather_radar_mod: float, 
                        cloud_ceiling_ft: float) -> tuple[int, int]:
    """
    DATA-ORIENTED DESIGN: This function has been stripped of all Python objects and strings. 
    It is compiled directly to machine code, acting like a GPU Compute Shader to process 
    the complex Radar Equation, ECM burn-through, and Line-of-Sight raycasts instantly.
    
    Returns: (classification_int, sensor_int)
    Classes: 0=NONE, 1=FAINT, 2=PROBABLE, 3=CONFIRMED
    Sensors: 0=NONE, 1=ESM,   2=IR,       3=RADAR
    """
    best_cls = 0
    best_sen = 0
    
    # -1 means unchecked. We defer heavy Line-of-Sight raycasting until absolutely necessary.
    has_los = -1 

    # 1. ESM (Passive radar sniffing - Unaffected by weather)
    if t_emitting and s_esm_km > 0:
        esm_range = s_esm_km * s_det_mod
        if dist_km <= esm_range:
            has_los = 1 if check_line_of_sight(s_lat, s_lon, s_alt, t_lat, t_lon, t_alt) else 0
            if has_los == 1:
                best_cls = 2 
                best_sen = 1 

    # 2. IR / FLIR / Optical (Heavily impacted by Weather & Time of Day)
    ir_range = s_ir_km * s_det_mod * weather_ir_mod
    if dist_km <= ir_range and weather_ir_mod > 0.0:
        # Cloud Deck Line-of-Sight Check
        if (s_alt > cloud_ceiling_ft and t_alt < cloud_ceiling_ft) or \
           (s_alt < cloud_ceiling_ft and t_alt > cloud_ceiling_ft):
            pass # Clouds block thermal optics
        else:
            if 3 > best_cls: # 3 = CONFIRMED
                if has_los == -1:
                    has_los = 1 if check_line_of_sight(s_lat, s_lon, s_alt, t_lat, t_lon, t_alt) else 0
                if has_los == 1:
                    best_cls = 3
                    best_sen = 2

    # 3. Active Radar (Band & Aspect-dependent logic)
    if s_radar_km > 0:
        target_rcs = t_rcs
        
        # ASPECT-DEPENDENT RCS MODIFIER
        brg_to_tgt = bearing(s_lat, s_lon, t_lat, t_lon)
        aspect = abs(t_heading - brg_to_tgt) % 360
        if (60 <= aspect <= 120) or (240 <= aspect <= 300):
            target_rcs *= 1.5  # Broadside bloom
        elif (aspect <= 30) or (150 <= aspect <= 210) or (aspect >= 330):
            target_rcs *= 0.7  # Head-on or Tail-on reduction
            
        if s_ew_band:
            target_rcs = max(target_rcs, 1.0) 

        rcs_ratio  = max(target_rcs, 0.01) / 5.0 # RCS_REFERENCE_M2
        R_rcs      = (s_radar_km * s_perf) * (rcs_ratio ** 0.25)
        
        ecm_penalty = 0.0
        if t_jamming:
            if dist_km > 25.0: # BURNTHROUGH_RANGE_KM
                # Smooth burn-through curve based on 1/R^2 vs 1/R^4 dynamics
                jam_factor = 1.0 - (25.0 / dist_km)**2
                ecm_penalty = t_ecm * 0.60 * jam_factor
            
        R_effective = R_rcs * max(0.0, 1.0 - ecm_penalty) * s_det_mod * weather_radar_mod
        
        if R_effective > 0.0 and dist_km <= R_effective * 1.00: # FAINT_BAND
            if has_los == -1:
                has_los = 1 if check_line_of_sight(s_lat, s_lon, s_alt, t_lat, t_lon, t_alt) else 0
            if has_los == 1:
                fraction = dist_km / R_effective       
                
                if fraction <= 0.50: cls = 3       # CONFIRM_BAND
                elif fraction <= 0.75: cls = 2     # PROBABLE_BAND
                else: cls = 1                      # FAINT_BAND
                
                # System-of-Systems: EW/Volume radars detect further but can't CONFIRM
                if s_ew_band and cls == 3:
                    cls = 2
                
                if cls > best_cls:
                    best_cls = cls
                    best_sen = 3 # RADAR

    return best_cls, best_sen


def classify_detection(sensor_unit: "Unit", target: "Unit", dist_km: float, weather: str = "CLEAR", time_of_day: str = "DAY") -> tuple[str, str]:
    # 1. Prepare Weather/Environment Float Modifiers
    ir_mod = 1.0
    if time_of_day == "NIGHT": ir_mod *= 0.8
    if weather == "RAIN": ir_mod *= 0.4
    elif weather == "STORM": ir_mod *= 0.1
    elif weather == "OVERCAST": ir_mod *= 0.7
    
    cloud_ceiling_ft = {"CLEAR": 50000.0, "OVERCAST": 10000.0, "RAIN": 5000.0, "STORM": 3000.0}.get(weather, 50000.0)
    radar_mod = 0.8 if weather == "STORM" else 1.0

    # 2. Extract Unit Data
    penalty = getattr(sensor_unit, 'inefficiency_penalty', 0.0)
    det_mod = 1.0 - penalty 
    s_ew_band = getattr(sensor_unit.platform, 'radar_band', 'fire_control') in ("early_warning", "volume_search")
    
    t_emitting = getattr(target, 'search_radar_active', False) or getattr(target, 'fc_radar_active', False)
    s_radar_active = getattr(sensor_unit, 'search_radar_active', True)
    s_radar_km = sensor_unit.platform.radar_range_km if s_radar_active else 0.0

    # 3. Execute Compiled High-Speed Math
    cls_int, sen_int = _fast_classify_math(
        dist_km, 
        sensor_unit.lat, sensor_unit.lon, sensor_unit.altitude_ft,
        s_radar_km, sensor_unit.platform.esm_range_km, sensor_unit.platform.ir_range_km, 
        sensor_unit.performance_mult, s_ew_band, det_mod,
        target.lat, target.lon, target.altitude_ft, 
        target.platform.rcs_m2, t_emitting, target.is_jamming, 
        target.platform.ecm_rating, target.heading,
        ir_mod, radar_mod, cloud_ceiling_ft
    )

    # 4. Map back to Strings for the Engine
    _CLS_MAP = {0: "NONE", 1: "FAINT", 2: "PROBABLE", 3: "CONFIRMED"}
    _SEN_MAP = {0: "NONE", 1: "ESM", 2: "IR", 3: "RADAR"}
    
    return _CLS_MAP[cls_int], _SEN_MAP[sen_int]


def update_local_contacts(sensor_units: list["Unit"], target_units: list["Unit"], 
                          local_contacts: dict[str, Contact], game_time: float,
                          weather: str = "CLEAR", time_of_day: str = "DAY") -> None:
    _RANK = {"NONE": 0, "FAINT": 1, "PROBABLE": 2, "CONFIRMED": 3}
    refreshed: set[str] = set()
    
    if not sensor_units: return
    primary_sensor = sensor_units[0]
    scan_offset = hash(primary_sensor.uid) % 4
    is_deep_scan_tick = (int(game_time) % 4) == scan_offset

    for target in target_units:
        if not target.alive: continue

        is_known = target.uid in local_contacts
        
        if not is_known and not is_deep_scan_tick:
            continue

        best_cls  = "NONE"
        best_sen  = "NONE"
        best_rank = 0
        best_dist = 9999.0

        for sensor in sensor_units:
            if not sensor.alive: continue
            dist = fast_dist_km(sensor.lat, sensor.lon, target.lat, target.lon)
            cls, sen = classify_detection(sensor, target, dist, weather, time_of_day)
            
            if _RANK[cls] > best_rank:
                best_rank = _RANK[cls]
                best_cls  = cls
                best_sen  = sen
                best_dist = dist

        if best_cls == "NONE": 
            continue  
        
        refreshed.add(target.uid)

        actual_side = target.side
        observer_side = sensor_units[0].side
        opp_side = "Red" if actual_side == "Blue" else "Blue"
        
        existing = local_contacts.get(target.uid)
        
        make_new_roll = False
        if not existing or existing.perceived_side == "UNKNOWN":
            make_new_roll = True
        elif best_sen == "IR" and existing.perceived_side != actual_side:
            make_new_roll = True 
        elif best_rank >= 2 and existing.perceived_side != actual_side and random.random() < 0.05:
            make_new_roll = True 
            
        if make_new_roll:
            misid_chance = 0.0
            if best_sen != "IR":
                if best_rank == 3: misid_chance = 0.02
                elif best_rank == 2: misid_chance = 0.15
                elif best_rank == 1: misid_chance = 0.35
                
                misid_chance += min(0.20, (best_dist / 150.0) * 0.15)
                if target.is_jamming: misid_chance += 0.25
                if actual_side == observer_side and getattr(target, 'iff_active', False):
                    misid_chance *= 0.10
                    
            if random.random() < misid_chance:
                if random.random() < 0.40:
                    p_side = opp_side  
                else:
                    p_side = "UNKNOWN" 
            else:
                p_side = actual_side
        else:
            p_side = existing.perceived_side if existing else "UNKNOWN"

        error_km = 0.0
        if best_sen == "ESM": error_km = best_dist * 0.04  
        elif best_sen == "RADAR": error_km = best_dist * 0.004 
        elif best_sen == "IR": error_km = best_dist * 0.0005    
        
        # ECM GHOSTING MECHANIC: If the target is jamming heavily, spawn false contacts
        if target.is_jamming and best_sen == "RADAR" and best_dist > BURNTHROUGH_RANGE_KM:
            error_km *= 5.0 # Blur the real contact
            
            # Chance to spawn a transient "Ghost" contact nearby
            if random.random() < 0.20 * target.platform.ecm_rating:
                ghost_uid = f"ghost_{target.uid}_{int(game_time)}"
                ghost_offset_dist = random.uniform(5.0, 25.0)
                ghost_angle = random.uniform(0, 360)
                g_lat = target.lat + (math.cos(math.radians(ghost_angle)) * ghost_offset_dist) / 111.32
                g_lon = target.lon + (math.sin(math.radians(ghost_angle)) * ghost_offset_dist) / (111.32 * max(0.0001, math.cos(math.radians(target.lat))))
                
                local_contacts[ghost_uid] = Contact(
                    uid=ghost_uid, est_lat=g_lat, est_lon=g_lon, altitude_ft=target.altitude_ft,
                    classification="FAINT", unit_type=None, perceived_side="UNKNOWN", last_update=game_time, 
                    sensor_type="RADAR", pos_error_km=2.0, error_angle=0.0, base_pos_error_km=2.0, is_ghost=True
                )
                refreshed.add(ghost_uid)
        
        unit_type = target.platform.unit_type if best_rank >= 2 else None

        contact = local_contacts.get(target.uid)
        if contact is None:
            angle = random.uniform(0, 360)
            dlat = (math.cos(math.radians(angle)) * error_km) / 111.32
            dlon = (math.sin(math.radians(angle)) * error_km) / (111.32 * max(0.0001, math.cos(math.radians(target.lat))))
            
            local_contacts[target.uid] = Contact(
                uid=target.uid, est_lat=target.lat + dlat, est_lon=target.lon + dlon, altitude_ft=target.altitude_ft,
                classification=best_cls, unit_type=unit_type, perceived_side=p_side, last_update=game_time, 
                sensor_type=best_sen, pos_error_km=error_km, error_angle=angle, base_pos_error_km=error_km
            )
        else:
            contact.error_angle = (contact.error_angle + random.uniform(-0.5, 0.5)) % 360
            dlat = (math.cos(math.radians(contact.error_angle)) * error_km) / 111.32
            dlon = (math.sin(math.radians(contact.error_angle)) * error_km) / (111.32 * max(0.0001, math.cos(math.radians(target.lat))))
            
            ideal_lat = target.lat + dlat
            ideal_lon = target.lon + dlon
            
            contact.est_lat += (ideal_lat - contact.est_lat) * 0.002
            contact.est_lon += (ideal_lon - contact.est_lon) * 0.002
            
            contact.altitude_ft = target.altitude_ft
            contact.last_update = game_time
            contact.sensor_type = best_sen
            contact.base_pos_error_km = error_km
            contact.pos_error_km = error_km
            
            if best_rank >= _RANK[contact.classification]:
                contact.classification = best_cls
                contact.unit_type = unit_type
            if p_side != "UNKNOWN":
                contact.perceived_side = p_side

    for uid, c in list(local_contacts.items()):
        if uid not in refreshed:
            staleness = game_time - c.last_update
            
            # Ghosts fade very quickly
            if c.is_ghost and staleness > 5.0:
                del local_contacts[uid]
                continue
                
            c.pos_error_km = c.base_pos_error_km + (staleness * 0.15) 
            c.classification = "FAINT" 
            
            if staleness > CONTACT_TIMEOUT_S:
                del local_contacts[uid]