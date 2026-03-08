# sensor.py — Data-Oriented multi-spectrum sensor model using SIMD Compute Pipeline

from __future__ import annotations

import math
import random
import zlib
from typing import Optional, TYPE_CHECKING

from constants import BURNTHROUGH_RANGE_KM
from gpu_compute import SensorComputePipeline

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

# Global compute pipeline to reuse memory buffers across calls
_compute_pipeline = SensorComputePipeline()


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


def update_local_contacts(sensor_units: list["Unit"], target_units: list["Unit"], 
                          local_contacts: dict[str, Contact], game_time: float,
                          weather: str = "CLEAR", time_of_day: str = "DAY") -> None:
    _RANK = {"NONE": 0, "FAINT": 1, "PROBABLE": 2, "CONFIRMED": 3}
    refreshed: set[str] = set()
    
    if not sensor_units: return
    primary_sensor = sensor_units[0]
    
    # OPTIMIZATION / BUGFIX: Using zlib.adler32 instead of built-in hash() 
    # to guarantee determinism across runs and architectures.
    scan_offset = zlib.adler32(primary_sensor.uid.encode('utf-8')) % 4
    is_deep_scan_tick = (int(game_time) % 4) == scan_offset

    # Filter targets to only those we need to check this tick
    targets_to_check = []
    for target in target_units:
        if not target.alive: continue

        is_known = target.uid in local_contacts
        
        if not is_known and not is_deep_scan_tick:
            continue
            
        targets_to_check.append(target)

    # --- DOD / GPU Compute Kernel Execution ---
    if targets_to_check:
        results = _compute_pipeline.run_sweep(sensor_units, targets_to_check, weather, time_of_day)
        
        target_map = {t.uid: t for t in targets_to_check}

        # OPTIMIZATION: Unpacking best_dist directly from the SIMD Kernel, skipping thousands of Python calls
        for target_uid, (best_cls, best_sen, best_dist) in results.items():
            target = target_map[target_uid]
            best_rank = _RANK[best_cls]
            
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

    # Purge stale contacts
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