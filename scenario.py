# scenario.py — data models + DB/scenario load & save

from __future__ import annotations
import json
import math
import os
import random
from collections import deque
from dataclasses import dataclass
from typing import Optional

from numba import njit

from constants import MIN_PK, MAX_PK, MISSILE_TRAIL_LEN, CHAFF_PK_PENALTY, FLARE_PK_PENALTY, BURNTHROUGH_RANGE_KM
from geo import fast_dist_km, bearing, slant_range_km, get_elevation_ft
from sensor import Contact

# ── DATA-ORIENTED MISSILE MATH ────────────────────────────────────────────────
@njit(cache=True)
def _compute_terminal_pk(
    is_ballistic: bool, impact_dist_km: float, cep_km: float,
    launch_dist: float, max_range_km: float, base_pk: float,
    shooter_penalty: float, shooter_perf: float, track_error_km: float,
    alt_delta_ft: float, tgt_is_air: bool, tgt_g_load: float,
    tgt_speed_kmh: float, seeker: int, throttle: int, aspect: float,
    tgt_alt_ft: float, search_on: bool, fc_on: bool, jamming: bool,
    ecm_rating: float, wdef_eccm: float, chaff_pen: float, flare_pen: float,
    is_inevadable: bool
) -> float:
    """
    Compiled SIMD execution for terminal missile intercepts.
    Seeker: 0=Other, 1=IR, 2=ARH/SARH, 3=ARM
    Throttle: 0=LOITER, 1=CRUISE, 2=FLANK
    """
    if is_inevadable:
        return 1.0

    ballistic_mult = 1.0
    if is_ballistic:
        if cep_km > 0.0: 
            ballistic_mult = math.exp(-impact_dist_km / cep_km)
        else:
            if impact_dist_km > 0.15: return 0.0
            
    range_fraction = launch_dist / max(1.0, max_range_km)
    range_penalty = (range_fraction ** 2) * 0.40
    
    # Apply Clausewitzian Friction (Drunkness/Corruption) directly to Base PK
    pk = (base_pk * (1.0 - shooter_penalty) * shooter_perf) - range_penalty
    pk *= ballistic_mult
    
    pk -= (track_error_km * 0.05)
    
    if alt_delta_ft > 10000.0: pk += 0.10
    elif alt_delta_ft < -10000.0: pk -= 0.15
    
    if tgt_is_air:
        if tgt_g_load > 3.0:
            em_penalty = (tgt_g_load - 3.0) * 0.05
            if tgt_speed_kmh > 1000.0: em_penalty *= 1.5
            pk -= em_penalty

    if seeker == 1: # IR
        if throttle == 2: pk += 0.15      # FLANK
        elif throttle == 0: pk -= 0.10    # LOITER

    if seeker == 2 and tgt_is_air: # ARH/SARH
        # Doppler Notch
        if (75.0 < aspect < 105.0) or (255.0 < aspect < 285.0):
            notch_penalty = 0.10
            if tgt_alt_ft < 5000.0: notch_penalty += 0.20
            pk -= notch_penalty

    if seeker == 3: # ARM
        if not search_on and not fc_on: pk -= 0.60
        elif not fc_on: pk -= 0.30

    if jamming and ecm_rating > 0.0 and launch_dist > 25.0: # BURNTHROUGH
        if seeker == 2: # ARH/SARH
            pk -= max(0.0, ecm_rating - wdef_eccm)

    pk -= chaff_pen
    pk -= flare_pen

    # Clamp between MIN_PK and MAX_PK
    if pk < 0.05: pk = 0.05
    if pk > 0.95: pk = 0.95
    return pk


@dataclass(frozen=True)
class WeaponDef:
    key:          str
    display_name: str
    seeker:       str
    range_km:     float
    min_range_km: float
    speed_kmh:    float
    base_pk:      float
    is_gun:       bool
    description:  str
    domain:       str     
    damage:       float   
    reload_time_s: float  
    eccm:         float
    cep_km:       float = 0.0  
    splash_radius_km: float = 0.0  
    is_point_defense: bool = False
    inevadable:   bool = False
    flight_profile: str = "direct" 
    rcs_m2:       float = 0.1   

@dataclass(frozen=True)
class PlatformDef:
    key:               str
    display_name:      str
    unit_type:         str
    speed_kmh:         float
    ceiling_ft:        int
    ecm_rating:        float
    chaff_capacity:    int
    flare_capacity:    int
    fuel_capacity_kg:  float
    fuel_burn_rate_kg_h: float
    radar_range_km:    float
    radar_type:        str
    radar_modes:       tuple[str, ...]
    radar_band:        str              
    esm_range_km:      float            
    ir_range_km:       float            
    default_loadout:   dict[str, int]
    available_weapons: tuple[str, ...]
    fleet_count:       int              
    player_side:       str              
    rcs_m2:            float            
    cruise_alt_ft:     float            
    rearm_time_s:      float
    max_g:             float            
    value_points:      int = 10

class Mission:
    __slots__ = (
        'name', 'mission_type', 'target_lat', 'target_lon', 
        'radius_km', 'altitude_ft', 'rtb_fuel_pct', 
        'time_on_target', 'package_id'
    )
    
    def __init__(self, name: str, mission_type: str, target_lat: float, target_lon: float, 
                 radius_km: float, altitude_ft: float, rtb_fuel_pct: float, 
                 time_on_target: float = 0.0, package_id: str = ""):
        self.name = name
        self.mission_type = mission_type
        self.target_lat = target_lat
        self.target_lon = target_lon
        self.radius_km = radius_km
        self.altitude_ft = altitude_ft
        self.rtb_fuel_pct = rtb_fuel_pct
        self.time_on_target = time_on_target
        self.package_id = package_id

class GameEvent:
    __slots__ = (
        'id', 'condition_type', 'condition_val', 
        'action_type', 'action_val', 'triggered'
    )
    
    def __init__(self, id: str, condition_type: str, condition_val: str, 
                 action_type: str, action_val: str, triggered: bool = False):
        self.id = id
        self.condition_type = condition_type
        self.condition_val = condition_val
        self.action_type = action_type
        self.action_val = action_val
        self.triggered = triggered

class Missile:
    __slots__ = (
        'shooter', 'lat', 'lon', 'altitude_ft', 'target', 'side', 'wdef',
        'active', 'status', 'detonated', 'launch_dist', 'trail',
        'eff_speed_kmh', 'is_ballistic', 'impact_lat', 'impact_lon',
        'impact_alt_ft', 'did_kill', 'guidance_timer', 'pd_check_timer',
        'motor_burnout_fraction', '_terminal_phase_triggered'
    )
    
    def __init__(self, shooter: "Unit", target: "Unit", weapon_def: WeaponDef):
        self.shooter    = shooter
        self.lat        = shooter.lat
        self.lon        = shooter.lon
        self.altitude_ft = shooter.altitude_ft
        self.target     = target
        self.side       = shooter.side
        self.wdef       = weapon_def
        self.active     = True
        self.status     = "IN_FLIGHT"
        self.detonated  = False
        self.launch_dist = slant_range_km(self.lat, self.lon, self.altitude_ft, target.lat, target.lon, target.altitude_ft)
        self.trail: deque[tuple[float, float]] = deque(maxlen=MISSILE_TRAIL_LEN)
        
        base_speed = shooter.current_speed_kmh if hasattr(shooter, 'current_speed_kmh') else 0.0
        self.eff_speed_kmh = base_speed + (weapon_def.speed_kmh if weapon_def.speed_kmh > 0 else 5000.0)
        self.motor_burnout_fraction = 0.35 
        
        self.is_ballistic = (self.wdef.flight_profile == "ballistic")
        self.impact_lat   = target.lat
        self.impact_lon   = target.lon
        self.impact_alt_ft = target.altitude_ft
        self.did_kill     = False 
        
        self.guidance_timer = 0.0 
        self.pd_check_timer = random.uniform(0.0, 0.25) 
        
        # Link state to the DOD kinematics pipeline
        self._terminal_phase_triggered = False
        
        if self.is_ballistic and weapon_def.cep_km > 0.0:
            angle = random.uniform(0, 360)
            dist = random.gauss(0, weapon_def.cep_km)
            self.impact_lat += (math.cos(math.radians(angle)) * dist) / 111.32
            self.impact_lon += (math.sin(math.radians(angle)) * dist) / (111.32 * max(0.0001, math.cos(math.radians(target.lat))))

    def _prepare_pk_args(self) -> dict:
        seeker_map = {"IR": 1, "ARH": 2, "SARH": 2, "ARM": 3}
        seeker_int = seeker_map.get(self.wdef.seeker, 0)
        
        throttle_map = {"LOITER": 0, "CRUISE": 1, "FLANK": 2}
        throttle_int = throttle_map.get(getattr(self.target, 'throttle_state', 'CRUISE'), 1)
        
        tgt_is_air = self.target.platform.unit_type.lower() in ("fighter", "attacker", "helicopter", "awacs")
        
        track = getattr(self.shooter, 'merged_contacts', {}).get(self.target.uid)
        track_err = track.pos_error_km if track else 0.0
        
        aspect = 0.0
        if tgt_is_air and seeker_int == 2:
            brg_to_msl = bearing(self.target.lat, self.target.lon, self.lat, self.lon)
            aspect = abs(self.target.heading - brg_to_msl) % 360.0
            
        shooter_pen = getattr(self.shooter, 'inefficiency_penalty', 0.0)
        if getattr(self.shooter, 'systems', {}).get('weapons') == 'DEGRADED': 
            shooter_pen += 0.25
            
        impact_dist = fast_dist_km(self.impact_lat, self.impact_lon, self.target.lat, self.target.lon) if self.is_ballistic else 0.0

        return {
            "seeker_int": seeker_int, "throttle_int": throttle_int, "tgt_is_air": tgt_is_air,
            "track_err": track_err, "aspect": aspect, "shooter_pen": shooter_pen, "impact_dist": impact_dist
        }

    def _calculate_terminal_pk(self) -> float:
        args = self._prepare_pk_args()
        
        chaff_pen = 0.0
        flare_pen = 0.0
        if args["seeker_int"] == 2 and self.target.chaff > 0:
            if random.random() < 0.50:
                self.target.chaff -= 1
                chaff_pen = 0.25
        elif args["seeker_int"] == 1 and self.target.flare > 0:
            if random.random() < 0.50:
                self.target.flare -= 1
                flare_pen = 0.25
                
        return _compute_terminal_pk(
            self.is_ballistic, args["impact_dist"], self.wdef.cep_km,
            self.launch_dist, self.wdef.range_km, self.wdef.base_pk,
            args["shooter_pen"], getattr(self.shooter, 'performance_mult', 1.0), args["track_err"],
            self.shooter.altitude_ft - self.target.altitude_ft, args["tgt_is_air"],
            getattr(self.target, 'current_g_load', 1.0), getattr(self.target, 'current_speed_kmh', 300.0),
            args["seeker_int"], args["throttle_int"], args["aspect"], self.target.altitude_ft,
            getattr(self.target, 'search_radar_active', True), getattr(self.target, 'fc_radar_active', True),
            self.target.is_jamming, self.target.platform.ecm_rating, self.wdef.eccm,
            chaff_pen, flare_pen, self.wdef.inevadable
        )

    def estimated_pk(self) -> float:
        args = self._prepare_pk_args()
        
        return _compute_terminal_pk(
            self.is_ballistic, args["impact_dist"], self.wdef.cep_km,
            self.launch_dist, self.wdef.range_km, self.wdef.base_pk,
            args["shooter_pen"], getattr(self.shooter, 'performance_mult', 1.0), args["track_err"],
            self.shooter.altitude_ft - self.target.altitude_ft, args["tgt_is_air"],
            getattr(self.target, 'current_g_load', 1.0), getattr(self.target, 'current_speed_kmh', 300.0),
            args["seeker_int"], args["throttle_int"], args["aspect"], self.target.altitude_ft,
            getattr(self.target, 'search_radar_active', True), getattr(self.target, 'fc_radar_active', True),
            self.target.is_jamming, self.target.platform.ecm_rating, self.wdef.eccm,
            0.0, 0.0, self.wdef.inevadable 
        )


class Unit:
    __slots__ = (
        'uid', 'callsign', 'lat', 'lon', 'side', 'platform', 'loadout',
        'current_loadout_role', 'image_path', 'fuel_kg', 'hp', 'damage_state',
        'systems', 'fire_intensity', 'drunkness', 'corruption', 'waypoints',
        'heading', 'selected', 'is_detected', 'flash_frames', 'selected_weapon',
        'tot_check_timer', '_cached_tot_speed', 'auto_engage', 'altitude_ft',
        'roe', 'mission', 'home_uid', 'duty_state', 'duty_timer', 'emcon_state',
        'is_jamming', 'search_radar_active', 'fc_radar_active', 'iff_active',
        'throttle_state', 'wra', 'chaff', 'flare', 'is_evading', 'last_evasion_time',
        'weapon_ready_times', 'target_altitude_ft', 'home_lat', 'home_lon',
        'datalink_active', 'local_contacts', 'merged_contacts', 'current_speed_kmh',
        'target_heading', 'current_g_load', 'leader_uid', 'leader_unit', 'formation_slot',
        'is_intercepting', 'saved_waypoints', 'formation', 'flight_doctrine',
        'is_cranking', 'crank_timer', 'crank_heading',
        'patrol_dir', 'patrol_angle', 'formation_target_speed', '_max_loadout'
    )
    
    def __init__(self, uid: str, callsign: str, lat: float, lon: float, side: str, platform: PlatformDef, loadout: dict[str, int], image_path: Optional[str] = None, drunkness: int = 1, corruption: int = 1):
        self.uid        = uid
        self.callsign   = callsign
        self.lat        = lat
        self.lon        = lon
        self.side       = side
        self.platform   = platform
        self.loadout    = dict(loadout)
        self._max_loadout = dict(loadout)
        self.current_loadout_role: str = "DEFAULT"
        self.image_path = image_path
        self.fuel_kg    = platform.fuel_capacity_kg
        self.hp: float = 1.0
        self.damage_state: str = "OK"
        self.systems = {"search_radar": "OK", "fc_radar": "OK", "mobility": "OK", "weapons": "OK"}
        self.fire_intensity: float = 0.0 
        self.drunkness  = drunkness
        self.corruption = corruption
        self.waypoints: list[tuple[float, float, float]] = []
        self.heading    = 0.0
        self.selected   = False
        self.is_detected = False
        self.flash_frames = 0
        self.selected_weapon: Optional[str] = None
        
        self.tot_check_timer: float = 0.0
        self._cached_tot_speed: float = 0.0
        
        unit_type_lower = platform.unit_type.lower()
        
        if platform.key in ("Flamingo_TEL",):
            self.auto_engage = False
        else:
            self.auto_engage = True if unit_type_lower in ("tank", "ifv", "apc", "recon", "tank_destroyer", "sam", "artillery") else False
        
        if unit_type_lower in ("fighter", "attacker", "helicopter", "awacs"):
            self.altitude_ft = platform.cruise_alt_ft
        else:
            self.altitude_ft = get_elevation_ft(lat, lon) + 15.0
            
        self.roe: str = "FREE" if side == "Red" else "TIGHT" 
        self.mission: Optional[Mission] = None
        self.home_uid: str = ""
        self.duty_state: str = "ACTIVE"  
        self.duty_timer: float = 0.0
        self.emcon_state: str = "ACTIVE" 
        self.is_jamming: bool = False
        self.search_radar_active: bool = True
        self.fc_radar_active: bool = True
        self.iff_active: bool = True 
        self.throttle_state: str = "CRUISE" 
        
        self.wra: dict[str, dict[str, float]] = {
            t: {"range": 0.90, "qty": 1} 
            for t in ["ALL", "fighter", "attacker", "helicopter", "awacs", "tank", "ifv", "apc", "recon", "tank_destroyer", "sam", "artillery", "airbase", "ship", "submarine"]
        }
        self.chaff: int = platform.chaff_capacity
        self.flare: int = platform.flare_capacity
        self.is_evading: bool = False
        self.last_evasion_time: float = 0.0
        
        self.weapon_ready_times: dict[str, float] = {k: 0.0 for k in self.loadout.keys()}
        
        self.target_altitude_ft: float = self.altitude_ft
        self.home_lat: float = lat
        self.home_lon: float = lon
        self.datalink_active: bool = True
        self.local_contacts: dict[str, Contact] = {}
        self.merged_contacts: dict[str, Contact] = {}
        self.current_speed_kmh: float = platform.speed_kmh
        self.target_heading: float = 0.0
        self.current_g_load: float = 1.0
        self.leader_uid: str = ""
        self.leader_unit: Optional["Unit"] = None
        self.formation_slot: int = 0 
        self.is_intercepting: bool = False
        self.saved_waypoints: list[tuple[float, float, float]] = []
        self.formation: str = "WEDGE"
        self.flight_doctrine: str = "STANDARD"
        self.is_cranking: bool = False
        self.crank_timer: float = 0.0
        self.crank_heading: float = 0.0
        
        self.patrol_dir: int = 1
        self.patrol_angle: float = random.uniform(0, 360)
        self.formation_target_speed: float = platform.speed_kmh

    def set_emcon(self, state: str) -> None:
        self.emcon_state = state
        if state == "SILENT":
            self.search_radar_active = False; self.fc_radar_active = False; self.is_jamming = False; self.iff_active = False
        elif state == "SEARCH_ONLY":
            self.search_radar_active = True; self.fc_radar_active = False; self.is_jamming = False; self.iff_active = True
        elif state == "ACTIVE":
            self.search_radar_active = True; self.fc_radar_active = True; self.is_jamming = False; self.iff_active = True
        elif state == "BLINDING":
            self.search_radar_active = True; self.fc_radar_active = True; self.is_jamming = True; self.iff_active = True

    @property
    def alive(self) -> bool: return self.hp > 0.0
    @property
    def performance_mult(self) -> float:
        if not self.alive: return 0.0
        mult = 1.0
        if self.damage_state == "LIGHT": mult = 0.8
        elif self.damage_state == "MODERATE": mult = 0.6
        elif self.damage_state == "HEAVY": mult = 0.4
        if self.systems["mobility"] == "DEGRADED": mult *= 0.6
        elif self.systems["mobility"] == "DESTROYED":
            is_air = self.platform.unit_type.lower() in ("fighter", "attacker", "helicopter", "awacs")
            if is_air: mult *= 0.3
            else: mult = 0.0
        return mult

    @property
    def drunkness_label(self) -> str: return {1: "Sober", 2: "Tipsy", 3: "Intoxicated", 4: "Wasted", 5: "Yeltsin"}.get(self.drunkness, "Sober")
    @property
    def corruption_label(self) -> str: return {1: "Clean", 2: "Grass Eater", 3: "Dirty", 4: "Meat Eater", 5: "Shoigu"}.get(self.corruption, "Clean")
    
    @property
    def inefficiency_penalty(self) -> float: 
        return min(0.40, ((self.drunkness - 1) + (self.corruption - 1)) * 0.05)

    def take_damage(self, amount: float, is_dot: bool = False) -> None:
        if not self.alive: return
        self.hp = max(0.0, self.hp - amount)
        if self.hp > 0 and amount > 0 and not is_dot:
            sys_hit_chance = (amount / 0.25) * 0.50
            if random.random() < sys_hit_chance:
                sys_key = random.choice(list(self.systems.keys()))
                if self.systems[sys_key] == "OK": self.systems[sys_key] = "DEGRADED"
                elif self.systems[sys_key] == "DEGRADED": self.systems[sys_key] = "DESTROYED"
            if random.random() < amount * 1.5:  
                self.fire_intensity = min(1.0, self.fire_intensity + 0.35)
        if self.hp <= 0.0:
            self.damage_state = "KILLED"; self.is_jamming = False; self.search_radar_active = False; self.fc_radar_active = False; self.iff_active = False; self.fire_intensity = 0.0
            self.systems = {"search_radar": "DESTROYED", "fc_radar": "DESTROYED", "mobility": "DESTROYED", "weapons": "DESTROYED"}
        elif self.hp <= 0.25: self.damage_state = "HEAVY"
        elif self.hp <= 0.50: self.damage_state = "MODERATE"
        elif self.hp <= 0.75: self.damage_state = "LIGHT"
        else: self.damage_state = "OK"
        if self.systems["search_radar"] == "DESTROYED": self.search_radar_active = False
        if self.systems["fc_radar"] == "DESTROYED": self.fc_radar_active = False

    def add_waypoint(self, lat: float, lon: float, alt: float = -1.0) -> None:
        if self.platform.unit_type.lower() not in ("fighter", "attacker", "helicopter", "awacs") and self.systems["mobility"] == "DESTROYED": return
        self.is_cranking = False
        self.waypoints.append((lat, lon, alt))
        self._recalc_heading()

    def clear_waypoints(self) -> None:
        self.is_cranking = False; self.waypoints.clear()

    def _recalc_heading(self) -> None:
        if self.waypoints:
            if self.platform.unit_type.lower() in ("fighter", "attacker", "helicopter", "awacs"): self.target_heading = bearing(self.lat, self.lon, self.waypoints[0][0], self.waypoints[0][1])
            else: self.heading = bearing(self.lat, self.lon, self.waypoints[0][0], self.waypoints[0][1])

    def has_ammo(self, weapon_key: str) -> bool: return self.loadout.get(weapon_key, 0) > 0
    def expend_round(self, weapon_key: str) -> bool:
        if self.has_ammo(weapon_key): self.loadout[weapon_key] -= 1; return True
        return False

    def cycle_loadout(self, db: "Database") -> str:
        roles = ["DEFAULT", "A2A", "A2G", "SEAD"]
        if "Storm_Shadow" in self.platform.available_weapons: roles.append("DEEP STRIKE")
        idx = (roles.index(self.current_loadout_role) + 1) % len(roles) if self.current_loadout_role in roles else 0
        new_role = roles[idx]; self.current_loadout_role = new_role
        
        if new_role == "DEFAULT": self.loadout = dict(self.platform.default_loadout); return new_role
            
        new_loadout = {}
        guns = {w: 1 for w in self.platform.available_weapons if db.weapons.get(w) and db.weapons[w].is_gun}
        new_loadout.update(guns)
        aw = [w for w in self.platform.available_weapons if db.weapons.get(w) and not db.weapons[w].is_gun]
        
        if new_role == "A2A":
            for w in [w for w in aw if db.weapons.get(w) and db.weapons[w].domain == "air"]: new_loadout[w] = 4 
        elif new_role == "A2G":
            for w in [w for w in aw if db.weapons.get(w) and db.weapons[w].domain == "ground" and db.weapons[w].seeker != "ARM" and w != "Storm_Shadow"]: new_loadout[w] = 4
            a2a = [w for w in aw if db.weapons.get(w) and db.weapons[w].domain == "air"]
            if a2a: new_loadout[a2a[0]] = 2 
        elif new_role == "SEAD":
            arms = [w for w in aw if db.weapons.get(w) and db.weapons[w].seeker == "ARM"]
            if arms:
                for w in arms: new_loadout[w] = 4
            else:
                for w in [w for w in aw if db.weapons.get(w) and db.weapons[w].domain == "ground" and w != "Storm_Shadow"]: new_loadout[w] = 4
            a2a = [w for w in aw if db.weapons.get(w) and db.weapons[w].domain == "air"]
            if a2a: new_loadout[a2a[0]] = 2
        elif new_role == "DEEP STRIKE":
            if "Storm_Shadow" in aw: new_loadout["Storm_Shadow"] = 2
            a2a = [w for w in aw if db.weapons.get(w) and db.weapons[w].domain == "air"]
            if a2a: new_loadout[a2a[0]] = 2
            
        self.loadout = new_loadout
        self._max_loadout = dict(new_loadout)
        self.weapon_ready_times = {k: 0.0 for k in self.loadout.keys()}
        return new_role

    def set_loadout_role(self, db: "Database", role: str) -> None:
        target = role if role in (["DEFAULT", "A2A", "A2G", "SEAD"] + (["DEEP STRIKE"] if "Storm_Shadow" in self.platform.available_weapons else [])) else "DEFAULT"
        attempts = 0
        while self.current_loadout_role != target and attempts < 6: self.cycle_loadout(db); attempts += 1

    def best_weapon_for(self, db: "Database", target: "Unit") -> Optional[str]:
        if self.systems["weapons"] == "DESTROYED": return None
        target_is_air = target.platform.unit_type.lower() in ("fighter", "attacker", "helicopter", "awacs")
        best_key, best_score = None, -1.0
        is_sead = self.mission and self.mission.mission_type == "SEAD"
        is_strike = self.mission and self.mission.mission_type in ("STRIKE", "DEEP STRIKE")
        
        for wkey, qty in self.loadout.items():
            if qty <= 0: continue
            wdef = db.weapons.get(wkey)
            if not wdef: continue
            if wdef.domain == "air" and not target_is_air: continue
            if wdef.domain == "ground" and target_is_air: continue
            if wdef.seeker == "ARM" and not (getattr(target, 'search_radar_active', True) or getattr(target, 'fc_radar_active', True)): continue 
            
            score = wdef.range_km + (wdef.damage * 100) + (wdef.base_pk * 50)
            if is_sead and wdef.seeker == "ARM": score += 10000 
            elif not target_is_air and target.platform.unit_type.lower() in ("sam", "airbase") and wdef.seeker == "ARM": score += 5000
            if is_strike and not target_is_air and wdef.range_km > 20: score += 5000
            if slant_range_km(self.lat, self.lon, self.altitude_ft, target.lat, target.lon, target.altitude_ft) < 15.0 and wdef.domain == "air" and wdef.seeker == "IR": score += 2000 
            if score > best_score: best_key, best_score = wkey, score
        return best_key

    def trigger_flash(self, frames: int = 12) -> None: self.flash_frames = frames
    def tick_flash(self) -> None:
        if self.flash_frames > 0: self.flash_frames -= 1
    def is_clicked(self, screen_pos: tuple[int, int], sx: float, sy: float, radius: int = 16) -> bool:
        return math.hypot(sx - screen_pos[0], sy - screen_pos[1]) <= radius

class Database:
    def __init__(self, weapons_path: str = None, units_path: str = None):
        self.weapons: dict[str, WeaponDef] = {}
        self.platforms: dict[str, PlatformDef] = {}
        raw_weapons = {}
        if weapons_path and os.path.exists(weapons_path):
            with open(weapons_path, encoding="utf-8") as fh: raw_weapons = json.load(fh)
        for key, d in raw_weapons.items():
            profile = d.get("flight_profile")
            if not profile:
                if d.get("seeker") == "SARH" or (d.get("range_km", 0) > 40 and d.get("domain") == "air"): profile = "lofted"
                elif "AShM" in d.get("description", "") or "Sea" in d.get("display_name", ""): profile = "sea_skimming"
                elif d.get("range_km", 0) > 100 and d.get("domain") == "ground": profile = "terrain_following"
                elif "ARTY" in key or "GMLRS" in key or "ROCKET" in key: profile = "ballistic"
                else: profile = "direct"
            w_rcs = float(d.get("rcs_m2", -1.0))
            if w_rcs < 0:
                if d.get("is_gun", False): w_rcs = 0.01
                elif "ARTY" in key: w_rcs = 0.02
                elif "ROCKET" in key or "GMLRS" in key: w_rcs = 0.04
                elif "Shadow" in key or "Low-observable" in d.get("description", ""): w_rcs = 0.005
                elif float(d.get("damage", 0)) > 1.5: w_rcs = 0.25 
                elif float(d.get("damage", 0)) > 0.8: w_rcs = 0.15 
                else: w_rcs = 0.10 
            self.weapons[key] = WeaponDef(
                key=key, display_name=d["display_name"], seeker=d["seeker"], range_km=d["range_km"], min_range_km=d["min_range_km"], speed_kmh=d["speed_kmh"], base_pk=d["base_pk"],
                is_gun=d["is_gun"], description=d["description"], domain=d.get("domain", "both"), damage=float(d.get("damage", 0.6)), reload_time_s=float(d.get("reload_time_s", 0.5 if d.get("is_gun") else 3.0)),
                eccm=float(d.get("eccm", 0.1)), cep_km=float(d.get("cep_km", 0.0)), splash_radius_km=float(d.get("splash_radius_km", 0.0)),
                is_point_defense=d.get("point_defense", False), inevadable=d.get("inevadable", False), flight_profile=profile, rcs_m2=w_rcs
            )

        raw_platforms = {}
        if units_path and os.path.exists(units_path):
            with open(units_path, encoding="utf-8") as fh: raw_platforms = json.load(fh)
        for key, d in raw_platforms.items():
            r_rng = d["radar"]["range_km"]
            esm_val = float(d.get("esm_range_km", r_rng * 1.5 if r_rng > 0 else (10.0 if d["type"] != "tank" else 0.0)))
            ir_val  = float(d.get("ir_range_km", 40.0 if d["type"] in ("fighter", "attacker", "awacs") else 8.0))
            mg = d.get("max_g")
            if not mg:
                if d["type"] == "fighter": mg = 9.0
                elif d["type"] == "attacker": mg = 5.0
                elif d["type"] == "helicopter": mg = 3.5
                elif d["type"] == "awacs": mg = 2.5
                else: mg = 1.0
                
            rt = float(d.get("rearm_time_s", 120.0))
            if d.get("player_side") == "Red" and d["type"] in ("fighter", "attacker"):
                rt = 1800.0
                
            self.platforms[key] = PlatformDef(
                key=key, display_name=d["display_name"], unit_type=d["type"], speed_kmh=d["speed_kmh"], ceiling_ft=d["ceiling_ft"], ecm_rating=d["ecm_rating"],
                chaff_capacity=int(d.get("chaff_capacity", 30 if d["type"] in ("fighter", "attacker", "helicopter", "awacs") else 0)), flare_capacity=int(d.get("flare_capacity", 30 if d["type"] in ("fighter", "attacker", "helicopter", "awacs") else 0)),
                fuel_capacity_kg=d.get("fuel_capacity_kg", 5000.0), fuel_burn_rate_kg_h=d.get("fuel_burn_rate_kg_h", 1500.0), radar_range_km=r_rng, radar_type=d["radar"]["type"], radar_modes=tuple(d["radar"]["modes"]),
                radar_band=d["radar"].get("band", "fire_control"), esm_range_km=esm_val, ir_range_km=ir_val, default_loadout=d["default_loadout"], available_weapons=tuple(d.get("available_weapons", list(d["default_loadout"].keys()))),
                fleet_count=d.get("fleet_count", 0), player_side=d.get("player_side", "Any"), rcs_m2=float(d.get("rcs_m2", 5.0)), cruise_alt_ft=float(d.get("cruise_alt_ft", 0)), 
                rearm_time_s=rt, max_g=float(mg), value_points=int(d.get("value_points", 10))
            )

def load_scenario(path: str, db: Database) -> tuple[list[Unit], dict, list[GameEvent]]:
    with open(path, encoding="utf-8") as fh: data = json.load(fh)
    units: list[Unit] = []
    for ud in data.get("units", []):
        platform = db.platforms.get(ud["platform"])
        if platform is None: continue
        unit = Unit(
            uid=ud["id"], callsign=ud["callsign"], lat=ud["lat"], lon=ud["lon"], side=ud["side"], platform=platform, loadout=ud.get("loadout", platform.default_loadout),
            image_path=ud.get("image_path"), drunkness=ud.get("drunkness", 1), corruption=ud.get("corruption", 1)
        )
        sys_data = ud.get("systems", {"search_radar": "OK", "fc_radar": "OK", "mobility": "OK", "weapons": "OK"})
        if "radar" in sys_data:
            r_state = sys_data.pop("radar"); sys_data["search_radar"] = r_state; sys_data["fc_radar"] = r_state
        unit.systems = sys_data; unit.fire_intensity = ud.get("fire_intensity", 0.0); unit.roe = ud.get("roe", "FREE" if unit.side == "Red" else "TIGHT")
        unit.throttle_state = ud.get("throttle_state", "CRUISE")
        if "wra" in ud: unit.wra = ud["wra"]
        else: unit.wra["ALL"]["range"] = ud.get("wra_range_pct", 0.90); unit.wra["ALL"]["qty"] = ud.get("wra_qty", 1)
        unit.emcon_state = ud.get("emcon_state", "ACTIVE"); unit.formation = ud.get("formation", "WEDGE"); unit.flight_doctrine = ud.get("flight_doctrine", "STANDARD")
        unit.set_emcon(unit.emcon_state)
        
        unit.home_uid = ud.get("home_uid", ""); unit.duty_state = ud.get("duty_state", "ACTIVE"); unit.duty_timer = ud.get("duty_timer", 0.0)
        
        unit.leader_uid = ud.get("leader_uid", "")
        unit.formation_slot = ud.get("formation_slot", 0)
        
        if unit.duty_state in ("READY", "REARMING") and unit.platform.unit_type.lower() in ("fighter", "attacker", "helicopter", "awacs"):
            unit.altitude_ft = get_elevation_ft(unit.lat, unit.lon) + 15.0
            unit.target_altitude_ft = unit.altitude_ft
            
        if ud.get("mission"):
            mdata = ud["mission"]
            unit.mission = Mission(name=mdata["name"], mission_type=mdata["type"], target_lat=mdata["lat"], target_lon=mdata["lon"], radius_km=mdata["radius"], altitude_ft=mdata["alt"], rtb_fuel_pct=mdata["rtb_fuel"], time_on_target=mdata.get("time_on_target", 0.0), package_id=mdata.get("package_id", ""))
        for wp in ud.get("waypoints", []): unit.add_waypoint(wp[0], wp[1], wp[2] if len(wp)>=3 else -1.0)
        units.append(unit)
        
    uid_map = {u.uid: u for u in units}
    for u in units:
        if u.leader_uid:
            u.leader_unit = uid_map.get(u.leader_uid)
            
    meta = {"name": data.get("name", "Unnamed Scenario"), "description": data.get("description", ""), "start_lat": data.get("start_lat", 50.0), "start_lon": data.get("start_lon", 30.0), "start_zoom": data.get("start_zoom", 7)}
    events = [GameEvent(id=ed["id"], condition_type=ed["condition_type"], condition_val=str(ed["condition_val"]), action_type=ed["action_type"], action_val=str(ed["action_val"]), triggered=ed.get("triggered", False)) for ed in data.get("events", [])]
    return units, meta, events

def save_scenario(path: str, units: list[Unit], meta: dict, events: list[GameEvent], game_time: float = 0.0) -> None:
    units_data = []
    for u in units:
        entry = {
            "id": u.uid, "platform": u.platform.key, "callsign": u.callsign, "side": u.side, "lat": round(u.lat, 6), "lon": round(u.lon, 6), "image_path": u.image_path,
            "drunkness": u.drunkness, "corruption": u.corruption, "systems": u.systems, "fire_intensity": u.fire_intensity, "loadout": u.loadout, "roe": u.roe,
            "emcon_state": u.emcon_state, "throttle_state": u.throttle_state, "wra": u.wra, "formation": u.formation, "flight_doctrine": u.flight_doctrine, "home_uid": u.home_uid,
            "duty_state": u.duty_state, "duty_timer": round(u.duty_timer, 1), "waypoints": [[round(wp[0], 6), round(wp[1], 6), round(wp[2], 1)] for wp in u.waypoints],
            "leader_uid": u.leader_uid, "formation_slot": u.formation_slot
        }
        if u.mission:
            entry["mission"] = {"name": u.mission.name, "type": u.mission.mission_type, "lat": round(u.mission.target_lat, 6), "lon": round(u.mission.target_lon, 6), "radius": round(u.mission.radius_km, 2), "alt": round(u.mission.altitude_ft, 2), "rtb_fuel": round(u.mission.rtb_fuel_pct, 2), "time_on_target": round(u.mission.time_on_target, 1), "package_id": u.mission.package_id}
        units_data.append(entry)
    events_data = [{"id": e.id, "condition_type": e.condition_type, "condition_val": e.condition_val, "action_type": e.action_type, "action_val": e.action_val, "triggered": e.triggered} for e in events]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh: json.dump({**meta, "game_time_seconds": round(game_time, 1), "units": units_data, "events": events_data}, fh, indent=2)

def save_deployment(path: str, units: list[Unit]) -> None:
    units_data = []
    for u in units:
        if u.side != "Blue": continue
        entry = {
            "id": u.uid, "platform": u.platform.key, "callsign": u.callsign, "side": u.side, "lat": round(u.lat, 6), "lon": round(u.lon, 6), "image_path": u.image_path,
            "drunkness": u.drunkness, "corruption": u.corruption, "systems": u.systems, "fire_intensity": u.fire_intensity, "loadout": u.loadout, "roe": u.roe,
            "emcon_state": u.emcon_state, "throttle_state": u.throttle_state, "wra": u.wra, "formation": u.formation, "flight_doctrine": u.flight_doctrine, "home_uid": u.home_uid,
            "duty_state": u.duty_state, "duty_timer": round(u.duty_timer, 1), "waypoints": [[round(wp[0], 6), round(wp[1], 6), round(wp[2], 1)] for wp in u.waypoints],
            "leader_uid": u.leader_uid, "formation_slot": u.formation_slot
        }
        if u.mission:
            entry["mission"] = {"name": u.mission.name, "type": u.mission.mission_type, "lat": round(u.mission.target_lat, 6), "lon": round(u.mission.target_lon, 6), "radius": round(u.mission.radius_km, 2), "alt": round(u.mission.altitude_ft, 2), "rtb_fuel": round(u.mission.rtb_fuel_pct, 2), "time_on_target": round(u.mission.time_on_target, 1), "package_id": u.mission.package_id}
        units_data.append(entry)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh: json.dump({"deployment": units_data}, fh, indent=2)

def load_deployment(path: str, db: Database) -> list[Unit]:
    with open(path, encoding="utf-8") as fh: data = json.load(fh)
    units: list[Unit] = []
    for ud in data.get("deployment", []):
        platform = db.platforms.get(ud["platform"])
        if platform is None: continue
        unit = Unit(uid=ud["id"], callsign=ud["callsign"], lat=ud["lat"], lon=ud["lon"], side=ud["side"], platform=platform, loadout=ud.get("loadout", platform.default_loadout), image_path=ud.get("image_path"), drunkness=ud.get("drunkness", 1), corruption=ud.get("corruption", 1))
        sys_data = ud.get("systems", {"search_radar": "OK", "fc_radar": "OK", "mobility": "OK", "weapons": "OK"})
        if "radar" in sys_data: r_state = sys_data.pop("radar"); sys_data["search_radar"] = r_state; sys_data["fc_radar"] = r_state
        unit.systems = sys_data; unit.fire_intensity = ud.get("fire_intensity", 0.0); unit.roe = ud.get("roe", "TIGHT")
        unit.throttle_state = ud.get("throttle_state", "CRUISE")
        if "wra" in ud: unit.wra = ud["wra"]
        else: unit.wra["ALL"]["range"] = ud.get("wra_range_pct", 0.90); unit.wra["ALL"]["qty"] = ud.get("wra_qty", 1)
        unit.emcon_state = ud.get("emcon_state", "ACTIVE"); unit.formation = ud.get("formation", "WEDGE"); unit.flight_doctrine = ud.get("flight_doctrine", "STANDARD")
        unit.set_emcon(unit.emcon_state)
        
        unit.home_uid = ud.get("home_uid", ""); unit.duty_state = ud.get("duty_state", "ACTIVE"); unit.duty_timer = ud.get("duty_timer", 0.0)
        unit.leader_uid = ud.get("leader_uid", "")
        unit.formation_slot = ud.get("formation_slot", 0)
        
        if unit.duty_state in ("READY", "REARMING") and unit.platform.unit_type.lower() in ("fighter", "attacker", "helicopter", "awacs"):
            unit.altitude_ft = get_elevation_ft(unit.lat, unit.lon) + 15.0
            unit.target_altitude_ft = unit.altitude_ft
            
        if ud.get("mission"):
            mdata = ud["mission"]
            unit.mission = Mission(name=mdata["name"], mission_type=mdata["type"], target_lat=mdata["lat"], target_lon=mdata["lon"], radius_km=mdata["radius"], altitude_ft=mdata["alt"], rtb_fuel_pct=mdata["rtb_fuel"], time_on_target=mdata.get("time_on_target", 0.0), package_id=mdata.get("package_id", ""))
        for wp in ud.get("waypoints", []): unit.add_waypoint(wp[0], wp[1], wp[2] if len(wp)>=3 else -1.0)
        units.append(unit)
        
    uid_map = {u.uid: u for u in units}
    for u in units:
        if u.leader_uid:
            u.leader_unit = uid_map.get(u.leader_uid)
            
    return units