# simulation.py — real-time simulation engine, optimized with O(1) mapping, centralized Broad-Phase, and Dynamic Standoff AI

from __future__ import annotations
import math
import random
from collections import deque
from typing import Optional
from dataclasses import dataclass

from geo import haversine, fast_dist_km, slant_range_km, bearing, check_line_of_sight, get_elevation_ft
from scenario import Database, Missile, Unit, WeaponDef, Mission, GameEvent
from sensor import Contact, update_local_contacts
from spatial import SpatialHashGrid

_MAX_LOG = 60
_HOME_ARRIVAL_KM       = 2.0    
_GROUND_TYPES = {"tank", "ifv", "apc", "recon", "tank_destroyer", "sam", "airbase", "artillery"}

@dataclass
class Explosion:
    lat: float
    lon: float
    max_radius_km: float
    life: float = 0.0
    max_life: float = 1.0

class PackageManager:
    def __init__(self, engine: "SimulationEngine"):
        self.engine = engine
        self.update_timer = 5.0

    def update_packages(self, game_time: float, sim_delta: float):
        self.update_timer -= sim_delta
        if self.update_timer > 0: return
        self.update_timer = 5.0
        
        packages = {}
        for u in self.engine.units:
            if u.alive and u.duty_state == "ACTIVE" and u.mission and u.mission.package_id:
                packages.setdefault(u.mission.package_id, []).append(u)
                
        for pid, members in packages.items():
            if len(members) <= 1: continue
            max_tot = max((u.mission.time_on_target for u in members), default=0.0)
            if max_tot == 0.0:
                max_eta = 0.0
                for u in members:
                    pts = [(u.lat, u.lon)] + [(wp[0], wp[1]) for wp in u.waypoints]
                    dist = sum(fast_dist_km(pts[i][0], pts[i][1], pts[i+1][0], pts[i+1][1]) for i in range(len(pts)-1))
                    spd_kms = (max(300.0, u.platform.speed_kmh * u.performance_mult)) / 3600.0
                    eta = dist / spd_kms
                    if eta > max_eta: max_eta = eta
                base_tot = game_time + max_eta + 120.0
                for u in members:
                    u.mission.time_on_target = base_tot - 60.0 if u.mission.mission_type == "SEAD" else base_tot
                self.engine.log(f"Package '{pid}' sequenced. ToT: {self.engine._fmt_time(base_tot)}.")

class SalvoMission:
    def __init__(self, shooter: Unit, target: Unit, weapon_key: str, count: int, doctrine: str):
        self.shooter = shooter; self.target = target; self.weapon_key = weapon_key; self.count = count; self.doctrine = doctrine
        self.active_missiles: list[Missile] = []

class SimulationEngine:
    def __init__(self, units: list[Unit], db: Database, events: list[GameEvent] = None):
        self.units:    list[Unit]    = units
        self.missiles: list[Missile] = []
        self.salvos:   list[SalvoMission] = []
        self.explosions: list[Explosion] = []
        self.db:       Database      = db
        self.events:   list[GameEvent] = events or []
        
        self.game_time:        float = 0.0
        self.time_compression: int   = 1
        self.paused:           bool  = False
        self.event_log: deque[str] = deque(maxlen=_MAX_LOG)
        self.total_log_count:  int   = 0  
        self.game_over_reason: Optional[str] = None
        
        self.weather:     str = "CLEAR"
        self.time_of_day: str = "DAY"
        
        self.score_blue: int = 0; self.score_red: int = 0
        self.aar_log: list[str] = [] 
        
        self.blue_network: dict[str, Contact] = {}; self.red_network:  dict[str, Contact] = {}
        self.blue_contacts = self.blue_network 
        self._units_by_uid: dict[str, Unit] = {u.uid: u for u in self.units}
        self.package_manager = PackageManager(self)
        
        self._logic_timer: float = 0.0 
        
        self.blue_grid = SpatialHashGrid(cell_size_deg=0.5)
        self.red_grid = SpatialHashGrid(cell_size_deg=0.5)
        
        self.log(f"Front Line Attrition Engine Online. {len(units)} units processing.")

    def get_unit_by_uid(self, uid: str) -> Optional[Unit]:
        if len(self.units) != len(self._units_by_uid): self._units_by_uid = {u.uid: u for u in self.units}
        return self._units_by_uid.get(uid)

    def set_compression(self, factor: int) -> None:
        self.time_compression = factor; self.paused = (factor == 0)
        self.log("PAUSED" if self.paused else f"Time compression → {factor}×")

    def log(self, msg: str) -> None:
        self.event_log.append(f"[{self._fmt_time(self.game_time)}] {msg}")
        self.total_log_count += 1

    def generate_aar(self) -> dict:
        return {"duration": self._fmt_time(self.game_time), "score_blue": self.score_blue, "score_red": self.score_red, "winner": "Blue" if self.score_blue > self.score_red else "Red" if self.score_red > self.score_blue else "Draw", "kill_log": self.aar_log}

    def update(self, real_delta: float) -> None:
        if self.paused or self.time_compression == 0: return
        sim_delta = real_delta * self.time_compression
        self.game_time += sim_delta

        # 1. Physics & Movement
        self._move_units(sim_delta)
        self._move_missiles(sim_delta)
        self._process_point_defense(sim_delta)
        self.package_manager.update_packages(self.game_time, sim_delta)
        
        # 2. Logic, AI & Sensors
        self._logic_timer += sim_delta
        if self._logic_timer >= 1.0:
            self._build_spatial_grids()
            self._process_unit_status(1.0)
            self._process_unit_missions(1.0)
            self._process_air_commander(1.0)
            self._update_contacts() 
            self._unit_defensive_ai(1.0)
            self._red_ai(1.0)
            self._blue_ai(1.0)
            self._logic_timer %= 1.0
        
        # 3. Combat Resolution
        self._process_salvos(sim_delta)
        self._resolve_missile_outcomes()
        
        for exp in self.explosions: exp.life += sim_delta
        self.explosions = [e for e in self.explosions if e.life < e.max_life]
        
        self._process_events()
        self._purge_dead()
        self._tick_flashes()

    def _build_spatial_grids(self) -> None:
        self.blue_grid.clear()
        self.red_grid.clear()
        for u in self.units:
            if not u.alive: continue
            if u.side == "Blue": self.blue_grid.insert(u)
            else: self.red_grid.insert(u)

    def _process_point_defense(self, sim_delta: float) -> None:
        for m in self.missiles:
            if not m.active or m.wdef.is_gun: continue 
            
            if not hasattr(m, 'pd_check_timer'):
                m.pd_check_timer = random.uniform(0.0, 0.25)
            m.pd_check_timer -= sim_delta
            if m.pd_check_timer > 0:
                continue
            m.pd_check_timer = 0.25
            
            grid = self.blue_grid if m.side == "Red" else self.red_grid
            defenders = grid.get_candidates(m.lat, m.lon, 45.0)
            
            if not defenders:
                continue

            for defender in defenders:
                if not defender.alive or defender.duty_state != "ACTIVE": continue

                pd_weapon_key = next((wkey for wkey, qty in defender.loadout.items() if qty > 0 and self.db.weapons[wkey].is_point_defense and defender.weapon_ready_times.get(wkey, 0.0) <= self.game_time), None)
                if not pd_weapon_key: continue
                
                pd_wdef = self.db.weapons[pd_weapon_key]
                fast_dist = fast_dist_km(defender.lat, defender.lon, m.lat, m.lon)
                if fast_dist > pd_wdef.range_km + 5.0: continue

                dist_to_missile = slant_range_km(defender.lat, defender.lon, defender.altitude_ft, m.lat, m.lon, m.altitude_ft)
                
                if pd_wdef.min_range_km <= dist_to_missile <= pd_wdef.range_km:
                    rcs_ratio = (max(0.001, m.wdef.rcs_m2) / 0.10) ** 0.33
                    base_chance = pd_wdef.base_pk * (0.60 * rcs_ratio)
                    spd_mult = 0.05 if pd_wdef.range_km > 30.0 else 0.15
                    speed_penalty = (m.eff_speed_kmh / 1000.0) * spd_mult
                    profile_penalty = 0.0
                    if m.wdef.flight_profile in ("sea_skimming", "terrain_following"): profile_penalty += 0.15 + (0.15 if m.wdef.rcs_m2 <= 0.01 else 0.0)
                    if m.is_ballistic and pd_wdef.range_km < 50.0 and not pd_wdef.is_gun: profile_penalty += 0.25

                    intercept_chance = max(0.01, (base_chance * defender.performance_mult) - speed_penalty - profile_penalty)
                    defender.expend_round(pd_weapon_key)
                    defender.weapon_ready_times[pd_weapon_key] = self.game_time + pd_wdef.reload_time_s
                    
                    if random.random() <= intercept_chance:
                        m.active = False; m.status = "INTERCEPTED"
                        self.explosions.append(Explosion(m.lat, m.lon, 0.02))
                        self.log(f"{defender.callsign} INTERCEPTED incoming {m.wdef.display_name} with {pd_wdef.display_name}!")
                        break

    def _process_events(self) -> None:
        for e in self.events:
            if e.triggered: continue
            triggered = False
            if e.condition_type == "TIME": triggered = self.game_time >= float(e.condition_val)
            elif e.condition_type == "UNIT_DEAD":
                u = self.get_unit_by_uid(e.condition_val)
                triggered = u is None or not u.alive
            elif e.condition_type == "AREA_ENTERED":
                try:
                    lat_str, lon_str, rad_str = e.condition_val.split(",")
                    tgt_lat, tgt_lon, tgt_rad = float(lat_str), float(lon_str), float(rad_str)
                    triggered = any(u.alive and u.side == "Blue" and fast_dist_km(u.lat, u.lon, tgt_lat, tgt_lon) <= tgt_rad for u in self.units)
                except Exception: pass

            if triggered:
                e.triggered = True
                if e.action_type == "LOG": self.log(f"EVENT: {e.action_val}")
                elif e.action_type == "SCORE":
                    try:
                        side, pts_str = e.action_val.split(":")
                        if side == "Blue": self.score_blue += int(pts_str)
                        elif side == "Red": self.score_red += int(pts_str)
                        self.log(f"EVENT: {side} scored {pts_str} objective points!")
                    except Exception: self.log(f"EVENT ERROR: Malformed SCORE action_val '{e.action_val}'")
                elif e.action_type == "VICTORY":
                    self.log(f"*** {e.action_val.upper()} WINS BY EVENT OBJECTIVE ***"); self.game_over_reason = f"{e.action_val} wins"

    def _process_air_commander(self, sim_delta: float) -> None:
        cap_groups = {}
        for u in self.units:
            if u.side == "Red" and u.platform.unit_type in ("fighter", "attacker") and u.mission and u.mission.name.endswith("CAP"):
                cap_groups.setdefault(u.mission.name, []).append(u)
                
        for msn_name, group in cap_groups.items():
            effective_cap = 0
            for u in group:
                if u.alive and u.duty_state == "ACTIVE" and u.mission.mission_type == "CAP" and not getattr(u, 'is_evading', False):
                    effective_cap += 1
                    
            if effective_cap == 0:
                ready_leaders = [u for u in group if u.alive and u.duty_state == "READY" and not u.leader_uid]
                if ready_leaders:
                    new_lead = ready_leaders[0]
                    new_lead.duty_state = "ACTIVE"
                    new_lead.altitude_ft = new_lead.platform.cruise_alt_ft
                    new_lead.target_altitude_ft = new_lead.platform.cruise_alt_ft
                    new_lead.clear_waypoints()
                    
                    wingmen = [u for u in group if u.alive and getattr(u, 'leader_uid', '') == new_lead.uid]
                    for w in wingmen:
                        w.duty_state = "ACTIVE"
                        w.altitude_ft = w.platform.cruise_alt_ft
                        w.target_altitude_ft = w.platform.cruise_alt_ft
                        w.clear_waypoints()
                        
                    self.log(f"Red Air Command: Scrambling {new_lead.callsign} flight to relieve {msn_name}!")

    def _red_ai(self, sim_delta: float) -> None:
        for red in self.units:
            if red.side == "Red" and red.alive and red.platform.unit_type in ("fighter", "attacker") and red.duty_state == "ACTIVE":
                threats = [c for c in self.blue_network.values() if c.classification == "CONFIRMED"]
                local_threats = [t for t in threats if fast_dist_km(red.lat, red.lon, t.est_lat, t.est_lon) < 80.0]
                
                if not local_threats and red.mission and red.mission.mission_type != "CAP" and red.mission.mission_type != "RTB":
                    red.mission.mission_type = "CAP"
                    red.clear_waypoints()

                valid_targets = [self.get_unit_by_uid(uid) for uid, c in red.merged_contacts.items() if c.perceived_side == "Blue"]
                if valid_targets := [t for t in valid_targets if t and t.alive]:
                    self._auto_engage_shooter(red, valid_targets, red.merged_contacts)

    def _blue_ai(self, sim_delta: float) -> None:
        for blue in self.units:
            if blue.side == "Blue" and blue.alive and (getattr(blue, 'auto_engage', False) or (blue.platform.unit_type.lower() in ("fighter", "attacker", "helicopter", "awacs") and blue.mission is not None)):
                valid_targets = [self.get_unit_by_uid(uid) for uid, c in blue.merged_contacts.items() if c.perceived_side == "Red" or (blue.roe == "FREE" and c.perceived_side == "UNKNOWN")]
                if valid_targets := [t for t in valid_targets if t and t.alive]: 
                    self._auto_engage_shooter(blue, valid_targets, blue.merged_contacts)

    def _auto_engage_shooter(self, shooter: Unit, targets: list[Unit], contacts: dict[str, Contact]) -> None:
        if shooter.roe == "HOLD": return
        penalty = getattr(shooter, 'inefficiency_penalty', 0.0)
        if random.random() < penalty * 0.5: return
        
        valid_targets = []
        for host in targets:
            contact = contacts.get(host.uid)
            if not contact or contact.classification == "FAINT": continue
            
            fast_dist = fast_dist_km(shooter.lat, shooter.lon, contact.est_lat, contact.est_lon)
            if fast_dist > 300.0: continue 
            
            dist = slant_range_km(shooter.lat, shooter.lon, shooter.altitude_ft, contact.est_lat, contact.est_lon, contact.altitude_ft)
            
            target_engaged_count = sum(1 for m in self.missiles if m.active and m.side == shooter.side and m.target.uid == host.uid and not getattr(m.wdef, 'is_gun', False))
            if target_engaged_count >= 2: continue
            
            score = host.platform.value_points
            score += max(0, 100 - dist) 
                
            valid_targets.append((score, host, contact, dist))

        valid_targets.sort(key=lambda x: x[0], reverse=True)
        
        for score, host, contact, dist in valid_targets:
            wkey = shooter.best_weapon_for(self.db, host)
            if wkey:
                wdef = self.db.weapons[wkey]
                tgt_wra = shooter.wra.get(host.platform.unit_type, shooter.wra.get("ALL", {"range":0.90, "qty":1}))
                if dist < wdef.range_km * tgt_wra["range"]:
                    actual_qty = min(tgt_wra["qty"], shooter.loadout.get(wkey, 0))
                    if actual_qty > 0:
                        self.queue_salvo(shooter, host, wkey, actual_qty, "salvo" if not wdef.is_gun else "SLS")
                    break 

    def _update_contacts(self) -> None:
        blue_active = [u for u in self.units if u.side == "Blue" and u.alive]
        red_active  = [u for u in self.units if u.side == "Red"  and u.alive]

        for b in blue_active:
            max_sensor_rng = max([b.platform.radar_range_km, b.platform.esm_range_km, b.platform.ir_range_km])
            candidates = self.red_grid.get_candidates(b.lat, b.lon, max_sensor_rng)
            update_local_contacts([b], candidates, b.local_contacts, self.game_time, self.weather, self.time_of_day)

        for r in red_active:
            max_sensor_rng = max([r.platform.radar_range_km, r.platform.esm_range_km, r.platform.ir_range_km])
            candidates = self.blue_grid.get_candidates(r.lat, r.lon, max_sensor_rng)
            update_local_contacts([r], candidates, r.local_contacts, self.game_time, self.weather, self.time_of_day)

        self.blue_network.clear()
        self.red_network.clear()
        for u in self.units:
            if not u.alive: continue
            master_net = self.blue_network if u.side == "Blue" else self.red_network
            if getattr(u, 'datalink_active', False):
                for uid, local_c in u.local_contacts.items():
                    net_c = master_net.get(uid)
                    if net_c is None or local_c.pos_error_km < net_c.pos_error_km: master_net[uid] = local_c

        for u in self.units:
            if not u.alive: continue
            master_net = self.blue_network if u.side == "Blue" else self.red_network
            u.merged_contacts = dict(u.local_contacts)
            if getattr(u, 'datalink_active', False):
                for uid, net_c in master_net.items():
                    local_c = u.merged_contacts.get(uid)
                    if local_c is None or net_c.pos_error_km < local_c.pos_error_km: u.merged_contacts[uid] = net_c

    def _unit_defensive_ai(self, sim_delta: float) -> None:
        incoming_map: dict[str, list[Missile]] = {}
        for m in self.missiles:
            if m.active:
                incoming_map.setdefault(m.target.uid, []).append(m)

        for u in self.units:
            if not u.alive: continue
            
            incoming = incoming_map.get(u.uid, [])
            penalty = getattr(u, 'inefficiency_penalty', 0.0)
            
            if u.platform.unit_type.lower() in _GROUND_TYPES:
                if any(m.wdef.seeker == "ARM" for m in incoming) and u.emcon_state in ("ACTIVE", "BLINDING"):
                    if random.random() > penalty:
                        u.set_emcon("SEARCH_ONLY")
            
            elif u.platform.unit_type.lower() in ("fighter", "attacker", "helicopter", "awacs"):
                if incoming:
                    if random.random() < penalty * sim_delta: continue
                    u.is_evading = True; u.last_evasion_time = self.game_time
                    if getattr(u.platform, 'ecm_rating', 0.0) > 0 and not u.is_jamming:
                        u.set_emcon("BLINDING")
                    
                    for threat in incoming:
                        if slant_range_km(u.lat, u.lon, u.altitude_ft, threat.lat, threat.lon, threat.altitude_ft) < 15.0:
                            chance_mult = 1.0 - penalty
                            if threat.wdef.seeker in ("ARH", "SARH") and u.chaff > 0 and random.random() < (0.25 * sim_delta * chance_mult): u.chaff -= 1
                            elif threat.wdef.seeker == "IR" and u.flare > 0 and random.random() < (0.35 * sim_delta * chance_mult): u.flare -= 1

                    closest_threat = min(incoming, key=lambda m: slant_range_km(u.lat, u.lon, u.altitude_ft, m.lat, m.lon, m.altitude_ft))
                    threat_brg = bearing(u.lat, u.lon, closest_threat.lat, closest_threat.lon)
                    opt1, opt2 = (threat_brg + 90) % 360, (threat_brg - 90) % 360
                    diff1, diff2 = abs((opt1 - u.heading + 360) % 360), abs((opt2 - u.heading + 360) % 360)
                    u.target_heading = opt1 if (diff1 if diff1 <= 180 else 360 - diff1) < (diff2 if diff2 <= 180 else 360 - diff2) else opt2
                    if u.altitude_ft > 2000: u.target_altitude_ft = max(1000, u.altitude_ft - 5000)
                else:
                    if getattr(u, 'is_evading', False) and (self.game_time - u.last_evasion_time) > 8.0:
                        u.is_evading = False; u.set_emcon("ACTIVE"); u.target_altitude_ft = u.mission.altitude_ft if u.mission else u.platform.cruise_alt_ft; u._recalc_heading()

    def _process_salvos(self, sim_delta: float) -> None:
        active_salvos = []
        for s in self.salvos:
            if not s.shooter.alive or not s.target.alive or s.count <= 0: continue
            
            if s.shooter.weapon_ready_times.get(s.weapon_key, 0.0) <= self.game_time:
                wdef = self.db.weapons[s.weapon_key]
                dist = slant_range_km(s.shooter.lat, s.shooter.lon, s.shooter.altitude_ft, s.target.lat, s.target.lon, s.target.altitude_ft)
                if dist < wdef.range_km and s.shooter.expend_round(s.weapon_key):
                    m = Missile(s.shooter, s.target, wdef)
                    self.missiles.append(m); s.count -= 1
                    s.shooter.weapon_ready_times[s.weapon_key] = self.game_time + wdef.reload_time_s
            active_salvos.append(s)
        self.salvos = active_salvos

    def _move_units(self, sim_delta: float) -> None:
        for u in self.units: u.update(sim_delta, self.game_time)

    def _process_unit_missions(self, sim_delta: float) -> None:
        for u in self.units:
            if not u.alive or not u.mission or u.duty_state != "ACTIVE": continue
            if u.platform.unit_type.lower() in ("fighter", "attacker", "helicopter", "awacs"):
                if u.mission.mission_type in ("CAP", "STRIKE", "SEAD", "DEEP STRIKE"):
                    fuel_pct = u.fuel_kg / max(1.0, u.platform.fuel_capacity_kg)
                    has_ammo = True
                    
                    if u.mission.mission_type == "CAP" and u.platform.unit_type.lower() in ("fighter", "attacker"):
                        has_ammo = any(qty > 0 and self.db.weapons.get(k) and self.db.weapons[k].domain in ("air", "both") for k, qty in u.loadout.items())
                    elif u.mission.mission_type in ("STRIKE", "SEAD", "DEEP STRIKE"):
                        has_ammo = any(qty > 0 and self.db.weapons.get(k) and self.db.weapons[k].domain in ("ground", "both") for k, qty in u.loadout.items())

                    if fuel_pct <= u.mission.rtb_fuel_pct or not has_ammo:
                        lead_id = getattr(u, 'leader_uid', '')
                        if lead_id:
                            leader = self.get_unit_by_uid(lead_id)
                            if leader and leader.mission and leader.mission.mission_type != "RTB":
                                leader.mission.mission_type = "RTB"
                                leader.clear_waypoints()
                                self.log(f"{leader.callsign} RTBs because wingman {u.callsign} is Bingo/Winchester.")
                                
                                for w in self.units:
                                    if getattr(w, 'leader_uid', '') == lead_id and w.mission and w.mission.mission_type != "RTB":
                                        w.mission.mission_type = "RTB"
                                        w.clear_waypoints()

                        reason = "Winchester (Out of Weapons)" if not has_ammo else "Bingo Fuel"
                        self.log(f"{u.callsign} is {reason}. Breaking off and returning to base.")
                        u.mission.mission_type = "RTB"
                        u.clear_waypoints()
                        
                        for w in self.units:
                            if getattr(w, 'leader_uid', '') == u.uid and w.mission and w.mission.mission_type != "RTB":
                                w.mission.mission_type = "RTB"
                                w.clear_waypoints()
                                self.log(f"{w.callsign} follows leader RTB.")

        for u in self.units:
            if not u.alive or not u.mission or u.duty_state != "ACTIVE" or getattr(u, 'is_evading', False): continue
            
            if u.mission.mission_type == "RTB":
                base = self.get_unit_by_uid(u.home_uid)
                if base:
                    if fast_dist_km(u.lat, u.lon, base.lat, base.lon) > _HOME_ARRIVAL_KM:
                        if not u.waypoints: u.add_waypoint(base.lat, base.lon, -1.0)
                    else: 
                        u.duty_state = "REARMING"
                        u.duty_timer = u.platform.rearm_time_s
                        
                        if u.side == "Red" and u.mission.name.endswith("CAP"):
                            u.mission.mission_type = "CAP"
                            u.clear_waypoints()
                        else:
                            u.mission = None
                            
                        u.altitude_ft = get_elevation_ft(u.lat, u.lon) + 15.0
                        u.target_altitude_ft = u.altitude_ft
            
            elif u.mission.mission_type == "CAP":
                if not getattr(u, 'is_intercepting', False) and not getattr(u, 'leader_uid', ""):
                    if not u.waypoints:
                        threat_contacts = [c for c in u.merged_contacts.values() if c.perceived_side == ("Red" if u.side == "Blue" else "Blue")]
                        
                        center_lat = u.mission.target_lat
                        center_lon = u.mission.target_lon
                        
                        # GAME DESIGN AI: Dynamic Standoff using JIT math
                        if u.side == "Red":
                            pushback_lat = 0.0
                            pushback_lon = 0.0
                            push_count = 0
                            
                            for c in threat_contacts:
                                is_sam = c.unit_type == "sam"
                                is_air = c.unit_type in ("fighter", "attacker")
                                if not is_sam and not is_air: continue
                                
                                dist = fast_dist_km(center_lat, center_lon, c.est_lat, c.est_lon)
                                threshold = 130.0 if is_sam else 90.0
                                
                                if dist < threshold:
                                    push_dist = threshold - dist
                                    esc_brg = (bearing(c.est_lat, c.est_lon, center_lat, center_lon)) % 360
                                    lat_rad_clamped = max(0.0001, math.cos(math.radians(center_lat)))
                                    pushback_lat += (math.cos(math.radians(esc_brg)) * push_dist) / 111.32
                                    pushback_lon += (math.sin(math.radians(esc_brg)) * push_dist) / (111.32 * lat_rad_clamped)
                                    push_count += 1
                                    
                            if push_count > 0:
                                center_lat += pushback_lat / push_count
                                center_lon += pushback_lon / push_count

                        # Dynamic intercept bearing for Blue, Static patrol axis for Red
                        if threat_contacts and u.side == "Blue": 
                            best_c = min(threat_contacts, key=lambda c: fast_dist_km(center_lat, center_lon, c.est_lat, c.est_lon))
                            threat_brg = bearing(center_lat, center_lon, best_c.est_lat, best_c.est_lon)
                        else:
                            axis = getattr(u.mission, 'time_on_target', 0.0)
                            if u.side == "Red" and axis > 0:
                                if not hasattr(u, 'patrol_dir'): u.patrol_dir = 1
                                else: u.patrol_dir *= -1
                                threat_brg = (axis if u.patrol_dir == 1 else axis + 180) % 360
                            else:
                                u.patrol_angle = (getattr(u, 'patrol_angle', random.uniform(0, 360)) + 45) % 360
                                threat_brg = u.patrol_angle
                        
                        # GAME DESIGN TWEAK: 150km racetrack loops for Red, 60km loops for Blue
                        leg_len = 75.0 if u.side == "Red" else 30.0
                        
                        lat_rad_clamped = max(0.0001, math.cos(math.radians(center_lat)))
                        lat1 = center_lat + (math.cos(math.radians(threat_brg)) * leg_len) / 111.32
                        lon1 = center_lon + (math.sin(math.radians(threat_brg)) * leg_len) / (111.32 * lat_rad_clamped)
                        lat2 = center_lat + (math.cos(math.radians((threat_brg + 180) % 360)) * leg_len) / 111.32
                        lon2 = center_lon + (math.sin(math.radians((threat_brg + 180) % 360)) * leg_len) / (111.32 * lat_rad_clamped)
                        u.waypoints = [(lat1, lon1, -1.0), (lat2, lon2, -1.0)]; u._recalc_heading()

    def _process_unit_status(self, sim_delta: float) -> None:
        for u in self.units:
            if not u.alive: continue
            is_ground = u.platform.unit_type.lower() in _GROUND_TYPES
            is_air = u.platform.unit_type.lower() in ("fighter", "attacker", "helicopter", "awacs")
            if is_ground and u.systems["mobility"] == "DESTROYED" and u.waypoints: u.clear_waypoints()
            if is_air and u.duty_state == "ACTIVE":
                if u.damage_state in ("HEAVY", "MODERATE") or u.systems["mobility"] != "OK" or u.systems["weapons"] == "DESTROYED":
                    if not u.mission or u.mission.mission_type != "RTB":
                        base = self.get_unit_by_uid(u.home_uid)
                        if base:
                            u.mission = Mission("Emergency RTB", "RTB", base.lat, base.lon, 0, u.altitude_ft, 0)
                            u.clear_waypoints()

    def _move_missiles(self, sim_delta: float) -> None:
        for m in self.missiles: m.update(sim_delta)

    def _resolve_missile_outcomes(self) -> None:
        for m in self.missiles:
            if not m.active and getattr(m, 'detonated', False):
                radius = m.wdef.splash_radius_km if m.wdef.splash_radius_km > 0 else 0.05 
                self.explosions.append(Explosion(m.lat, m.lon, radius))
                if m.status == "HIT":
                    if m.target.alive: m.target.trigger_flash()
                    if getattr(m, 'did_kill', False): 
                        if m.shooter.side == "Blue": self.score_blue += m.target.platform.value_points
                        else: self.score_red += m.target.platform.value_points
                        self.aar_log.append(f"[{self._fmt_time(self.game_time)}] {m.shooter.callsign} destroyed {m.target.callsign}.")
                m.detonated = False

    def _purge_dead(self) -> None:
        self.missiles = [m for m in self.missiles if m.active]
        self.units = [u for u in self.units if u.alive]

    def _tick_flashes(self) -> None:
        for u in self.units: u.tick_flash()

    def queue_salvo(self, shooter: Unit, target: Unit, weapon_key: str, count: int, doctrine: str) -> None:
        self.salvos.append(SalvoMission(shooter, target, weapon_key, count, doctrine))

    def blue_units(self) -> list[Unit]: return [u for u in self.units if u.side == "Blue"]
    def red_units(self) -> list[Unit]: return [u for u in self.units if u.side == "Red"]

    def is_game_over(self) -> Optional[str]:
        if self.game_over_reason: return self.game_over_reason
        if not any(u.side == "Blue" for u in self.units): return "Red wins"
        if not any(u.side == "Red" for u in self.units): return "Blue wins"
        return None

    @staticmethod
    def _fmt_time(seconds: float) -> str:
        h, m = divmod(int(seconds), 3600); m, s = divmod(m, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"