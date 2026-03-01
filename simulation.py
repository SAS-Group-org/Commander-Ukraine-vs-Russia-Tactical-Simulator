# simulation.py — real-time simulation engine

from __future__ import annotations

from collections import deque
from typing import Optional

from geo import haversine, slant_range_km
from scenario import Database, Missile, Unit, WeaponDef
from sensor import Contact, update_contacts

_MAX_LOG = 60

_HOME_ARRIVAL_KM       = 2.0    
_AI_COOLDOWN_MISSILE   = 45.0   
_AI_COOLDOWN_GUN       = 8.0    
_AI_ENGAGE_FRAC        = 0.90   

_GROUND_TYPES = {"tank", "ifv", "apc", "recon", "tank_destroyer", "sam"}

class SimulationEngine:
    def __init__(self, units: list[Unit], db: Database):
        self.units:    list[Unit]    = units
        self.missiles: list[Missile] = []
        self.db:       Database      = db

        self.game_time:        float = 0.0
        self.time_compression: int   = 1
        self.paused:           bool  = False

        self.event_log: deque[str] = deque(maxlen=_MAX_LOG)
        self.blue_contacts: dict[str, Contact] = {}
        self._red_contacts:  dict[str, Contact] = {}
        self.log(f"Scenario loaded — {len(units)} units ready.")

    def log(self, msg: str) -> None:
        self.event_log.append(f"[{self._fmt_time(self.game_time)}] {msg}")

    def set_compression(self, factor: int) -> None:
        self.time_compression = factor
        self.paused = (factor == 0)
        self.log("PAUSED" if self.paused else f"Time compression → {factor}×")

    def update(self, real_delta: float) -> None:
        if self.paused or self.time_compression == 0:
            return

        sim_delta = real_delta * self.time_compression
        self.game_time += sim_delta

        self._move_units(sim_delta)
        self._red_ai(sim_delta)
        self._blue_ai(sim_delta)
        self._move_missiles(sim_delta)
        self._resolve_missile_outcomes()
        self._purge_dead()
        self._update_contacts()
        self._tick_flashes()

    def fire_weapon(self, shooter: Unit, target: Unit,
                    weapon_key: str) -> Optional[Missile]:
        wdef = self.db.weapons.get(weapon_key)
        if wdef is None:
            self.log(f"{shooter.callsign}: unknown weapon '{weapon_key}'.")
            return None
        if not shooter.has_ammo(weapon_key):
            self.log(f"{shooter.callsign}: out of {wdef.display_name}.")
            return None

        target_is_air = target.platform.unit_type in ("fighter", "attacker", "helicopter")
        if wdef.domain == "air" and not target_is_air:
            self.log(f"{shooter.callsign}: {wdef.display_name} cannot target ground units.")
            return None
        if wdef.domain == "ground" and target_is_air:
            self.log(f"{shooter.callsign}: {wdef.display_name} cannot target air units.")
            return None

        dist = slant_range_km(shooter.lat, shooter.lon, shooter.altitude_ft, 
                              target.lat, target.lon, target.altitude_ft)
                              
        if dist < wdef.min_range_km:
            self.log(f"{shooter.callsign}: target inside min range "
                     f"({dist:.1f} km < {wdef.min_range_km} km).")
            return None
        if dist > wdef.range_km:
            self.log(f"{shooter.callsign}: target out of range "
                     f"({dist:.1f} km > {wdef.range_km} km).")
            return None

        shooter.expend_round(weapon_key)
        missile = Missile(shooter.lat, shooter.lon, shooter.altitude_ft, target, shooter.side, wdef)
        self.missiles.append(missile)

        est_pk = missile.estimated_pk()
        fox = {"SARH": "Fox 1", "IR": "Fox 2", "ARH": "Fox 3",
               "CANNON": "Guns", "SACLOS": "ATGM", "LASER": "ATGM"
               }.get(wdef.seeker, "Fox 3")
        self.log(f"{shooter.callsign}: {fox}! {wdef.display_name} → "
                 f"{target.callsign}  dist={dist:.1f} km  "
                 f"est.Pk={int(est_pk * 100)}%")
        return missile

    def _move_units(self, sim_delta: float) -> None:
        for u in self.units:
            was_alive = u.alive
            had_waypoints = bool(u.waypoints)
            
            u.update(sim_delta)
            
            if had_waypoints and not u.waypoints and u.alive:
                self.log(f"{u.callsign} reached destination.")
                
            if was_alive and not u.alive and u.fuel_kg <= 0:
                reason = "crashed" if u.platform.unit_type in ["fighter", "attacker", "helicopter"] else "abandoned"
                self.log(f"MAYDAY: {u.callsign} out of fuel and {reason}!")

    def _move_missiles(self, sim_delta: float) -> None:
        for m in self.missiles:
            m.update(sim_delta)

    def _blue_ai(self, sim_delta: float) -> None:
        red_units = [u for u in self.units if u.side == "Red" and u.alive]
        if not red_units: return

        for blue in self.units:
            if blue.side != "Blue" or not blue.alive or not getattr(blue, 'auto_engage', False): continue
            
            is_ground = blue.platform.unit_type in _GROUND_TYPES
            
            if blue.ai_fire_cooldown > 0: blue.ai_fire_cooldown = max(0.0, blue.ai_fire_cooldown - sim_delta)
            if not hasattr(blue, "ai_gun_cooldown"): blue.ai_gun_cooldown = 0.0  
            if blue.ai_gun_cooldown > 0: blue.ai_gun_cooldown = max(0.0, blue.ai_gun_cooldown - sim_delta) 
            
            self._auto_engage_shooter(blue, red_units, self.blue_contacts, is_ground)

    def _red_ai(self, sim_delta: float) -> None:
        blue_units = [u for u in self.units if u.side == "Blue" and u.alive]
        if not blue_units: return

        for red in self.units:
            if red.side != "Red" or not red.alive: continue

            is_ground = red.platform.unit_type in _GROUND_TYPES

            if red.ai_fire_cooldown > 0: red.ai_fire_cooldown = max(0.0, red.ai_fire_cooldown - sim_delta)
            if not hasattr(red, "ai_gun_cooldown"): red.ai_gun_cooldown = 0.0  
            if red.ai_gun_cooldown > 0: red.ai_gun_cooldown = max(0.0, red.ai_gun_cooldown - sim_delta) 

            if not is_ground:
                if not red.waypoints:
                    if red.ai_state == "patrol":
                        dist_home = haversine(red.lat, red.lon, red.home_lat, red.home_lon)
                        if dist_home > _HOME_ARRIVAL_KM:
                            red.add_waypoint(red.home_lat, red.home_lon)
                            red.ai_state = "returning"
                            self.log(f"{red.callsign}: patrol complete — RTB.")
                        else:
                            red.ai_state = "idle"
                    elif red.ai_state == "returning":
                        dist_home = haversine(red.lat, red.lon, red.home_lat, red.home_lon)
                        if dist_home <= _HOME_ARRIVAL_KM:
                            red.ai_state = "idle"
                            self.log(f"{red.callsign}: at base — holding.")

            self._auto_engage_shooter(red, blue_units, self._red_contacts, is_ground)

    def _auto_engage_shooter(self, shooter: Unit, hostile_targets: list[Unit], contacts: dict[str, Contact], is_ground: bool) -> None:
        gun_ready     = (shooter.ai_gun_cooldown <= 0)       
        missile_ready = (shooter.ai_fire_cooldown <= 0)

        if not gun_ready and not missile_ready: return  

        candidates: list[tuple[str, WeaponDef]] = []
        for wkey, qty in shooter.loadout.items():
            if qty <= 0: continue
            wdef = self.db.weapons.get(wkey)
            if wdef is None: continue
            if wdef.is_gun and not gun_ready: continue
            if not wdef.is_gun and not missile_ready: continue
            if not is_ground and wdef.is_gun: continue
            candidates.append((wkey, wdef))

        if not candidates: return
        candidates.sort(key=lambda x: -x[1].range_km)

        for wkey, wdef in candidates:
            target      = None
            target_dist = float("inf")
            engage_range = wdef.range_km * _AI_ENGAGE_FRAC

            for host in hostile_targets:
                if host.uid not in contacts: continue
                
                target_is_air = host.platform.unit_type in ("fighter", "attacker", "helicopter")
                if wdef.domain == "air" and not target_is_air: continue
                if wdef.domain == "ground" and target_is_air: continue
                
                dist = slant_range_km(shooter.lat, shooter.lon, shooter.altitude_ft, host.lat, host.lon, host.altitude_ft)
                if dist > engage_range or dist < wdef.min_range_km: continue
                if dist < target_dist:
                    target      = host
                    target_dist = dist

            if target is None: continue   

            shooter.expend_round(wkey)
            missile = Missile(shooter.lat, shooter.lon, shooter.altitude_ft, target, shooter.side, wdef)
            self.missiles.append(missile)

            if wdef.is_gun: shooter.ai_gun_cooldown = _AI_COOLDOWN_GUN  
            else: shooter.ai_fire_cooldown = _AI_COOLDOWN_MISSILE

            est_pk  = missile.estimated_pk()
            fire_tag = {"SACLOS": "ATGM", "LASER": "ATGM", "CANNON": "Guns"}.get(wdef.seeker, "Fire")
            prefix = "⚠ " if shooter.side == "Red" else ""
            self.log(f"{prefix}{shooter.callsign} (AUTO): {fire_tag}! {wdef.display_name} → {target.callsign}  {target_dist:.1f} km  Pk={int(est_pk*100)}%")
            return   

    def _resolve_missile_outcomes(self) -> None:
        for m in self.missiles:
            if not m.active and m.status in ("HIT", "MISSED"):
                if m.status == "HIT":
                    if m.target.alive: 
                        m.target.trigger_flash()
                        self.log(f"HIT! {m.target.callsign} damaged by {m.wdef.display_name} (State: {m.target.damage_state}).")
                    else:
                        m.target.trigger_flash()
                        kill_word = "SPLASH" if m.target.platform.unit_type in ("fighter", "attacker", "helicopter") else "SHACK"
                        self.log(f"{kill_word}! {m.target.callsign} destroyed by {m.wdef.display_name}.")
                else:
                    self.log(f"{m.target.callsign} survived {m.wdef.display_name}.")

    def _update_contacts(self) -> None:
        blue_alive = [u for u in self.units if u.side == "Blue" and u.alive]
        red_alive  = [u for u in self.units if u.side == "Red"  and u.alive]

        update_contacts(blue_alive, red_alive, self.blue_contacts, self.game_time)
        update_contacts(red_alive, blue_alive, self._red_contacts, self.game_time)

        red_uid_set  = set(self.blue_contacts.keys())
        blue_uid_set = set(self._red_contacts.keys())
        for u in red_alive: u.is_detected = u.uid in red_uid_set
        for u in blue_alive: u.is_detected = u.uid in blue_uid_set

    def _tick_flashes(self) -> None:
        for u in self.units: u.tick_flash()

    def _purge_dead(self) -> None:
        self.missiles = [m for m in self.missiles if m.active]
        self.units    = [u for u in self.units    if u.alive]

    def blue_units(self) -> list[Unit]: return [u for u in self.units if u.side == "Blue"]
    def red_units(self) -> list[Unit]: return [u for u in self.units if u.side == "Red"]

    def is_game_over(self) -> Optional[str]:
        blues_alive = any(u.alive for u in self.units if u.side == "Blue")
        reds_alive  = any(u.alive for u in self.units if u.side == "Red")
        if not blues_alive and not reds_alive: return "Draw"
        if not blues_alive: return "Red wins"
        if not reds_alive: return "Blue wins"
        return None

    @staticmethod
    def _fmt_time(seconds: float) -> str:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        return f"{h:02d}:{m:02d}:{s:02d}"