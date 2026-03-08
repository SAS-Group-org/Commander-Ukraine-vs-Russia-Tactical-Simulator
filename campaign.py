# campaign.py — level design, geography, and procedural force generation

from __future__ import annotations
import math
import random
from typing import Optional, TYPE_CHECKING

from geo import haversine, bearing, get_elevation_ft
from scenario import Database, Unit, Mission

if TYPE_CHECKING:
    from simulation import SimulationEngine

# ── Global ID Generator ───────────────────────────────────────────────────────
_UID_COUNTER = 0

def get_next_uid(prefix: str = "u") -> str:
    global _UID_COUNTER
    _UID_COUNTER += 1
    return f"{prefix}_{_UID_COUNTER:04d}"

# ── Level Design: Strategic Geography ─────────────────────────────────────────

def is_water(lat: float, lon: float) -> bool:
    if 45.3 < lat < 47.05 and 34.8 < lon < 38.5: return True
    if lat < 46.2 and lon < 32.8: return True
    if lat < 45.0 and lon < 33.4: return True
    if lat < 44.38: return True
    if lat < 44.8 and lon > 34.5: return True
    if lat < 45.0 and lon > 35.3: return True
    if 45.8 < lat < 46.2 and 34.0 < lon < 35.0: return True
    return False

FRONT_LINE_POINTS = [
    (46.53, 32.30), (46.80, 33.30), (47.10, 34.30), (47.45, 35.40),
    (47.60, 36.10), (47.75, 36.80), (47.90, 37.50), (47.95, 37.65),
    (48.00, 37.80), (48.20, 37.85), (48.40, 37.90), (48.60, 38.00),
    (48.80, 38.10), (49.00, 38.20), (49.30, 38.10), (49.60, 38.00),
    (49.80, 37.90), (50.00, 37.80), (50.20, 37.70), (50.35, 36.30), 
    (50.50, 35.50), (51.00, 35.10), (51.30, 34.30), (51.80, 34.00), 
    (52.20, 33.60), (52.11, 31.78)
]

DENSE_LOC = []
for i in range(len(FRONT_LINE_POINTS) - 1):
    p1 = FRONT_LINE_POINTS[i]
    p2 = FRONT_LINE_POINTS[i+1]
    dist = haversine(p1[0], p1[1], p2[0], p2[1])
    steps = max(1, int(dist / 1.0)) 
    for step in range(steps):
        f = step / steps
        DENSE_LOC.append((p1[0] + f*(p2[0]-p1[0]), p1[1] + f*(p2[1]-p1[1])))
DENSE_LOC.append(FRONT_LINE_POINTS[-1])

def dist_to_loc(lat: float, lon: float) -> float:
    min_d = 999999.0
    for dp in DENSE_LOC:
        d = haversine(lat, lon, dp[0], dp[1])
        if d < min_d: min_d = d
    return min_d

def get_front_line_coords(side: str, min_dist: float, max_dist: float, segment_range: tuple[int, int] = None) -> tuple[float, float]:
    min_idx = segment_range[0] if segment_range else 0
    max_idx = segment_range[1] if segment_range else len(FRONT_LINE_POINTS) - 2
    
    for _ in range(100): 
        idx = random.randint(min_idx, max_idx)
        p1 = FRONT_LINE_POINTS[idx]
        p2 = FRONT_LINE_POINTS[idx+1]
        
        f = random.random()
        clat = p1[0] + f * (p2[0] - p1[0])
        clon = p1[1] + f * (p2[1] - p1[1])
        
        brg = bearing(p1[0], p1[1], p2[0], p2[1])
        normal_brg = (brg - 90) % 360 if side == "Blue" else (brg + 90) % 360
        
        dist_km = random.uniform(min_dist, max_dist)
        lat_rad_clamped = max(0.0001, math.cos(math.radians(clat)))
        tlat = clat + (math.cos(math.radians(normal_brg)) * dist_km) / 111.32
        tlon = clon + (math.sin(math.radians(normal_brg)) * dist_km) / (111.32 * lat_rad_clamped)
        
        actual_dist = dist_to_loc(tlat, tlon)
        if min_dist - 2.0 <= actual_dist <= max_dist + 15.0:
            return tlat, tlon
            
    p1 = FRONT_LINE_POINTS[random.randint(min_idx, max_idx)]
    offset = min_dist / 111.32
    return (p1[0], p1[1] - offset) if side == "Blue" else (p1[0], p1[1] + offset)

# ── Unified Call-Sign Generator ───────────────────────────────────────────────
def get_callsign_for(platform_key: str, index: int) -> str:
    prefixes = {
        "MiG-29UA": "GHOST", "Su-27UA": "PHANTOM", "F-16AM": "VIPER", "F-16A": "FALCON",
        "F-16C": "FALCON", "F-16UA": "FALCON", "Su-25UA": "WARTHOG", "Su-24M": "SWORD", 
        "Mirage2000-5F": "ANGEL", "Mi-8UA": "BEAR", "Mi-24V": "HIND", "Mi-2UA": "SWIFT", 
        "AirbaseB": "ALPHA BASE", "E-3G_Sentry": "SENTRY", "Leopard2": "LEO", 
        "Bradley": "BRAD", "Gepard": "FLAK", "M777": "ARCHER", "Patriot": "CASTLE", 
        "Stryker": "GHOST", "Flamingo_TEL": "FLAMINGO",
        "S-400": "TRIUMF", "Buk-M2": "BUK", "Tor-M1": "TOR", "AirbaseR": "RED BASE"
    }
    return f"{prefixes.get(platform_key, 'UNIT')} {index}"

# ── Live Unit Builder (For Real-Time Spawning) ────────────────────────────────
def make_live_unit(platform_key: str, lat: float, lon: float, side: str, db: Database, callsign: str) -> Unit | None:
    plat = db.platforms.get(platform_key)
    if not plat: return None
    
    is_air = plat.unit_type.lower() in ("fighter", "attacker", "helicopter", "awacs")
    img_path = "assets/blue_jet.png" if (is_air and side == "Blue") else None
    
    return Unit(
        uid        = get_next_uid(side.lower()),
        callsign   = callsign,
        lat        = lat,
        lon        = lon,
        side       = side,
        platform   = plat,
        loadout    = dict(plat.default_loadout),
        image_path = img_path,
        drunkness  = 1, 
        corruption = 1
    )


class CampaignBuilder:
    @staticmethod
    def _get_drunkness(rng: random.Random, is_elite: bool) -> int:
        if is_elite:
            return rng.choices([1, 2, 3, 4, 5], weights=[10, 40, 40, 8, 2])[0]
        else:
            return rng.choices([1, 2, 3, 4, 5], weights=[10, 20, 40, 20, 10])[0]

    @staticmethod
    def _get_corruption(rng: random.Random, is_elite: bool) -> int:
        if is_elite:
            return rng.choices([1, 2, 3, 4, 5], weights=[15, 30, 40, 10, 5])[0]
        else:
            return rng.choices([1, 2, 3, 4, 5], weights=[5, 15, 30, 30, 20])[0]

    @staticmethod
    def generate_historical_campaign(db: Database) -> dict:
        rng = random.Random()
        units = []
        events = []
        uid_counter = 0
        
        corr_south = CampaignBuilder._get_corruption(rng, is_elite=False)
        corr_north = CampaignBuilder._get_corruption(rng, is_elite=False)
        corr_front_sam = CampaignBuilder._get_corruption(rng, is_elite=True)
        corr_crimea = CampaignBuilder._get_corruption(rng, is_elite=True)

        # ── 1. Red Ground Forces (Heavy Deployment: South/East) ──────────
        for _ in range(300):
            uid_counter += 1
            utype = rng.choices(["tank", "ifv", "apc", "artillery"], weights=[35, 35, 20, 10])[0]
            pool = {"tank": ["T-72R", "T-80R", "T-90R"], "ifv": ["BMP-2R", "BMP-3R"], "apc": ["BTR-80R", "MTLBR"], "artillery": ["2S1_Gvozdika", "2S3_Akatsiya", "BM-21_Grad"]}[utype]
            plat_key = rng.choice(pool)
            
            if utype in ("ifv", "apc"): min_d, max_d = 20.0, 35.0
            elif utype == "tank": min_d, max_d = 25.0, 40.0
            elif utype == "artillery": min_d, max_d = 30.0, 50.0
            
            while True:
                flat, flon = get_front_line_coords("Red", min_d, max_d, segment_range=(0, 18))
                if not is_water(flat, flon): break
            
            drunk = CampaignBuilder._get_drunkness(rng, is_elite=False)
            
            units.append({
                "id": f"red_gnd_{uid_counter}", "platform": plat_key, "callsign": f"RF GROUP {uid_counter}",
                "side": "Red", "lat": round(flat, 5), "lon": round(flon, 5),
                "drunkness": drunk, "corruption": corr_south,
                "loadout": dict(db.platforms[plat_key].default_loadout), "waypoints": []
            })

        # ── 2. Red Ground Forces (Light Deployment: North/Belarus Border) ──
        for _ in range(50):
            uid_counter += 1
            utype = rng.choices(["tank", "ifv", "apc", "artillery"], weights=[30, 40, 20, 10])[0]
            pool = {"tank": ["T-72R", "T-80R"], "ifv": ["BMP-2R"], "apc": ["BTR-80R", "MTLBR"], "artillery": ["2S1_Gvozdika", "BM-21_Grad"]}[utype]
            plat_key = rng.choice(pool)
            
            if utype in ("ifv", "apc"): min_d, max_d = 20.0, 35.0
            elif utype == "tank": min_d, max_d = 25.0, 40.0
            elif utype == "artillery": min_d, max_d = 30.0, 50.0
            
            while True:
                flat, flon = get_front_line_coords("Red", min_d, max_d, segment_range=(19, 24))
                if not is_water(flat, flon): break
                
            drunk = CampaignBuilder._get_drunkness(rng, is_elite=False)
                
            units.append({
                "id": f"red_gnd_{uid_counter}", "platform": plat_key, "callsign": f"NORTH GRP {uid_counter}",
                "side": "Red", "lat": round(flat, 5), "lon": round(flon, 5),
                "drunkness": drunk, "corruption": corr_north,
                "loadout": dict(db.platforms[plat_key].default_loadout), "waypoints": []
            })

        # ── 3. Red Air Bases (With local S-400 defense) ───────────────────
        red_bases = [
            {"id": "red_base_1", "lat": 47.25, "lon": 39.72, "callsign": "ROSTOV-ON-DON"},
            {"id": "red_base_2", "lat": 45.03, "lon": 38.97, "callsign": "KRASNODAR"},
            {"id": "red_base_3", "lat": 44.68, "lon": 33.56, "callsign": "BELBEK AIR"},
            {"id": "red_base_4", "lat": 48.93, "lon": 40.39, "callsign": "MILLEROVO AIR"},
            {"id": "red_base_5", "lat": 51.75, "lon": 36.29, "callsign": "KURSK AIR"} 
        ]
        for rb in red_bases:
            base_corr = CampaignBuilder._get_corruption(rng, is_elite=True)
            rb["corr"] = base_corr
            
            drunk = CampaignBuilder._get_drunkness(rng, is_elite=False)
            units.append({"id": rb["id"], "platform": "AirbaseR", "callsign": rb["callsign"], "side": "Red", "lat": rb["lat"], "lon": rb["lon"], "drunkness": drunk, "corruption": base_corr, "loadout": {}, "waypoints": []})
            
            uid_counter += 1
            slat, slon = rb["lat"] + rng.uniform(-0.05, 0.05), rb["lon"] + rng.uniform(-0.05, 0.05)
            drunk = CampaignBuilder._get_drunkness(rng, is_elite=True)
            units.append({"id": f"r_s400_{uid_counter}", "platform": "S-400", "callsign": f"TRIUMF {uid_counter}", "side": "Red", "lat": slat, "lon": slon, "drunkness": drunk, "corruption": base_corr, "loadout": dict(db.platforms["S-400"].default_loadout), "waypoints": []})

        # ── 4. Critical Infrastructure (Real Russian Targets) ─────────────
        REAL_TARGETS = [
            {"key": "Red_Refinery", "name": "Tuapse Oil Refinery", "lat": 44.103, "lon": 39.093, "pts": 500},
            {"key": "Red_Refinery", "name": "Novoshakhtinsk Refinery", "lat": 48.783, "lon": 39.920, "pts": 500},
            {"key": "Red_Refinery", "name": "Ryazan Refinery", "lat": 54.580, "lon": 39.776, "pts": 500},
            {"key": "Red_PowerPlant", "name": "Novocherkasskaya GRES", "lat": 47.399, "lon": 40.038, "pts": 400},
            {"key": "Red_PowerPlant", "name": "Kursk Nuclear Plant", "lat": 51.674, "lon": 35.603, "pts": 1000},
            {"key": "Red_CommandCenter", "name": "SMD HQ Rostov", "lat": 47.222, "lon": 39.719, "pts": 600},
            {"key": "Red_CommandCenter", "name": "Voronezh C2 Bunker", "lat": 51.624, "lon": 39.145, "pts": 600},
            {"key": "Red_AmmoDepot", "name": "Tikhoretsk Arsenal", "lat": 45.882, "lon": 40.042, "pts": 300},
            {"key": "Red_AmmoDepot", "name": "Karachev Arsenal", "lat": 53.123, "lon": 35.011, "pts": 300},
            {"key": "Red_AmmoDepot", "name": "Toropets Arsenal", "lat": 56.495, "lon": 31.720, "pts": 300},
            {"key": "Red_Kremlin", "name": "The Kremlin", "lat": 55.752, "lon": 37.617, "pts": 5000},
        ]

        for t in REAL_TARGETS:
            uid_counter += 1
            infra_uid = f"red_infra_{uid_counter}"
            
            plat = db.platforms.get(t["key"])
            if not plat: continue
            
            is_kremlin = (t["key"] == "Red_Kremlin")
            infra_corr = CampaignBuilder._get_corruption(rng, is_elite=is_kremlin)
            drunk = CampaignBuilder._get_drunkness(rng, is_elite=is_kremlin)
            
            units.append({
                "id": infra_uid, "platform": t["key"], "callsign": t["name"].upper(),
                "side": "Red", "lat": t["lat"], "lon": t["lon"],
                "drunkness": drunk, "corruption": infra_corr,
                "loadout": {}, "waypoints": []
            })

            events.append({
                "id": f"evt_score_{infra_uid}", "condition_type": "UNIT_DEAD", "condition_val": infra_uid,
                "action_type": "SCORE", "action_val": f"Blue:{t['pts']}"
            })
            events.append({
                "id": f"evt_log_{infra_uid}", "condition_type": "UNIT_DEAD", "condition_val": infra_uid,
                "action_type": "LOG", "action_val": f"CRITICAL HIT: {t['name']} has been destroyed!"
            })

            if is_kremlin:
                events.append({
                    "id": f"evt_win_{infra_uid}", "condition_type": "UNIT_DEAD", "condition_val": infra_uid,
                    "action_type": "VICTORY", "action_val": "Blue"
                })
                
                moscow_ring_radius_km = 40.0
                lat_rad_clamped = max(0.0001, math.cos(math.radians(t["lat"])))
                for i in range(6):
                    uid_counter += 1
                    angle = (360 / 6) * i
                    slat = t["lat"] + (math.cos(math.radians(angle)) * moscow_ring_radius_km) / 111.32
                    slon = t["lon"] + (math.sin(math.radians(angle)) * moscow_ring_radius_km) / (111.32 * lat_rad_clamped)
                    drunk = CampaignBuilder._get_drunkness(rng, is_elite=True)
                    units.append({
                        "id": f"r_s400_{uid_counter}", "platform": "S-400", "callsign": f"MOSCOW DEF {i+1}", 
                        "side": "Red", "lat": round(slat, 5), "lon": round(slon, 5), 
                        "drunkness": drunk, "corruption": infra_corr,
                        "loadout": dict(db.platforms["S-400"].default_loadout), "waypoints": []
                    })
            else:
                uid_counter += 1
                sam_uid = f"r_s400_{uid_counter}"
                sam_plat = db.platforms.get("S-400")
                if sam_plat:
                    angle = rng.uniform(0, 360)
                    dist_km = 10.0
                    lat_rad = max(0.0001, math.cos(math.radians(t["lat"])))
                    sam_lat = t["lat"] + (math.cos(math.radians(angle)) * dist_km) / 111.32
                    sam_lon = t["lon"] + (math.sin(math.radians(angle)) * dist_km) / (111.32 * lat_rad)
                    
                    drunk = CampaignBuilder._get_drunkness(rng, is_elite=True)
                    units.append({
                        "id": sam_uid, "platform": "S-400", "callsign": f"{t['name'][:8].upper()} DEF",
                        "side": "Red", "lat": round(sam_lat, 5), "lon": round(sam_lon, 5),
                        "drunkness": drunk, "corruption": infra_corr,
                        "loadout": dict(sam_plat.default_loadout), "waypoints": []
                    })

        # ── 4b. Dispersed Air Defense (Crimea) ────────────────────────────────
        crimea_points = [(45.28, 34.03), (45.70, 34.42), (44.95, 34.10), (45.30, 33.30)]
        for cp in crimea_points:
            uid_counter += 1
            drunk = CampaignBuilder._get_drunkness(rng, is_elite=True)
            units.append({"id": f"r_s400_{uid_counter}", "platform": "S-400", "callsign": f"TRIUMF {uid_counter}", "side": "Red", "lat": cp[0] + rng.uniform(-0.05, 0.05), "lon": cp[1] + rng.uniform(-0.05, 0.05), "drunkness": drunk, "corruption": corr_crimea, "loadout": dict(db.platforms["S-400"].default_loadout), "waypoints": []})

        # ── 4c. Integrated Air Defense System (IADS) Network ──────────────────
        num_iads_nodes = 12
        for i in range(num_iads_nodes):
            f = (i + 0.5) / num_iads_nodes
            total_pts = len(FRONT_LINE_POINTS) - 1
            idx = int(f * total_pts)
            rem = (f * total_pts) - idx
            
            p1 = FRONT_LINE_POINTS[idx]
            p2 = FRONT_LINE_POINTS[idx+1]
            clat = p1[0] + rem * (p2[0] - p1[0])
            clon = p1[1] + rem * (p2[1] - p1[1])
            
            brg = bearing(p1[0], p1[1], p2[0], p2[1]) 
            norm_east = (brg + 90) % 360              
            lat_rad = max(0.0001, math.cos(math.radians(clat)))
            
            # Tier 1: Medium Range (Buk-M2) ~ 70km behind lines
            m_lat = clat + (math.cos(math.radians(norm_east)) * 70.0) / 111.32 
            m_lon = clon + (math.sin(math.radians(norm_east)) * 70.0) / (111.32 * lat_rad)
            uid_counter += 1
            units.append({"id": f"r_sam_{uid_counter}", "platform": "Buk-M2", "callsign": f"BUK BATTERY {i+1}", "side": "Red", "lat": round(m_lat, 5), "lon": round(m_lon, 5), "drunkness": CampaignBuilder._get_drunkness(rng, True), "corruption": corr_front_sam, "loadout": dict(db.platforms["Buk-M2"].default_loadout), "waypoints": []})

            # Tier 2: Short Range (Tor-M1) ~ 50km behind lines
            s_lat = clat + (math.cos(math.radians(norm_east)) * 50.0) / 111.32 
            s_lon = clon + (math.sin(math.radians(norm_east)) * 50.0) / (111.32 * lat_rad)
            uid_counter += 1
            units.append({"id": f"r_sam_{uid_counter}", "platform": "Tor-M1", "callsign": f"TOR BATTERY {i+1}", "side": "Red", "lat": round(s_lat, 5), "lon": round(s_lon, 5), "drunkness": CampaignBuilder._get_drunkness(rng, False), "corruption": corr_front_sam, "loadout": dict(db.platforms["Tor-M1"].default_loadout), "waypoints": []})
            
            # Tier 3: Strategic (S-400) mixed in roughly every 3rd node, ~ 100km back
            if i % 3 == 1:
                st_lat = clat + (math.cos(math.radians(norm_east)) * 100.0) / 111.32 
                st_lon = clon + (math.sin(math.radians(norm_east)) * 100.0) / (111.32 * lat_rad)
                uid_counter += 1
                units.append({"id": f"r_s400_{uid_counter}", "platform": "S-400", "callsign": f"FRONT DEF {i+1}", "side": "Red", "lat": round(st_lat, 5), "lon": round(st_lon, 5), "drunkness": CampaignBuilder._get_drunkness(rng, True), "corruption": corr_front_sam, "loadout": dict(db.platforms["S-400"].default_loadout), "waypoints": []})

        # ── 5. Stand-off CAP Stations (3 Pairs per Base) ───────────────────
        for b_idx, rb in enumerate(red_bases):
            best_dist = 999999.0
            best_idx = 0
            for i, p in enumerate(FRONT_LINE_POINTS):
                d = haversine(rb["lat"], rb["lon"], p[0], p[1])
                if d < best_dist:
                    best_dist = d
                    best_idx = i
                    
            p1 = FRONT_LINE_POINTS[best_idx]
            if best_idx < len(FRONT_LINE_POINTS) - 1: p2 = FRONT_LINE_POINTS[best_idx+1]
            else: p1 = FRONT_LINE_POINTS[best_idx-1]; p2 = FRONT_LINE_POINTS[best_idx]
            
            brg = bearing(p1[0], p1[1], p2[0], p2[1]) 
            norm_east = (brg + 90) % 360              
            
            lat_rad = max(0.0001, math.cos(math.radians(p1[0])))
            
            so_lat = p1[0] + (math.cos(math.radians(norm_east)) * 80.0) / 111.32
            so_lon = p1[1] + (math.sin(math.radians(norm_east)) * 80.0) / (111.32 * lat_rad)
            axis_val = round(brg, 1) if round(brg, 1) != 0.0 else 360.0
            
            for pair_idx in range(3):
                leader_uid = f"red_air_{uid_counter+1}"
                
                state = "ACTIVE" if pair_idx == 0 else "READY"
                timer = 0.0
                
                rtb_fuel = 0.25 + (rng.uniform(0.0, 0.05))
                plat_key = rng.choice(["Su-35S", "Su-30SM"])

                for wing_idx in range(2):
                    uid_counter += 1
                    is_leader = (wing_idx == 0)
                    drunk = CampaignBuilder._get_drunkness(rng, is_elite=True)
                    
                    u_dict = {
                        "id": f"red_air_{uid_counter}",
                        "platform": plat_key,
                        "callsign": f"VULTUR {b_idx+1}-{pair_idx+1}{'A' if is_leader else 'B'}",
                        "side": "Red",
                        "lat": round(so_lat + rng.uniform(-0.02, 0.02) if state == "ACTIVE" else rb["lat"], 5),
                        "lon": round(so_lon + rng.uniform(-0.02, 0.02) if state == "ACTIVE" else rb["lon"], 5),
                        "drunkness": drunk,
                        "corruption": rb["corr"],
                        "home_uid": rb["id"],
                        "duty_state": state,
                        "duty_timer": timer,
                        "mission": {
                            "name": f"{rb['callsign']} CAP",
                            "type": "CAP",
                            "lat": round(so_lat, 5),
                            "lon": round(so_lon, 5),
                            "radius": 60.0,
                            "alt": 30000.0,
                            "rtb_fuel": round(rtb_fuel, 2),
                            "time_on_target": axis_val 
                        },
                        "waypoints": []
                    }
                    if not is_leader:
                        u_dict["leader_uid"] = leader_uid
                        u_dict["formation_slot"] = 1
                    else:
                        u_dict["flight_doctrine"] = "STANDARD"
                        
                    units.append(u_dict)

        return {
            "name": "Operation: Deep Strike",
            "description": "Massive front-line. Penetrate S-400 defense rings and strike critical infrastructure.",
            "start_lat": 49.5,
            "start_lon": 34.0,
            "start_zoom": 6,
            "units": units,
            "events": events
        }

    @staticmethod
    def deploy_blue_forces(db: Database, sim: "SimulationEngine", placement_counts: dict[str, int]) -> None:
        rng = random.Random()
        
        blue_bases = [
            {"id": "base_west", "lat": 49.83, "lon": 24.02, "name": "LVIV AIR"},
            {"id": "base_central", "lat": 50.45, "lon": 30.52, "name": "KYIV HUB"},
            {"id": "base_south", "lat": 46.48, "lon": 30.72, "name": "ODESA AIR"},
            {"id": "base_east", "lat": 48.46, "lon": 35.04, "name": "DNIPRO AIR"},
            {"id": "base_north", "lat": 51.50, "lon": 31.30, "name": "CHERNIHIV AIR"} 
        ]
        
        base_map = {}
        for b in blue_bases:
            n = placement_counts.get("AirbaseB", 0) + 1
            placement_counts["AirbaseB"] = n
            unit = make_live_unit("AirbaseB", b["lat"], b["lon"], "Blue", db, b["name"])
            if unit:
                sim.units.append(unit)
                base_map[b["id"]] = unit.uid

            if b["id"] in ("base_central", "base_south"): 
                sam_type = "Patriot" 
            elif b["id"] == "base_east":
                sam_type = "S-300PS"
            else:
                sam_type = rng.choice(["NASAMS", "IRIS-T_SLM"])
                
            n_sam = placement_counts.get(sam_type, 0) + 1
            placement_counts[sam_type] = n_sam
            plat, plon = b["lat"] + rng.uniform(-0.05, 0.05), b["lon"] + rng.uniform(-0.05, 0.05)
            u_sam = make_live_unit(sam_type, plat, plon, "Blue", db, get_callsign_for(sam_type, n_sam))
            if u_sam: sim.units.append(u_sam)
            
            n_flam = placement_counts.get("Flamingo_TEL", 0) + 1
            placement_counts["Flamingo_TEL"] = n_flam
            flat, flon = b["lat"] + rng.uniform(-0.02, 0.02), b["lon"] + rng.uniform(-0.02, 0.02)
            u_flam = make_live_unit("Flamingo_TEL", flat, flon, "Blue", db, get_callsign_for("Flamingo_TEL", n_flam))
            if u_flam: sim.units.append(u_flam)

        # ── 1. Blue Ground Forces (Heavy Deployment: South/East) ──────────
        for _ in range(300):
            utype = rng.choices(["tank", "ifv", "apc", "artillery"], weights=[35, 35, 20, 10])[0]
            
            if utype == "tank":
                plat_key = rng.choices(["Leopard2", "M1Abrams", "Challenger2", "T-64BV", "T-72", "T-80BV"], weights=[10, 5, 2, 40, 33, 10])[0]
            elif utype == "ifv":
                plat_key = rng.choices(["Bradley", "CV9040C", "Marder1A3", "BMP-1"], weights=[20, 5, 10, 65])[0]
            elif utype == "apc":
                plat_key = rng.choices(["Stryker", "BTR-4", "M113", "MaxxPro", "MT-LB"], weights=[15, 15, 35, 15, 20])[0]
            elif utype == "artillery":
                plat_key = rng.choices(["M142_HIMARS", "M270_MLRS", "M777", "PzH2000", "CAESAR", "2S1_Gvozdika", "2S3_Akatsiya", "AHS_Krab"], weights=[4, 1, 20, 5, 5, 35, 20, 10])[0]
                
            if utype in ("ifv", "apc"): min_d, max_d = 20.0, 35.0
            elif utype == "tank": min_d, max_d = 25.0, 40.0
            elif utype == "artillery": min_d, max_d = 30.0, 50.0
            
            while True:
                flat, flon = get_front_line_coords("Blue", min_d, max_d, segment_range=(0, 18))
                if not is_water(flat, flon): break
            
            n = placement_counts.get(plat_key, 0) + 1
            placement_counts[plat_key] = n
            unit = make_live_unit(plat_key, flat, flon, "Blue", db, get_callsign_for(plat_key, n))
            if unit: sim.units.append(unit)

        # ── 2. Blue Ground Forces (Light Deployment: North/Belarus Border) ──
        for _ in range(50):
            utype = rng.choices(["tank", "ifv", "apc", "artillery"], weights=[30, 40, 20, 10])[0]
            
            if utype == "tank":
                plat_key = rng.choices(["Leopard1A5", "T-72", "PT-91"], weights=[20, 60, 20])[0]
            elif utype == "ifv":
                plat_key = rng.choices(["Marder1A3", "BMP-1"], weights=[20, 80])[0]
            elif utype == "apc":
                plat_key = rng.choices(["M113", "VAB", "Bushmaster", "MT-LB"], weights=[40, 20, 10, 30])[0]
            elif utype == "artillery":
                plat_key = rng.choices(["M109", "AHS_Krab", "BM-21_Grad"], weights=[20, 10, 70])[0]
                
            if utype in ("ifv", "apc"): min_d, max_d = 20.0, 35.0
            elif utype == "tank": min_d, max_d = 25.0, 40.0
            elif utype == "artillery": min_d, max_d = 30.0, 50.0
            
            while True:
                flat, flon = get_front_line_coords("Blue", min_d, max_d, segment_range=(19, 24))
                if not is_water(flat, flon): break
                
            n = placement_counts.get(plat_key, 0) + 1
            placement_counts[plat_key] = n
            unit = make_live_unit(plat_key, flat, flon, "Blue", db, get_callsign_for(plat_key, n))
            if unit: sim.units.append(unit)

        # ── 3. Integrated Air Defense System (IADS) Network ──────────────────
        num_iads_nodes = 12
        for i in range(num_iads_nodes):
            f = (i + 0.5) / num_iads_nodes
            total_pts = len(FRONT_LINE_POINTS) - 1
            idx = int(f * total_pts)
            rem = (f * total_pts) - idx
            
            p1 = FRONT_LINE_POINTS[idx]
            p2 = FRONT_LINE_POINTS[idx+1]
            clat = p1[0] + rem * (p2[0] - p1[0])
            clon = p1[1] + rem * (p2[1] - p1[1])
            
            brg = bearing(p1[0], p1[1], p2[0], p2[1]) 
            norm_west = (brg - 90) % 360
            lat_rad = max(0.0001, math.cos(math.radians(clat)))
            
            # Tier 1: Medium Range (~70km back)
            m_lat = clat + (math.cos(math.radians(norm_west)) * 70.0) / 111.32 
            m_lon = clon + (math.sin(math.radians(norm_west)) * 70.0) / (111.32 * lat_rad)
            plat_key = rng.choices(["Buk-M1_UA", "NASAMS", "IRIS-T_SLM"], weights=[50, 25, 25])[0]
            n = placement_counts.get(plat_key, 0) + 1
            placement_counts[plat_key] = n
            u_sam = make_live_unit(plat_key, round(m_lat, 5), round(m_lon, 5), "Blue", db, f"MED DEF {n}")
            if u_sam: sim.units.append(u_sam)

            # Tier 2: Short Range (~50km back)
            s_lat = clat + (math.cos(math.radians(norm_west)) * 50.0) / 111.32 
            s_lon = clon + (math.sin(math.radians(norm_west)) * 50.0) / (111.32 * lat_rad)
            plat_key = rng.choices(["Gepard", "Osa-AKM", "Stormer_HVM", "Avenger"], weights=[40, 30, 15, 15])[0]
            n = placement_counts.get(plat_key, 0) + 1
            placement_counts[plat_key] = n
            u_sam = make_live_unit(plat_key, round(s_lat, 5), round(s_lon, 5), "Blue", db, f"SHORAD {n}")
            if u_sam: sim.units.append(u_sam)

            # Tier 3: Strategic (~100km back, every 3rd node)
            if i % 3 == 1:
                st_lat = clat + (math.cos(math.radians(norm_west)) * 100.0) / 111.32 
                st_lon = clon + (math.sin(math.radians(norm_west)) * 100.0) / (111.32 * lat_rad)
                plat_key = rng.choices(["Patriot", "S-300PS"], weights=[40, 60])[0]
                n = placement_counts.get(plat_key, 0) + 1
                placement_counts[plat_key] = n
                u_sam = make_live_unit(plat_key, round(st_lat, 5), round(st_lon, 5), "Blue", db, f"STRAT DEF {n}")
                if u_sam: sim.units.append(u_sam)

        # ── 4. Stand-off CAP Stations (Frontline Cover) ──────────────────────
        for b_idx, rb in enumerate(blue_bases):
            best_dist = 999999.0
            best_idx = 0
            for i, p in enumerate(FRONT_LINE_POINTS):
                d = haversine(rb["lat"], rb["lon"], p[0], p[1])
                if d < best_dist:
                    best_dist = d
                    best_idx = i
                    
            p1 = FRONT_LINE_POINTS[best_idx]
            if best_idx < len(FRONT_LINE_POINTS) - 1: p2 = FRONT_LINE_POINTS[best_idx+1]
            else: p1 = FRONT_LINE_POINTS[best_idx-1]; p2 = FRONT_LINE_POINTS[best_idx]
            
            brg = bearing(p1[0], p1[1], p2[0], p2[1]) 
            norm_west = (brg - 90) % 360              
            lat_rad = max(0.0001, math.cos(math.radians(p1[0])))
            
            so_lat = p1[0] + (math.cos(math.radians(norm_west)) * 80.0) / 111.32
            so_lon = p1[1] + (math.sin(math.radians(norm_west)) * 80.0) / (111.32 * lat_rad)
            axis_val = round(brg, 1) if round(brg, 1) != 0.0 else 360.0
            
            for pair_idx in range(3):
                leader_uid = ""
                state = "ACTIVE" if pair_idx == 0 else "READY"
                timer = 0.0
                
                rtb_fuel = 0.25 + (rng.uniform(0.0, 0.05))
                plat_key = rng.choices(["F-16AM", "F-16UA", "Mirage2000-5F", "Su-27UA", "MiG-29UA"], weights=[20, 10, 5, 25, 40])[0]

                for wing_idx in range(2):
                    n = placement_counts.get(plat_key, 0) + 1
                    placement_counts[plat_key] = n
                    is_leader = (wing_idx == 0)
                    
                    u_lat = round(so_lat + rng.uniform(-0.02, 0.02) if state == "ACTIVE" else rb["lat"], 5)
                    u_lon = round(so_lon + rng.uniform(-0.02, 0.02) if state == "ACTIVE" else rb["lon"], 5)
                    
                    callsign = get_callsign_for(plat_key, n)
                    
                    unit = make_live_unit(plat_key, u_lat, u_lon, "Blue", db, callsign)
                    if not unit: continue
                    
                    if is_leader: leader_uid = unit.uid
                    
                    unit.home_uid = base_map[rb["id"]]
                    unit.duty_state = state
                    unit.duty_timer = timer
                    
                    unit.mission = Mission(
                        name=f"{rb['name']} CAP",
                        mission_type="CAP",
                        target_lat=round(so_lat, 5),
                        target_lon=round(so_lon, 5),
                        radius_km=60.0,
                        altitude_ft=30000.0,
                        rtb_fuel_pct=round(rtb_fuel, 2),
                        time_on_target=axis_val 
                    )
                    
                    if not is_leader:
                        unit.leader_uid = leader_uid
                        unit.formation_slot = 1
                    else:
                        unit.flight_doctrine = "STANDARD"
                        
                    sim.units.append(unit)

        # ── 4b. Strategic Capital Defense CAP (Kyiv) ────────────────────────
        kyiv_base = next((b for b in blue_bases if b["id"] == "base_central"), None)
        if kyiv_base:
            plat_key = "F-16UA"
            leader_uid = ""
            for wing_idx in range(2):
                n = placement_counts.get(plat_key, 0) + 1
                placement_counts[plat_key] = n
                is_leader = (wing_idx == 0)
                
                u_lat = kyiv_base["lat"] + rng.uniform(-0.02, 0.02)
                u_lon = kyiv_base["lon"] + rng.uniform(-0.02, 0.02)
                callsign = f"CAPITAL 1{'A' if is_leader else 'B'}"
                
                unit = make_live_unit(plat_key, u_lat, u_lon, "Blue", db, callsign)
                if unit:
                    if is_leader: leader_uid = unit.uid
                    unit.home_uid = base_map[kyiv_base["id"]]
                    unit.duty_state = "ACTIVE"
                    unit.duty_timer = 0.0
                    
                    unit.mission = Mission(
                        name="Kyiv Defense CAP",
                        mission_type="CAP",
                        target_lat=kyiv_base["lat"],
                        target_lon=kyiv_base["lon"] + 0.8,
                        radius_km=50.0,
                        altitude_ft=35000.0,
                        rtb_fuel_pct=0.20,
                        time_on_target=0.0
                    )
                    
                    if not is_leader:
                        unit.leader_uid = leader_uid
                        unit.formation_slot = 1
                    else:
                        unit.flight_doctrine = "STANDARD"
                        
                    sim.units.append(unit)

        # ── 5. Strategic Assets (AWACS, Strike) ──────────────────────────────
        strategic_assets = [("E-3G_Sentry", 4), ("Su-24M", 12)]
        for utype, qty in strategic_assets:
            for _ in range(qty):
                target_base = rng.choice(blue_bases)
                n = placement_counts.get(utype, 0) + 1
                placement_counts[utype] = n
                
                unit = make_live_unit(utype, target_base["lat"], target_base["lon"], "Blue", db, get_callsign_for(utype, n))
                if unit:
                    if unit.platform.unit_type in ("fighter", "attacker", "awacs"):
                        unit.home_uid = base_map[target_base["id"]]
                        unit.duty_state = "READY"
                        unit.altitude_ft = get_elevation_ft(unit.lat, unit.lon) + 15.0
                        unit.target_altitude_ft = unit.altitude_ft 
                    sim.units.append(unit)

        sim.log("Full theater Blue forces deployed via Campaign Builder.")