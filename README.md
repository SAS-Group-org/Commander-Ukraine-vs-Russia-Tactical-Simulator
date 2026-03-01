# WarSim — Real-Time Air & Ground Combat Simulator

A physics-based, real-time wargame simulation built with Python and pygame, featuring authentic sensor modelling, electronic warfare, and multi-domain (air + ground) combat set in a modern eastern European theatre.

---

## Features

- **CARTO Voyager map** — Web-Mercator tile streaming with English labels, disk cache, and 4 concurrent download workers
- **Physics-based sensor model** — Radar detection ranges adjusted for RCS, ECM jamming, radar horizon (earth curvature), and altitude
- **Fog of War** — Blue player sees only what their sensors can detect; contacts degrade through FAINT → PROBABLE → CONFIRMED classifications
- **Electronic warfare** — Active ECM jamming reduces enemy detection range; burn-through range forces detection at close quarters; chaff and flares defeat radar and IR-guided missiles
- **Multi-domain combat** — Air-to-air, air-to-ground, and ground-to-ground engagements using domain-locked weapon rules
- **Damage model** — Four-state health system (OK / LIGHT / MODERATE / HEAVY / KILLED) that degrades speed, radar performance, and climb rate
- **Fuel system** — Units burn fuel in real time and automatically RTB at configurable bingo fuel levels; crashes if fuel reaches zero
- **Airbase & duty cycle** — Aircraft park at airbases between sorties; on landing they rearm and refuel over a timed ground turn before becoming available again
- **Salvo doctrine** — Single Last Shot (SLS) and free-salvo firing modes; AI fire cooldowns prevent spam
- **Adjustable time compression** — PAUSE / 1× / 15× / 60× / 300×
- **Scenario & deployment save/load** — JSON-based format for full state persistence
- **Setup + combat modes** — Place units on the map in setup mode, then transition to live combat

---

## Requirements

```
Python 3.11+
pygame
pygame-gui
requests
```

Install dependencies:

```bash
pip install pygame pygame-gui requests
```

---

## Running the Game

```bash
python main.py
```

An optional scenario file path can be specified at launch. Without one, the game starts in **Setup Mode** so you can build your own deployment.

---

## Game Modes

### Setup Mode

The bottom panel shows a unit roster. Select a platform, set a quantity, click **Place on Map**, then click a location on the map to deploy units. Right-clicking a placed unit removes it. When satisfied, click **Start Combat** to begin the simulation.

Use **Save Deployment** and **Load Deployment** in the panel to persist Blue-side force compositions between sessions. You can also save the full scenario at any time with **Ctrl+S**.

### Combat Mode

The simulation runs in real time. Units with assigned missions and waypoints will navigate autonomously. Select a Blue unit to issue orders from the bottom panel.

---

## Controls

| Input | Action |
|---|---|
| **Left-click** (map) | Select a unit |
| **Right-click** (map, combat) | Add waypoint for selected unit (auto-activates a READY unit) |
| **Right-click** (unit, setup) | Remove unit from map |
| **Scroll wheel** | Zoom in / out |
| **Middle-click drag** | Pan camera |
| **1 – 5** | Set time compression (Pause / 1× / 15× / 60× / 300×) |
| **Space** | Toggle pause |
| **Ctrl+S** | Save current scenario |
| **Delete** | Remove selected unit |
| **Escape** | Deselect / cancel placement |

---

## Bottom Panel — Combat Controls

| Control | Description |
|---|---|
| **▲ / ▼ altitude buttons** | Climb or dive selected aircraft by 500 / 1k / 5k ft |
| **AUTO** | Toggle autonomous weapon engagement for selected unit |
| **ROE** | Cycle Rules of Engagement: **TIGHT** (confirmed only) → **FREE** (probable+) → **HOLD** (no fire) |
| **ECM** | Toggle active jamming on the selected unit |
| **ASSIGN CAP** | Assign a Combat Air Patrol mission (click map to set centre) |
| **CLEAR MSN** | Cancel active mission |
| **Weapon buttons** | Select active weapon for manual fire |
| **SLS / Salvo** | Toggle salvo doctrine (SLS = one missile at a time; Salvo = continuous fire) |
| **FOG OF WAR** | Toggle between full visibility and sensor-only contact view |
| **Save Deployment / Load Deployment** | (Setup mode) Persist or restore Blue force layout |

When an **airbase** unit is selected in combat, the panel also shows a launch button for each parked aircraft. Aircraft showing `READY` can be scrambled immediately; those showing `REARM Xs` are still on the ground turn.

---

## Sensor & Detection Model

Detection range for each radar is computed as:

```
R_effective = R_rcs × (1 − ECM_penalty)

where:
  R_rcs  = radar_range_km × performance_mult × (target_rcs / reference_rcs)^0.25
  ECM_penalty = target.ecm_rating × 0.60   (only beyond burn-through range of 15 km)
```

Contacts are classified by how close they are relative to the effective detection range:

| Band | Threshold | Classification | Info revealed |
|---|---|---|---|
| Outer | > 75% of range | **FAINT** | Position only, grey blip |
| Middle | 50–75% of range | **PROBABLE** | Position + unit type, amber symbol |
| Inner | < 50% of range | **CONFIRMED** | Full identity + callsign, red symbol |

Contacts time out after **30 seconds** without a refresh.

---

## Electronic Warfare

- **Active ECM jamming** — Automatically enabled on units with `ecm_rating > 0` when threats are detected. Scales radar detection range of the sensor by `1 − (ecm_rating × 0.60)`.
- **Burn-through** — Jamming is ineffective within **15 km**; the radar overpowers it.
- **Chaff** — Depletes one bundle per incoming SARH/ARH missile; reduces Pk by **−0.25** per bundle.
- **Flares** — Depletes one bundle per incoming IR missile; reduces Pk by **−0.25** per bundle.
- **ECCM** — Weapons have an ECCM rating that partially offsets ECM effects.

---

## Airbase & Duty Cycle

Aircraft are assigned a **home airbase** via `home_uid`. Between sorties they exist in one of four duty states:

| State | Meaning |
|---|---|
| `READY` | On the ground, fully armed and fuelled — can be launched |
| `ACTIVE` | Airborne and operating |
| `REARMING` | On the ground, reloading weapons and refuelling (timer counts down) |

**Sortie lifecycle:**

1. Player (or scenario file) launches a READY aircraft → state becomes `ACTIVE`, unit climbs to cruise altitude and begins its mission.
2. When bingo fuel is reached the unit's mission switches to `RTB`. It navigates to the nearest friendly airbase.
3. On arrival (within 2 km) the unit lands: fuel is restored to full, loadout is reset to default, and state changes to `REARMING` for `rearm_time_s` seconds (120 s for land bases, 90 s for carriers).
4. After the ground turn completes the unit returns to `READY`.

If the assigned home base is destroyed the unit automatically diverts to the nearest surviving friendly airbase. If no friendly base remains the unit is removed.

**Airbase platforms** (`type: airbase`) are stationary, have very high RCS (making them easy to detect), and carry no weapons. Three variants exist: `AirbaseB` (land base), `AircraftCarrier`, and `MilitaryAirbase`, differing only in display name and rearm time.

When a unit is not `ACTIVE` it is invisible on the map and cannot be targeted.

```
Pk = base_pk − (launch_distance / 50) × 0.10 − ECM_effect − chaff/flare_penalty
Pk = clamp(Pk, 0.05, 0.95)
```

A random roll against Pk determines hit or miss on intercept.

---

## Platforms

### Blue (Ukraine)

| Platform | Type | Speed | Radar | Default Weapons |
|---|---|---|---|---|
| MiG-29 Fulcrum | Fighter | 2400 km/h | 150 km | R-27R, R-27T, R-73 |
| Su-27 Flanker-B | Fighter | 2500 km/h | 200 km | R-27R, R-27T, R-73 |
| F-16AM (MLU) | Fighter | 2150 km/h | 130 km | AIM-120C, AIM-9X |
| F-16 Block 52 | Fighter | 2150 km/h | 148 km | AIM-120C, AIM-9X |
| Su-25 Frogfoot | Attacker | 950 km/h | 20 km | R-60, S-8 rockets |
| Su-24M Fencer-D | Attacker | 1700 km/h | 50 km | Kh-25ML, R-60 |
| Mirage 2000-5F | Fighter | 2530 km/h | 185 km | MICA EM, MICA IR |
| Mi-24V Hind-E | Helicopter | 320 km/h | 20 km | Shturm ATGM, S-8 |
| T-72 (various) | Tank | 60 km/h | 5 km | 125mm gun, Konkurs ATGM |
| Leopard 2A4/A6 | Tank | 72 km/h | 7 km | 120mm NATO, TOW ATGM |
| M1A1 Abrams | Tank | 68 km/h | 7 km | 120mm NATO, TOW ATGM |
| M2 Bradley | IFV | 66 km/h | 5 km | 25mm Bushmaster, TOW |
| Patriot PAC-3 | SAM | — | 160 km | PAC-3 |
| NASAMS II | SAM | — | 75 km | AIM-120 (ground-launched) |
| IRIS-T SLM | SAM | — | 250 km | IRIS-T |
| Flakpanzer Gepard | SPAAG | 65 km/h | 15 km | 35mm Oerlikon |
| Airbase / Runway (`AirbaseB`) | Airbase | — | — | — |
| Military Airbase | Airbase | — | — | — |
| Aircraft Carrier | Airbase | — | — | — |

### Red (Russia)

| Platform | Type | Speed | Radar | Default Weapons |
|---|---|---|---|---|
| Su-27S Flanker-B | Fighter | 2500 km/h | 240 km | R-27R, R-27T, R-73 |
| Su-35S Flanker-E | Fighter | 2500 km/h | 300 km | R-77, R-27T, R-73 |
| MiG-29A Fulcrum-A | Fighter | 2400 km/h | 180 km | R-27R, R-27T, R-73 |
| Su-30SM Flanker-H | Fighter | 2125 km/h | 280 km | R-77, R-73 |
| T-90A / T-90M | Tank | 65 km/h | 5 km | 125mm gun, Konkurs ATGM |
| T-72B3 | Tank | 60 km/h | 5 km | 125mm gun, Konkurs ATGM |
| BMP-2 | IFV | 65 km/h | 4 km | 30mm autocannon, Konkurs |
| BTR-82A | APC | 80 km/h | 3 km | 30mm autocannon |
| BRDM-2 | Recon | 95 km/h | 6 km | Konkurs ATGM |
| S-400 Triumf | SAM | — | 400 km | 48N6E |
| Buk-M2 | SAM | — | 120 km | 9M317 |
| Tor-M1 | SAM | — | 25 km | 9M331 |

---

## Weapons Reference

| Weapon | Seeker | Range | Base Pk | Notes |
|---|---|---|---|---|
| R-77 Adder | ARH | 110 km | 0.84 | Fire-and-forget BVR |
| AIM-120C AMRAAM | ARH | 105 km | 0.85 | NATO standard BVR |
| MICA EM | ARH | 80 km | 0.86 | French ARH BVR/WVR |
| R-27R Alamo-A | SARH | 73 km | 0.82 | Requires radar illumination |
| R-27T Alamo-B | IR | 70 km | 0.80 | IR BVR variant |
| MICA IR | IR | 60 km | 0.87 | French IR BVR/WVR |
| 48N6E (S-400) | SARH | 250 km | 0.85 | Heavy long-range SAM |
| PAC-3 | ARH | 100 km | 0.90 | Hit-to-kill SAM |
| AIM-9X Sidewinder | IR | 35 km | 0.90 | High off-boresight WVR |
| R-73 Archer | IR | 30 km | 0.88 | High-agility dogfight |
| 9M317 (Buk-M2) | SARH | 45 km | 0.80 | Medium tactical SAM |
| AIM-120 (NASAMS) | ARH | 30 km | 0.88 | Ground-launched AMRAAM |
| IRIS-T SLM | IR | 40 km | 0.92 | Imaging IR SAM |
| Kh-25ML | Laser | 25 km | 0.78 | Air-to-ground AGM |
| Stugna-P | Laser | 5.5 km | 0.82 | Ukrainian laser ATGM |
| TOW BGM-71 | SACLOS | 3.7 km | 0.80 | Wire-guided ATGM |
| 9M113 Konkurs | SACLOS | 4.0 km | 0.78 | Wire-guided ATGM |
| 120mm NATO APFSDS | Cannon | 4.5 km | 0.82 | Leopard 2 / M1 main gun |
| 125mm APFSDS | Cannon | 4.0 km | 0.80 | T-72/T-90 main gun |
| M61A1 Vulcan | Cannon | 0.8 km | 0.65 | 20mm rotary (aircraft) |

---

## Scenario File Format

Scenarios are stored as JSON files. Aircraft units now include `home_uid`, `duty_state`, and `duty_timer` fields:

```json
{
  "name": "Scenario Name",
  "description": "Brief description",
  "start_lat": 50.0,
  "start_lon": 30.0,
  "start_zoom": 7,
  "units": [
    {
      "id": "base_001",
      "platform": "AirbaseB",
      "callsign": "ALPHA BASE 1",
      "side": "Blue",
      "lat": 50.08, "lon": 25.57,
      "loadout": {},
      "roe": "TIGHT",
      "home_uid": "",
      "duty_state": "ACTIVE",
      "duty_timer": 0.0,
      "waypoints": []
    },
    {
      "id": "unit_001",
      "platform": "F-16AM",
      "callsign": "VIPER 1",
      "side": "Blue",
      "lat": 50.08, "lon": 25.57,
      "loadout": {"AIM-120C": 4, "AIM-9X": 2, "M61A1": 1},
      "roe": "TIGHT",
      "home_uid": "base_001",
      "duty_state": "READY",
      "duty_timer": 0.0,
      "waypoints": []
    }
  ]
}
```

### Supported Mission Types

| Type | Behaviour |
|---|---|
| `CAP` | Combat Air Patrol — orbits a defined area at set altitude |
| `PATROL` | Ground patrol — roams within radius of a point |
| `STRIKE` | Strike mission — orbits target area |
| `SEAD` | Suppression of enemy air defences |
| `RTB` | Return to base — navigates to home coordinates |

---

## Project Structure

```
main.py             — Entry point, event loop, app state
scenario.py         — Data models, platform/weapon DB, save/load
simulation.py       — Real-time engine: movement, AI, missile resolution, duty cycle
sensor.py           — Radar detection, contact classification, ECM
renderer.py         — All pygame rendering (tiles, units, contacts, HUD)
ui.py               — pygame-gui panels, buttons, event routing
geo.py              — Web-Mercator projection, haversine, bearing
map_tiles.py        — Async CARTO tile fetcher with disk cache
constants.py        — Shared constants
airbases.json       — Default Blue deployment (airbases + parked aircraft)
units.json          — Platform override database (optional)
weapons.json        — Weapon override database (optional)
map_cache_en/       — Cached CARTO Voyager tile images (auto-created)
```

---

## AI Behaviour

Both sides use the same autonomous engagement logic:

- Ground units have **auto-engage on by default**; aircraft do not.
- The AI selects the highest-range available weapon that can reach a contact within 90% of its maximum range.
- Missile cooldown: **45 seconds** between shots per unit. Gun cooldown: **8 seconds**.
- ECM jamming is automatically activated when the unit has an `ecm_rating > 0` and enemies are detected.
- ROE is respected: **TIGHT** requires a CONFIRMED contact before firing; **FREE** allows engagement at PROBABLE; **HOLD** disables all AI fire.

---

## Map Tiles

Map tiles are served by [CARTO Voyager](https://carto.com/basemaps/) (OpenStreetMap data with English place-name labels) and cached to `map_cache_en/` on disk. The in-memory LRU cache holds up to 512 tile surfaces. Subdomains `a`–`d` are rotated to stay within rate limits. Please respect CARTO's [tile usage policy](https://carto.com/legal/).
