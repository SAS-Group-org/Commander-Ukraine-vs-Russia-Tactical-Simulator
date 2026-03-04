
<img width="2816" height="1536" alt="Commander-Ukraine-vs-Russia-Tactical-Simulator" src="https://github.com/user-attachments/assets/5f936f08-11fb-4b8c-ad54-4e905a412191" />

# Commander: Ukraine vs Russia — Tactical Simulator
### A real-time air and ground combat wargame set in the Donbas theatre

---

## Requirements

| Dependency | Version |
|---|---|
| Python | 3.11 or newer |
| pygame | 2.x |
| pygame_gui | 0.6.x |
| requests | any recent |

Install dependencies:

```bash
pip install pygame pygame_gui requests
```

---

## File Structure

```
project/
├── main.py               # Entry point — run this
├── simulation.py         # Core simulation engine
├── scenario.py           # Data models, unit logic, save/load
├── sensor.py             # Multi-spectrum sensor model
├── renderer.py           # Map and unit drawing
├── ui.py                 # Bottom panel UI
├── geo.py                # Geo-math and terrain masking
├── map_tiles.py          # Async OSM tile downloader
├── constants.py          # Shared tuning values
├── weapons.json          # Weapon definitions database
├── units.json            # Platform definitions database
├── assets/
│   ├── blue_jet.png
│   └── red_jet.png
└── data/
    └── scenarios/        # Auto-created on first launch
```

---

## Running the Game

```bash
python main.py
```

On first launch the game will:
1. Generate a fresh Red order-of-battle across Donbas, Luhansk, and Crimea and write it to `data/scenarios/ukraine_russia.json`.
2. Begin downloading map tiles from CARTO in the background into `map_cache_en/`. The map will fill in over the first minute of play — this only happens once; tiles are cached to disk permanently.
3. Open in **Setup Mode** so you can deploy Blue forces before the simulation starts.

The window is fully resizable at any time.

---

## Game Modes

The game has two modes that you switch between at any time.

### Setup Mode
Deploy and configure your Blue force before the battle begins, or call in reinforcements during a pause mid-game. The simulation is frozen while in this mode.

### Combat Mode
The simulation runs in real time (or at compressed speed). You issue orders to individual units by selecting them on the map.

---

## Setup Mode — Deploying Blue Forces

When the game opens you are in Setup Mode. The bottom panel shows your available Blue roster on the left.

### Placing Units

1. **Browse the roster** in the left column. Units are grouped by category (AWACS, Fixed-Wing, Rotary, Air Defense, Armor, etc.) and show their fleet count.
2. **Click a unit type** in the roster to select it for placement. Enter a quantity in the number field if you want to place several at once.
3. **Click a location on the map** to place the unit there.
   - Aircraft and helicopters must be placed within 100 km of an existing Blue airbase. They will snap to the nearest one automatically.
   - Ground units cannot be placed in water.
4. Repeat until you are satisfied with your deployment.

> **Tip:** Click **Auto-Deploy** to have the game automatically distribute a full Blue force across seven historical clusters — Kyiv, Lviv, Starokostiantyniv, Zhytomyr, Dnipro, Kherson, and Odesa.

### Managing Placed Units

| Action | How |
|---|---|
| Remove a single unit | Right-click it on the map, or select it then click **Remove Selected** |
| Remove all Blue units | Click **Clear All Blue** |
| Save your deployment | Click **Save Deployment** — exports a `.json` file you can reload later |
| Load a saved deployment | Click **Load Deployment** — adds those units to the current scenario |

### Starting the Simulation

Click **Start Simulation** in the bottom-right corner of the Setup panel. You can return to Setup Mode at any time during combat by clicking **Reinforce** (pauses the simulation).

---

## Combat Mode — Controls

### Map Navigation

| Action | Control |
|---|---|
| Pan the map | Click and drag (left button on empty space) |
| Zoom in / out | Scroll wheel, or scroll over the map |
| Zoom range | Levels 4 (world) through 12 (city-level) |

### Time Controls

The time speed buttons are in the bottom panel. You can also use keyboard shortcuts:

| Key | Action |
|---|---|
| `Space` | Pause / Resume |
| `1` | 1× real time |
| `2` | 15× compression |
| `3` | 60× compression |
| `4` | 300× compression |
| `5` | Pause |

### Selecting Units

- **Left-click** a unit on the map to select it. The bottom panel will update to show that unit's status, loadout, and controls.
- **Left-click** empty map space to deselect.
- **Escape** deselects and cancels any pending action.

Only Blue units can be selected and commanded. Red units appear on the map only if they are within your sensor coverage — see the **Sensor Model** section below.

---

## Commanding Selected Units

### Waypoints (All Unit Types)

- **Right-click** the map to add a waypoint. The unit will proceed through waypoints in order.
- **Delete key** clears all waypoints for the selected unit and removes it from any formation.

### Aircraft and Helicopter Controls

The panel shows dedicated controls when an aircraft is selected.

#### Mission Type
Click **Mission** to cycle through mission types:
- **CAP** — Combat Air Patrol. The aircraft flies a racetrack orbit at the assigned area and automatically intercepts hostiles that enter sensor range.
- **STRIKE** — The aircraft navigates to the mission area and engages ground targets.
- **SEAD** — Suppression of Enemy Air Defenses. The aircraft prioritises radar-emitting SAM systems and airbases.

#### Loadout
Click **Loadout** to cycle through weapon configurations:
- **DEFAULT** — Platform's standard mixed load.
- **A2A** — Maximises air-to-air missiles.
- **A2G** — Maximises ground-attack weapons with a self-defence missile.
- **SEAD** — Maximises anti-radiation missiles (ARM) if available; falls back to A2G otherwise.

#### Launching Parked Aircraft
Aircraft start on the ground in **READY** state. Select a parked aircraft from the **Parked Aircraft** list in the combat panel, configure its mission type and loadout, then click **Launch**.

#### Altitude
Use the **+5000 / +1000 / +500** and **−5000 / −1000 / −500** buttons to set the target altitude in feet. The aircraft will climb or descend to the new altitude gradually.

#### RTB
Aircraft return to base automatically when fuel drops below the mission RTB threshold, or when they sustain critical damage. They cycle through a rearm/refuel timer on the ground and return to **READY** state automatically.

### Engaging Targets Manually

1. **Select** the unit you want to fire from.
2. **Select a weapon** in the loadout panel (it will highlight). Estimated Pk % is shown.
3. **Right-click** an enemy contact on the map to fire. The game validates range, domain (air vs. ground), and weapon availability before launching.
4. To fire without a pre-selected weapon, right-clicking a contact will have the unit automatically choose its best available weapon.

**Salvo Mode** (top of the weapon list) controls how many rounds are fired per engagement order:
- **1 / 2 / 4** — fixed salvo size.
- **SLS** — Shoot-Look-Shoot. Fires one round and waits to assess the result before firing again.

### Electronic Warfare Controls

When an aircraft or SAM is selected, the following toggles appear in the panel:

| Button | Effect |
|---|---|
| **Auto-Engage** | Unit will autonomously engage contacts within range (on/off) |
| **ROE** | Cycles through FREE → TIGHT → HOLD. TIGHT requires a CONFIRMED contact; HOLD prevents all firing |
| **ECM** | Toggles active jamming. Degrades enemy radar detection range but increases your own radar signature |
| **Radar** | Toggles radar emission. Turning radar off makes you invisible to ESM sensors but limits your own detection range to IR/optical only |
| **IFF** | Toggles IFF transponder. Turning it off makes you harder to classify but also prevents friendly units from recognising you |

---

## Display Toggles

The combat panel includes three view toggles:

| Button | Effect |
|---|---|
| **Air Labels** | Show/hide callsign labels for air units |
| **Gnd Labels** | Show/hide callsign labels for ground units |
| **Radar Rings** | Show/hide the teal radar detection rings around units |

**Fog of War** toggle: shows or hides the true positions of all Red units regardless of sensor coverage. Useful for learning the scenario layout, but disables any meaningful challenge.

---

## The Sensor Model

Detection is the core of the game. You only see what your sensors can see.

### Contact Classification

Enemy units appear as contact symbols, not unit icons, until your sensors resolve them:

| Symbol | Colour | Meaning |
|---|---|---|
| Small blip | Grey | **FAINT** — Something is there but type is unknown |
| Blip + type | Amber | **PROBABLE** — Unit type identified, side uncertain |
| Full icon | Red | **CONFIRMED** — Full resolution, side and type known |

Contacts fade and lose accuracy when the detecting sensor moves away or is destroyed. A stale track will degrade back to FAINT and eventually disappear after 30 seconds without a refresh.

### Sensor Types

Each platform carries multiple sensor types, all operating simultaneously:

| Sensor | Range | What it detects |
|---|---|---|
| **Active Radar** | Platform-dependent (shown as teal ring) | Any target within radar horizon; range reduced by target RCS and enemy ECM |
| **ESM (Passive)** | ~1.5× radar range | Targets that are actively emitting radar; yields PROBABLE classification only |
| **IR / Optical** | Short range (~8–40 km depending on platform) | Any target; yields CONFIRMED classification but requires close proximity |

### Terrain Masking

Hills and mountains block radar line-of-sight. Low-flying aircraft and ground units in the Carpathians may be invisible to sensors on flat ground even within nominal radar range. AWACS aircraft at altitude have much longer effective radar horizons because they see over terrain.

### Datalink Network

Units do not share contacts automatically. A unit must have line-of-sight to an AWACS or airbase to join the Blue datalink network. When connected, it receives the best available track from any other networked sensor — with the lowest positional error winning the merge. Units that lose datalink (terrain masking, AWACS destroyed) fall back to their own local sensor picture only.

---

## Damage Model

Units sustain graduated damage rather than dying instantly.

| Damage State | HP Range | Effect |
|---|---|---|
| OK | 75–100% | No degradation |
| LIGHT | 50–75% | 20% performance reduction |
| MODERATE | 25–50% | 40% reduction; aircraft trigger emergency RTB |
| HEAVY | 0–25% | 60% reduction; aircraft trigger emergency RTB |
| KILLED | 0% | Unit removed |

In addition to the HP pool, three **system slots** can be independently destroyed:

- **Radar** — destroyed radar forces the unit to radar-off; ESM and IR still function.
- **Mobility** — ground units stop moving; aircraft suffer a severe performance penalty.
- **Weapons** — unit can no longer fire.

Hits have a chance to start a **fire** on the target. Burning units take continuous damage; crew effectiveness (affected by drunkness and corruption ratings on Red units) determines how quickly they extinguish it.

---

## Crew Quality (Red Forces)

Each Red unit is assigned randomised **Drunkness** and **Corruption** ratings at scenario generation. These are not cosmetic — they directly reduce sensor range, weapon accuracy (Pk), and fire-fighting effectiveness:

| Rating | Drunkness Label | Corruption Label |
|---|---|---|
| 1 | Sober | Clean |
| 2 | Tipsy | Grass Eater |
| 3 | Intoxicated | Dirty |
| 4 | Wasted | Meat Eater |
| 5 | Yeltsin | Shoigu |

A unit with both ratings at 5 suffers a ~64% penalty across all performance metrics. Click a Red unit (with Fog of War on) to see its stats in the info panel.

---

## Saving and Loading

| Action | Control |
|---|---|
| Save current game state | `Ctrl+S` during Combat Mode |
| Load save (on next launch) | The game auto-loads `data/scenarios/ukraine_russia_save.json` if present |
| Save Blue deployment only | Setup Mode → **Save Deployment** button |
| Load a deployment | Setup Mode → **Load Deployment** button |
| Restart scenario | Combat Mode → **Restart** button (regenerates fresh Red order-of-battle) |

---

## Scenario Overview

**Operation East Wind — Donbas 2024**

Red forces (Russia) are entrenched across Luhansk Oblast, Donetsk Oblast, Zaporizhia, and Crimea, anchored around clusters at Donetsk City, Mariupol, Horlivka, Luhansk City, Sevastopol, and several air bases with fighter and AWACS coverage.

Blue forces (Ukraine and allies) deploy from western and central Ukraine — primarily Kyiv, Lviv, Zhytomyr, Odesa, Dnipro, and Kherson — and must project combat power eastward across a contested airspace defended by S-400, Buk-M2, and Tor-M1 batteries.

The Red order-of-battle is **randomly regenerated** every time you click Restart, so no two games are identical.

---

## Tips

- **Deploy airbases before aircraft.** Aircraft snap to the nearest Blue airbase on placement. If no base is within 100 km, the aircraft cannot be placed.
- **Use AWACS first.** The E-3G Sentry dramatically expands the Blue datalink picture. Place it behind friendly lines at high altitude to maximise horizon coverage.
- **Turn radar off on strike packages** until they are close to the target area. ESM-equipped Red SAMs will pick up your radar emissions from much further than their own radar can see you.
- **SEAD before STRIKE.** Set SEAD aircraft to launch first, wait for SAM radars to go offline (contact turns FAINT or disappears from the board), then launch the strike package.
- **Watch the event log** (bottom-right panel during combat) for splash confirmations, evasion reports, and system damage alerts.
- **At 300× time compression** missiles still move at correct simulated speeds but the map update rate is reduced. Pause to issue precision orders during a busy engagement.
