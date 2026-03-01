# sensor.py — physics-based sensor model
#
# Replaces the old binary radius check with:
#   1. Radar horizon (earth curvature limits line-of-sight)
#   2. Radar range equation (RCS-adjusted detection range)
#   3. ECM / jamming reduction
#   4. Contact classification ladder: FAINT → PROBABLE → CONFIRMED
#   5. Accounts for Damage State limiting the performance of the sensor

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from scenario import PlatformDef, Unit


# ── Physical constants ────────────────────────────────────────────────────────

# Nominal RCS (m²) used as the reference when computing radar_range_km entries
# in the platform database.  Every platform's radar_range_km is calibrated
# against a target with this cross-section.
RCS_REFERENCE_M2: float = 5.0

# Radio/optical horizon coefficient (km / √m)
# R_horizon = HORIZON_COEFF × (√h_sensor + √h_target)  [km, metres input]
HORIZON_COEFF: float = 4.12

# Minimum sensor height above ground (metres).  Prevents √0 on ground units.
GROUND_SENSOR_HEIGHT_M: float = 5.0

# ECM effectiveness scalar applied to ecm_rating when reducing detection range.
# effective_range = nominal_range × (1 − ECM_SCALE × ecm_rating)
ECM_SCALE: float = 0.60

# Contact classification range bands as fractions of effective detection range:
#   dist > FAINT_BAND    × R_eff  →  NONE      (below detection threshold)
#   dist > PROBABLE_BAND × R_eff  →  FAINT     (blip on scope, no detail)
#   dist > CONFIRM_BAND  × R_eff  →  PROBABLE  (type resolved, side unsure)
#   dist ≤ CONFIRM_BAND  × R_eff  →  CONFIRMED (full resolution)
FAINT_BAND:    float = 1.00
PROBABLE_BAND: float = 0.75
CONFIRM_BAND:  float = 0.50

# Seconds of game-time before a non-refreshed contact is dropped as "lost track"
CONTACT_TIMEOUT_S: float = 30.0


# ── Contact dataclass ─────────────────────────────────────────────────────────

@dataclass
class Contact:
    uid:            str
    lat:            float
    lon:            float
    altitude_ft:    float
    classification: str                # FAINT | PROBABLE | CONFIRMED
    unit_type:      Optional[str]      # resolved at PROBABLE+
    side:           Optional[str]      # resolved at CONFIRMED
    last_update:    float              # game clock seconds

    @property
    def label(self) -> str:
        if self.classification == "CONFIRMED":
            return f"[{self.side}] {self.unit_type}"
        if self.classification == "PROBABLE":
            return f"? {self.unit_type}"
        return "?"


# ── Core detection function ───────────────────────────────────────────────────

def classify_detection(sensor_unit: "Unit",
                       target: "Unit",
                       dist_km: float) -> str:
    """Return the contact classification a sensor achieves against *target*."""

    # ── 1. Radar horizon (earth curvature) ────────────────────────────────────
    sensor_alt_m = max(sensor_unit.altitude_ft * 0.3048, GROUND_SENSOR_HEIGHT_M)
    target_alt_m = max(target.altitude_ft * 0.3048, GROUND_SENSOR_HEIGHT_M)
    horizon_km   = HORIZON_COEFF * (math.sqrt(sensor_alt_m)
                                    + math.sqrt(target_alt_m))
    if dist_km > horizon_km:
        return "NONE"

    # ── 2. RCS-adjusted detection range & Damage Penalty ──────────────────────
    # Radar range equation: R ∝ RCS^(1/4). 
    # Sensor performance degrades linearly with physical damage state.
    rcs_ratio  = max(target.platform.rcs_m2, 0.01) / RCS_REFERENCE_M2
    R_rcs      = (sensor_unit.platform.radar_range_km * sensor_unit.performance_mult) * (rcs_ratio ** 0.25)

    # ── 3. ECM reduction ─────────────────────────────────────────────────────
    ecm_penalty = target.platform.ecm_rating * ECM_SCALE
    R_effective = R_rcs * max(0.0, 1.0 - ecm_penalty)

    if R_effective <= 0.0 or dist_km > R_effective * FAINT_BAND:
        return "NONE"

    # ── 4. Classification bands ───────────────────────────────────────────────
    fraction = dist_km / R_effective        # 0 = point-blank, 1 = detection edge

    if fraction > PROBABLE_BAND:
        return "FAINT"
    if fraction > CONFIRM_BAND:
        return "PROBABLE"
    return "CONFIRMED"


# ── Multi-sensor scan ─────────────────────────────────────────────────────────

def update_contacts(detecting_units: list["Unit"],
                    target_units: list["Unit"],
                    existing_contacts: dict[str, Contact],
                    game_time: float) -> dict[str, Contact]:
    from geo import haversine

    _RANK = {"NONE": 0, "FAINT": 1, "PROBABLE": 2, "CONFIRMED": 3}
    refreshed: set[str] = set()

    for target in target_units:
        if not target.alive:
            continue

        best_cls  = "NONE"
        best_rank = 0

        for sensor_unit in detecting_units:
            if not sensor_unit.alive:
                continue

            dist = haversine(sensor_unit.lat, sensor_unit.lon,
                             target.lat, target.lon)
            
            # Pass the full Unit object so damage state can be evaluated
            cls  = classify_detection(sensor_unit, target, dist)
            
            rank = _RANK[cls]
            if rank > best_rank:
                best_rank = rank
                best_cls  = cls

        if best_cls == "NONE":
            continue   # not detected this tick — let it age out

        refreshed.add(target.uid)

        unit_type = target.platform.unit_type if best_rank >= 2 else None
        side      = target.side               if best_rank >= 3 else None

        contact = existing_contacts.get(target.uid)
        if contact is None:
            existing_contacts[target.uid] = Contact(
                uid            = target.uid,
                lat            = target.lat,
                lon            = target.lon,
                altitude_ft    = target.altitude_ft,
                classification = best_cls,
                unit_type      = unit_type,
                side           = side,
                last_update    = game_time,
            )
        else:
            contact.lat            = target.lat
            contact.lon            = target.lon
            contact.altitude_ft    = target.altitude_ft
            contact.last_update    = game_time
            if best_rank > _RANK[contact.classification]:
                contact.classification = best_cls
                contact.unit_type      = unit_type
                contact.side           = side

    # Expire stale contacts
    expired = [uid for uid, c in existing_contacts.items()
               if uid not in refreshed
               and (game_time - c.last_update) > CONTACT_TIMEOUT_S]
    for uid in expired:
        del existing_contacts[uid]

    return existing_contacts