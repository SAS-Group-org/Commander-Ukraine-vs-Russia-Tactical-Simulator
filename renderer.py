# renderer.py — Data-Oriented Pygame rendering with pre-baked 360-degree sprite arrays

from __future__ import annotations

import math
import os
from typing import Optional
from collections import OrderedDict

import pygame

from constants import (
    TILE_SIZE,
    BLUE_UNIT_COLOR, RED_UNIT_COLOR, SELECTED_COLOR,
    WAYPOINT_COLOR, ROUTE_LINE_COLOR,
    RADAR_RING_COLOR,
    MISSILE_BLUE_COLOR, MISSILE_RED_COLOR, TRAIL_COLOR,
    CONTACT_FAINT_COLOR, CONTACT_PROBABLE_COLOR, CONTACT_CONFIRM_COLOR,
)
from geo import lat_lon_to_pixel, world_to_screen
from map_tiles import get_tile
from scenario import Missile, Unit
from sensor import Contact

class Renderer:
    def __init__(self, surface: pygame.Surface):
        self.surface     = surface
        
        # RENDERING OPTIMIZATIONS: Bounded Caches
        self._text_cache: OrderedDict[tuple, pygame.Surface] = OrderedDict()
        self._explosion_cache: OrderedDict[tuple, pygame.Surface] = OrderedDict()
        self._radar_radius_cache: OrderedDict[tuple, int] = OrderedDict()
        self._circle_cache: OrderedDict[tuple, pygame.Surface] = OrderedDict()
        
        # PRE-BAKED SPRITES: O(1) lookup arrays for 360-degree rotations
        # Structure: dict[image_path_or_utype, dict[color, list[pygame.Surface]]]
        self._baked_sprites: dict[str, dict[tuple, list[pygame.Surface]]] = {}
        
        # GEO CACHE: FIFO queue to prevent map screen stutter
        self._geo_cache: dict[tuple[float, float, int], tuple[float, float]] = {}
        self._loc_pixel_cache: dict[int, list[tuple[float, float]]] = {}
        
        self._missing_tile_surf: Optional[pygame.Surface] = None
        
        # Safe maximums
        self._MAX_TEXT_CACHE = 1024
        self._MAX_EXPLOSION_CACHE = 512
        self._MAX_RADAR_CACHE = 2048
        self._MAX_CIRCLE_CACHE = 1024
        self._MAX_GEO_CACHE = 50000 
        
        self._font_sm    = pygame.font.SysFont(None, 15)
        self._font_med   = pygame.font.SysFont(None, 18)

    def _fast_world_to_screen(self, lat: float, lon: float, cam_px: float, cam_py: float, zoom: int, win_w: int, map_h: int, is_static: bool = False) -> tuple[float, float]:
        if is_static:
            key = (lat, lon, zoom)
            if key in self._geo_cache:
                ax, ay = self._geo_cache[key]
            else:
                ax, ay = lat_lon_to_pixel(lat, lon, zoom)
                if len(self._geo_cache) >= self._MAX_GEO_CACHE:
                    self._geo_cache.pop(next(iter(self._geo_cache)))
                self._geo_cache[key] = (ax, ay)
        else:
            ax, ay = lat_lon_to_pixel(lat, lon, zoom)
            
        return ax - cam_px + win_w / 2, ay - cam_py + map_h / 2

    def _get_missing_tile(self) -> pygame.Surface:
        if not self._missing_tile_surf:
            self._missing_tile_surf = pygame.Surface((TILE_SIZE, TILE_SIZE)).convert()
            self._missing_tile_surf.fill((28, 38, 48))
            pygame.draw.rect(self._missing_tile_surf, (40, 54, 68), (0, 0, TILE_SIZE, TILE_SIZE), 1)
        return self._missing_tile_surf

    def _get_text_surface(self, text: str, color: tuple[int, int, int], font: pygame.font.Font) -> pygame.Surface:
        key = (text, color, font)
        if key in self._text_cache:
            self._text_cache.move_to_end(key)
            return self._text_cache[key]
            
        if len(self._text_cache) >= self._MAX_TEXT_CACHE:
            self._text_cache.popitem(last=False)
            
        surf = font.render(text, True, color)
        self._text_cache[key] = surf
        return surf

    def _get_circle_surface(self, radius: int, color: tuple, width: int = 0) -> pygame.Surface:
        key = (radius, color, width)
        if key in self._circle_cache:
            self._circle_cache.move_to_end(key)
            return self._circle_cache[key]
        
        size = radius * 2 + 2
        surf = pygame.Surface((size, size), pygame.SRCALPHA).convert_alpha()
        pygame.draw.circle(surf, color, (radius + 1, radius + 1), radius, width)
        
        if len(self._circle_cache) >= self._MAX_CIRCLE_CACHE:
            self._circle_cache.popitem(last=False)
        self._circle_cache[key] = surf
        return surf

    def _get_explosion_surface(self, radius: int, alpha: int) -> pygame.Surface:
        key = (radius, alpha)
        if key in self._explosion_cache:
            self._explosion_cache.move_to_end(key)
            return self._explosion_cache[key]
            
        if len(self._explosion_cache) >= self._MAX_EXPLOSION_CACHE:
            self._explosion_cache.popitem(last=False)
            
        s = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA).convert_alpha()
        pygame.draw.circle(s, (255, 100, 50, alpha), (radius, radius), radius, max(1, radius // 4))
        self._explosion_cache[key] = s
        return s

    def _get_aou_surface(self, radius: int, cls: str, color: tuple) -> pygame.Surface:
        key = ("AOU", radius, cls, color)
        if key in self._explosion_cache:
            self._explosion_cache.move_to_end(key)
            return self._explosion_cache[key]
            
        aou_surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA).convert_alpha()
        fill_alpha = 40 if cls == "FAINT" else 15
        edge_alpha = 90 if cls == "FAINT" else 50
        
        pygame.draw.circle(aou_surf, (*color[:3], fill_alpha), (radius, radius), radius)
        pygame.draw.circle(aou_surf, (*color[:3], edge_alpha), (radius, radius), radius, 1)
        if radius > 8:
            pygame.draw.line(aou_surf, (*color[:3], edge_alpha), (radius, radius - 8), (radius, radius + 8), 1)
            pygame.draw.line(aou_surf, (*color[:3], edge_alpha), (radius - 8, radius), (radius + 8, radius), 1)
            
        if len(self._explosion_cache) >= self._MAX_EXPLOSION_CACHE:
            self._explosion_cache.popitem(last=False)
        self._explosion_cache[key] = aou_surf
        return aou_surf

    def _generate_base_symbol(self, utype: str, color: tuple) -> pygame.Surface:
        """Generates the un-rotated base drawing for a symbol."""
        surf = pygame.Surface((32, 32), pygame.SRCALPHA).convert_alpha()
        cx, cy = 16, 16
        
        if utype == "tank":
            hw, hh = 10, 7
            pygame.draw.rect(surf, color, (cx-hw, cy-hh, hw*2, hh*2))
            pygame.draw.rect(surf, (0,0,0), (cx-hw, cy-hh, hw*2, hh*2), 1)
            pygame.draw.line(surf, color, (cx, cy), (cx, cy - 14), 2)
        elif utype == "ifv":
            hw, hh = 9, 6
            pygame.draw.rect(surf, color, (cx-hw, cy-hh, hw*2, hh*2), 2)
            pygame.draw.circle(surf, color, (cx, cy), 4)
        elif utype == "apc":
            hw, hh = 9, 6
            pygame.draw.rect(surf, color, (cx-hw, cy-hh, hw*2, hh*2), 2)
        elif utype == "recon":
            pts = [(cx, cy-10), (cx+9, cy), (cx, cy+10), (cx-9, cy)]
            pygame.draw.polygon(surf, color, pts, 2)
        elif utype == "tank_destroyer":
            pts = [(cx, cy-12), (cx+8, cy+6), (cx-8, cy+6)]
            pygame.draw.polygon(surf, color, pts, 2)
            pygame.draw.line(surf, color, (cx, cy), (cx, cy - 14), 2)
        elif utype == "sam":
            hw, hh = 10, 7
            pygame.draw.rect(surf, color, (cx-hw, cy-hh, hw*2, hh*2), 2)
            rect = pygame.Rect(cx-5, cy+hh-5, 10, 10)
            pygame.draw.arc(surf, color, rect, 0, math.pi, 2)
        elif utype == "airbase":
            hw, hh = 12, 6
            pygame.draw.rect(surf, color, (cx-hw, cy-hh, hw*2, hh*2), 2)
            pygame.draw.line(surf, color, (cx-hw, cy), (cx+hw, cy), 1)
        elif utype == "artillery":
            hw, hh = 10, 7
            pygame.draw.rect(surf, color, (cx-hw, cy-hh, hw*2, hh*2), 2)
            pygame.draw.circle(surf, color, (cx, cy), 3)
        elif utype == "faint_contact":
            pts = [(cx, cy - 6), (cx + 6, cy), (cx, cy + 6), (cx - 6, cy)]
            pygame.draw.polygon(surf, color, pts, 1)
        elif utype == "probable_air":
            pts = [(cx, cy - 9), (cx + 8, cy + 5), (cx - 8, cy + 5)]
            pygame.draw.polygon(surf, color, pts, 2)
        elif utype == "probable_gnd":
            pts = [(cx, cy - 9), (cx + 9, cy), (cx, cy + 9), (cx - 9, cy)]
            pygame.draw.polygon(surf, color, pts, 2)
        else: 
            pts = [(cx, cy-12), (cx+8, cy+6), (cx+2, cy+4), (cx+2, cy+12), (cx-2, cy+12), (cx-2, cy+4), (cx-8, cy+6)]
            pygame.draw.polygon(surf, color, pts, 2)
            
        return surf

    def _get_baked_sprite(self, key: str, color: tuple, angle: int, is_image: bool = False) -> pygame.Surface:
        """O(1) lookup for pre-baked 360-degree rotations."""
        angle = int(angle) % 360
        
        if key not in self._baked_sprites:
            self._baked_sprites[key] = {}
            
        if color not in self._baked_sprites[key]:
            # Generate the 360 array on first request
            self._baked_sprites[key][color] = [None] * 360
            
            if is_image and os.path.exists(key):
                try:
                    img = pygame.image.load(key).convert_alpha()
                    base_surf = pygame.transform.smoothscale(img, (32, 32))
                except pygame.error:
                    base_surf = self._generate_base_symbol("unknown", color)
            else:
                base_surf = self._generate_base_symbol(key, color)
                
            for a in range(360):
                self._baked_sprites[key][color][a] = pygame.transform.rotate(base_surf, -a)
                
        return self._baked_sprites[key][color][angle]

    def update_surface(self, surface: pygame.Surface) -> None:
        self.surface = surface

    def draw_frame(self,
                   cam_px: float, cam_py: float, zoom: int,
                   units: list[Unit],
                   missiles: list[Missile],
                   win_w: int, map_h: int,
                   blue_contacts: dict[str, Contact] | None = None,
                   explosions: list = None,
                   placing_type: str | None = None,
                   placing_remaining: int = 0,
                   mouse_pos: tuple[int, int] | None = None,
                   show_all_enemies: bool = False,
                   show_air_labels: bool = True,
                   show_ground_labels: bool = True,
                   show_radar_rings: bool = True,
                   package_waypoints: list[tuple[float, float, float]] | None = None,
                   loc_points: list[tuple[float, float]] | None = None,
                   air_label_zoom_threshold: int = 10,
                   gnd_label_zoom_threshold: int = 10) -> None:
        
        contacts = blue_contacts or {}
        explosions = explosions or []
        package_waypoints = package_waypoints or []
        
        tile_blits = []
        aou_blits = []
        radar_blits = []
        misc_blits = [] 
        sprite_blits = []
        text_blits = []
        
        self.surface.fill((18, 26, 34))
        
        # 1. Background / Geography
        self._queue_tiles(cam_px, cam_py, zoom, win_w, map_h, tile_blits)
        if tile_blits: self.surface.blits(tile_blits)
        self._draw_loc(cam_px, cam_py, zoom, win_w, map_h, loc_points)
        
        # 2. Geometry Caching & Queueing
        if show_radar_rings:
            self._queue_radar_rings(cam_px, cam_py, zoom, units, win_w, map_h, show_all_enemies, radar_blits)
        self._queue_routes(cam_px, cam_py, zoom, units, win_w, map_h, show_all_enemies, misc_blits)
        if placing_type == "STRIKE PACKAGE TARGET" and (package_waypoints or mouse_pos):
            self._draw_pending_route(cam_px, cam_py, zoom, win_w, map_h, package_waypoints, mouse_pos, misc_blits)
        self._queue_missiles(cam_px, cam_py, zoom, missiles, win_w, map_h, misc_blits)
        self._queue_explosions(cam_px, cam_py, zoom, explosions, win_w, map_h, aou_blits)
        
        # 3. Entity Caching & Queueing
        self._queue_contacts(cam_px, cam_py, zoom, contacts, units, win_w, map_h, show_all_enemies, show_air_labels, show_ground_labels, air_label_zoom_threshold, gnd_label_zoom_threshold, aou_blits, sprite_blits, text_blits)
        self._queue_units(cam_px, cam_py, zoom, units, win_w, map_h, show_all_enemies, show_air_labels, show_ground_labels, air_label_zoom_threshold, gnd_label_zoom_threshold, misc_blits, sprite_blits, text_blits)
        
        # 4. Execute Hardware Batch Blitting (Z-Ordered)
        if aou_blits:    self.surface.blits(aou_blits)
        if radar_blits:  self.surface.blits(radar_blits)
        if misc_blits:   self.surface.blits(misc_blits)
        if sprite_blits: self.surface.blits(sprite_blits)
        if text_blits:   self.surface.blits(text_blits)
        
        if placing_type and mouse_pos and mouse_pos[1] < map_h:
            self._draw_placement_cursor(mouse_pos, placing_type, placing_remaining)

    def _draw_loc(self, cam_px, cam_py, zoom, win_w, map_h, loc_points) -> None:
        if not loc_points: return
        
        if zoom not in self._loc_pixel_cache:
            self._loc_pixel_cache[zoom] = [lat_lon_to_pixel(lat, lon, zoom) for lat, lon in loc_points]
            
        pts = []
        ox = win_w / 2 - cam_px
        oy = map_h / 2 - cam_py
        
        for ax, ay in self._loc_pixel_cache[zoom]:
            pts.append((ax + ox, ay + oy))
            
        if len(pts) > 1:
            pygame.draw.lines(self.surface, (200, 40, 40), False, pts, 4)

    def _queue_tiles(self, cam_px, cam_py, zoom, win_w, map_h, tile_blits: list) -> None:
        tl_x = cam_px - win_w / 2
        tl_y = cam_py - map_h / 2
        sx = int(tl_x // TILE_SIZE)
        sy = int(tl_y // TILE_SIZE)
        ex = sx + (win_w  // TILE_SIZE) + 2
        ey = sy + (map_h  // TILE_SIZE) + 2

        for tx in range(sx, ex):
            for ty in range(sy, ey):
                bx = int(tx * TILE_SIZE - tl_x)
                by = int(ty * TILE_SIZE - tl_y)
                surf = get_tile(zoom, tx, ty)
                tile_blits.append((surf if surf else self._get_missing_tile(), (bx, by)))

    def _queue_radar_rings(self, cam_px, cam_py, zoom, units, win_w, map_h, show_all_enemies: bool, radar_blits: list) -> None:
        for unit in units:
            if not unit.alive or unit.duty_state != "ACTIVE": continue
            if unit.side == "Red" and not show_all_enemies: continue         
            if not getattr(unit, 'search_radar_active', True): continue
            
            radar_type_lower = unit.platform.radar_type.lower()
            is_non_radar = any(kw in radar_type_lower for kw in ("none", "optical", "thermal", "hunter-killer", "laser", "wescam", "adad"))
            if is_non_radar and not unit.selected: continue
                
            eff_range = unit.platform.radar_range_km * unit.performance_mult
            if eff_range <= 0: continue
                
            sx, sy = self._fast_world_to_screen(unit.lat, unit.lon, cam_px, cam_py, zoom, win_w, map_h, is_static=False)

            cache_key = (round(unit.lat, 2), round(eff_range, 1), zoom)
            if cache_key in self._radar_radius_cache:
                self._radar_radius_cache.move_to_end(cache_key)
            else:
                if len(self._radar_radius_cache) >= self._MAX_RADAR_CACHE:
                    self._radar_radius_cache.popitem(last=False)
                lat2 = unit.lat + (eff_range / 111.32)
                _, py1 = lat_lon_to_pixel(unit.lat, unit.lon, zoom)
                _, py2 = lat_lon_to_pixel(lat2,     unit.lon, zoom)
                self._radar_radius_cache[cache_key] = int(abs(py1 - py2))
                
            radius = self._radar_radius_cache[cache_key]
            
            if 2 <= radius <= 4000:
                if (sx + radius >= 0 and sx - radius <= win_w and sy + radius >= 0 and sy - radius <= map_h):
                    color = RADAR_RING_COLOR if unit.side == "Blue" else (160, 50, 50)
                    
                    if unit.is_jamming:
                        jam_surf = self._get_circle_surface(8, (255, 255, 0), 1)
                        radar_blits.append((jam_surf, (int(sx) - 9, int(sy) - 9)))

                    if radius <= 400:
                        r_surf = self._get_circle_surface(radius, color, 1)
                        radar_blits.append((r_surf, (int(sx) - radius - 1, int(sy) - radius - 1)))
                    else:
                        pygame.draw.circle(self.surface, color, (int(sx), int(sy)), radius, 1)

    def _queue_routes(self, cam_px, cam_py, zoom, units, win_w, map_h, show_all_enemies: bool, misc_blits: list) -> None:
        for unit in units:
            if not unit.alive or not unit.waypoints or unit.duty_state != "ACTIVE": continue
            if unit.side == "Red" and not show_all_enemies: continue
                
            sx, sy = self._fast_world_to_screen(unit.lat, unit.lon, cam_px, cam_py, zoom, win_w, map_h, is_static=False)
            points = [(sx, sy)]
            
            wp_surf = self._get_circle_surface(3, WAYPOINT_COLOR, 0)
            
            for wp in unit.waypoints:
                wx, wy = self._fast_world_to_screen(wp[0], wp[1], cam_px, cam_py, zoom, win_w, map_h, is_static=True)
                points.append((wx, wy))
                
                if -10 < wx < win_w + 10 and -10 < wy < map_h + 10:
                    misc_blits.append((wp_surf, (int(wx) - 4, int(wy) - 4)))
            
            if len(points) > 1:
                pygame.draw.lines(self.surface, ROUTE_LINE_COLOR, False, points, 1)

    def _draw_pending_route(self, cam_px, cam_py, zoom, win_w, map_h, package_waypoints, mouse_pos, misc_blits: list) -> None:
        points = []
        wp_surf = self._get_circle_surface(4, (200, 200, 50), 1)
        
        for wp in package_waypoints:
            wx, wy = self._fast_world_to_screen(wp[0], wp[1], cam_px, cam_py, zoom, win_w, map_h, is_static=True)
            points.append((wx, wy))
            misc_blits.append((wp_surf, (int(wx) - 5, int(wy) - 5)))
            
            if wp[2] >= 0:
                alt_lbl = self._get_text_surface(f"{int(wp[2])}ft", (200, 200, 50), self._font_sm)
                self.surface.blit(alt_lbl, (int(wx) + 6, int(wy) - 10))

        if mouse_pos and mouse_pos[1] < map_h:
            points.append(mouse_pos)
            
        if len(points) > 1:
            pygame.draw.lines(self.surface, (200, 200, 50), False, points, 1)

    def _queue_missiles(self, cam_px, cam_py, zoom, missiles, win_w, map_h, misc_blits: list) -> None:
        ox = win_w / 2 - cam_px
        oy = map_h / 2 - cam_py
        
        for m in missiles:
            if not m.active: continue
            color = MISSILE_BLUE_COLOR if m.side == "Blue" else MISSILE_RED_COLOR
            points = []
            
            for tlat, tlon in m.trail:
                key = (tlat, tlon, zoom)
                if key in self._geo_cache:
                    ax, ay = self._geo_cache[key]
                else:
                    ax, ay = lat_lon_to_pixel(tlat, tlon, zoom)
                    if len(self._geo_cache) >= self._MAX_GEO_CACHE:
                        self._geo_cache.pop(next(iter(self._geo_cache)))
                    self._geo_cache[key] = (ax, ay)
                points.append((ax + ox, ay + oy))
            
            mx, my = self._fast_world_to_screen(m.lat, m.lon, cam_px, cam_py, zoom, win_w, map_h, is_static=False)
            points.append((mx, my))

            if len(points) > 1:
                pygame.draw.lines(self.surface, TRAIL_COLOR, False, points, 2)
                
            if -5 < mx < win_w + 5 and -5 < my < map_h + 5:
                msl_surf = self._get_circle_surface(3, color, 0)
                misc_blits.append((msl_surf, (int(mx) - 4, int(my) - 4)))

    def _queue_explosions(self, cam_px, cam_py, zoom, explosions, win_w, map_h, aou_blits: list) -> None:
        for exp in explosions:
            progress = exp.life / exp.max_life
            alpha = int(255 * (1.0 - progress))
            current_radius_km = exp.max_radius_km * (0.1 + 0.9 * progress)
            ex, ey = self._fast_world_to_screen(exp.lat, exp.lon, cam_px, cam_py, zoom, win_w, map_h, is_static=True)
            
            lat_edge = exp.lat + (current_radius_km / 111.32)
            _, py_center = lat_lon_to_pixel(exp.lat, exp.lon, zoom)
            _, py_edge = lat_lon_to_pixel(lat_edge, exp.lon, zoom)
            px_radius = max(2, int(abs(py_center - py_edge)))

            if (ex + px_radius >= 0 and ex - px_radius <= win_w and ey + px_radius >= 0 and ey - px_radius <= map_h):
                s = self._get_explosion_surface(px_radius, alpha)
                aou_blits.append((s, (int(ex) - px_radius, int(ey) - px_radius)))

    def _queue_units(self, cam_px, cam_py, zoom, units, win_w, map_h,
                   show_all_enemies: bool, show_air_labels: bool, show_ground_labels: bool,
                   air_label_zoom_threshold: int, gnd_label_zoom_threshold: int,
                   misc_blits: list, sprite_blits: list, text_blits: list) -> None:
        for unit in units:
            if not unit.alive or unit.duty_state != "ACTIVE": continue
            if unit.side == "Red" and not show_all_enemies: continue

            sx, sy = self._fast_world_to_screen(unit.lat, unit.lon, cam_px, cam_py, zoom, win_w, map_h, is_static=False)
            if not (-80 < sx < win_w + 80 and -80 < sy < map_h + 80): continue

            base_color = (BLUE_UNIT_COLOR if unit.side == "Blue" else RED_UNIT_COLOR)
            if unit.flash_frames > 0: base_color = (255, 255, 255)
            color = SELECTED_COLOR if unit.selected else base_color

            if unit.selected: 
                sel_surf = self._get_circle_surface(18, SELECTED_COLOR, 2)
                misc_blits.append((sel_surf, (int(sx) - 19, int(sy) - 19)))

            utype = unit.platform.unit_type
            
            # O(1) Pre-baked Sprite Lookup
            if unit.image_path:
                rotated = self._get_baked_sprite(unit.image_path, color, unit.heading, is_image=True)
            else:
                rotated = self._get_baked_sprite(utype, color, unit.heading, is_image=False)
            
            rect = rotated.get_rect(center=(int(sx), int(sy)))
            sprite_blits.append((rotated, rect.topleft))

            is_air = utype in ("fighter", "attacker", "helicopter", "awacs")
            show_label = (is_air and show_air_labels) or (not is_air and show_ground_labels)

            threshold = air_label_zoom_threshold if is_air else gnd_label_zoom_threshold
            if zoom < threshold and not unit.selected:
                show_label = False

            if show_label:
                det_tag = " ◆" if unit.is_detected and unit.side == "Blue" else ""
                label   = self._get_text_surface(f"{unit.callsign}{det_tag}", (0, 0, 0), self._font_sm)
                text_blits.append((label, (int(sx) + 13, int(sy) - 10)))
                
                type_label = self._get_text_surface(unit.platform.display_name, (0, 0, 0), self._font_sm)
                text_blits.append((type_label, (int(sx) + 13, int(sy) + 2)))


    def _queue_contacts(self, cam_px, cam_py, zoom,
                       contacts: dict, units: list,
                       win_w: int, map_h: int,
                       show_all_enemies: bool, show_air_labels: bool, show_ground_labels: bool,
                       air_label_zoom_threshold: int, gnd_label_zoom_threshold: int,
                       aou_blits: list, sprite_blits: list, text_blits: list) -> None:
        if show_all_enemies: return

        uid_to_unit = {u.uid: u for u in units if u.side == "Red"}

        for uid, contact in contacts.items():
            sx, sy = self._fast_world_to_screen(contact.est_lat, contact.est_lon, cam_px, cam_py, zoom, win_w, map_h, is_static=False)
            if not (-80 < sx < win_w + 80 and -80 < sy < map_h + 80): continue

            cls = contact.classification
            utype = contact.unit_type or "unknown"
            
            is_air = utype in ("fighter", "attacker", "helicopter", "awacs")
            show_label = (is_air and show_air_labels) or (not is_air and show_ground_labels)
            threshold = air_label_zoom_threshold if is_air else gnd_label_zoom_threshold
            
            track_color = CONTACT_CONFIRM_COLOR
            if contact.perceived_side == "UNKNOWN": track_color = (200, 200, 200) 
            elif contact.perceived_side == "Blue": track_color = BLUE_UNIT_COLOR

            if contact.pos_error_km > 0.5:
                lat2 = contact.est_lat + (contact.pos_error_km / 111.32)
                _, py1 = lat_lon_to_pixel(contact.est_lat, contact.est_lon, zoom)
                _, py2 = lat_lon_to_pixel(lat2, contact.est_lon, zoom)
                radius = int(abs(py1 - py2))
                
                if radius > 2:
                    if radius <= 800: 
                        aou_surf = self._get_aou_surface(radius, cls, track_color)
                        aou_blits.append((aou_surf, (int(sx) - radius, int(sy) - radius)))
                    else:
                        pygame.draw.rect(self.surface, track_color, (int(sx) - radius, int(sy) - radius, radius*2, radius*2), 1)

            if cls == "FAINT":
                surf = self._get_baked_sprite("faint_contact", CONTACT_FAINT_COLOR, 0)
                rect = surf.get_rect(center=(int(sx), int(sy)))
                sprite_blits.append((surf, rect.topleft))
                
                if zoom < threshold: show_label = False
                if show_label:
                    lbl = self._get_text_surface("?", CONTACT_FAINT_COLOR, self._font_sm)
                    text_blits.append((lbl, (int(sx) + 8, int(sy) - 6)))

            elif cls == "PROBABLE":
                key = f"probable_{'air' if is_air else 'gnd'}"
                surf = self._get_baked_sprite(key, track_color, 0)
                rect = surf.get_rect(center=(int(sx), int(sy)))
                sprite_blits.append((surf, rect.topleft))
                
                if zoom < threshold: show_label = False    
                if show_label:
                    lbl = self._get_text_surface(f"? {utype}", track_color, self._font_sm)
                    text_blits.append((lbl, (int(sx) + 12, int(sy) - 6)))

            else:  
                unit = uid_to_unit.get(uid)
                if unit is None: continue  

                if unit.flash_frames > 0: track_color = (255, 255, 255)

                if unit.image_path:
                    rotated = self._get_baked_sprite(unit.image_path, track_color, unit.heading, is_image=True)
                else:
                    rotated = self._get_baked_sprite(utype, track_color, unit.heading, is_image=False)
                
                rect = rotated.get_rect(center=(int(sx), int(sy)))
                sprite_blits.append((rotated, rect.topleft))

                if zoom < threshold: show_label = False
                if show_label:
                    callsign_lbl = self._get_text_surface(unit.callsign, (0, 0, 0), self._font_sm)
                    type_lbl = self._get_text_surface(unit.platform.display_name, (0, 0, 0), self._font_sm)
                    text_blits.append((callsign_lbl, (int(sx) + 13, int(sy) - 10)))
                    text_blits.append((type_lbl, (int(sx) + 13, int(sy) + 2)))

    def _draw_placement_cursor(self, mouse_pos: tuple[int, int],
                               placing_type: str,
                               placing_remaining: int = 0) -> None:
        mx, my = mouse_pos
        size   = 16
        col    = (80, 220, 120)
        pygame.draw.line(self.surface, col, (mx - size, my), (mx + size, my), 2)
        pygame.draw.line(self.surface, col, (mx, my - size), (mx, my + size), 2)
        pygame.draw.circle(self.surface, col, (mx, my), size, 1)
        rem_txt = f" ({placing_remaining} left)" if placing_remaining > 1 else ""
        label = self._get_text_surface(f"Place: {placing_type}{rem_txt}", col, self._font_sm)
        self.surface.blit(label, (mx + size + 4, my - 8))