# map_tiles.py — async OSM tile fetcher with disk cache

import os
import queue
import threading
import time
import itertools
import pygame

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "map_cache_en")
os.makedirs(CACHE_DIR, exist_ok=True)

# ── Clean up any orphaned temp files from a previous crashed run ─────────────
for _f in os.listdir(CACHE_DIR):
    if _f.endswith(".tmp"):
        try:
            os.remove(os.path.join(CACHE_DIR, _f))
        except OSError:
            pass

# ── State ─────────────────────────────────────────────────────────────────────
_tile_queue      = queue.Queue()
_queued_tiles    = set()          
_queued_lock     = threading.Lock()
_loaded_surfaces: dict[str, pygame.Surface] = {}  
_LRU_MAX         = 512

_SUBDOMAINS = itertools.cycle(["a", "b", "c", "d"])

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

_NUM_WORKERS = 4     


def _valid_tile(z: int, x: int, y: int) -> tuple[int, int, int] | None:
    if not (0 <= z <= 19):
        return None
    n = 2 ** z
    x = x % n          
    if not (0 <= y < n):
        return None     
    return z, x, y


def _worker() -> None:
    import requests
    while True:
        z, x, y = _tile_queue.get()
        key        = f"{z}_{x}_{y}"
        cache_path = os.path.join(CACHE_DIR, f"{key}.png")
        tmp_path   = os.path.join(CACHE_DIR, f"{key}.tmp")

        try:
            if not os.path.exists(cache_path):
                sub = next(_SUBDOMAINS)
                url = f"https://{sub}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}.png"
                
                r   = requests.get(url, headers=_HEADERS, timeout=10)
                if r.status_code == 200:
                    with open(tmp_path, "wb") as fh:
                        fh.write(r.content)
                    os.replace(tmp_path, cache_path)
                elif r.status_code == 429:
                    with _queued_lock:
                        _queued_tiles.discard(key)
                    time.sleep(1.0)
                    _tile_queue.task_done()
                    continue
                time.sleep(0.05)
        except Exception:
            pass
        finally:
            with _queued_lock:
                _queued_tiles.discard(key)
            _tile_queue.task_done()

for _i in range(_NUM_WORKERS):
    threading.Thread(target=_worker, daemon=True).start()


def get_tile(z: int, x: int, y: int) -> pygame.Surface | None:
    coords = _valid_tile(z, x, y)
    if coords is None:
        return None
    z, x, y = coords
    key = f"{z}_{x}_{y}"

    # TRUE LRU FIX: Pop and re-insert to move it to the end (most recently used)
    if key in _loaded_surfaces:
        surf = _loaded_surfaces.pop(key)
        _loaded_surfaces[key] = surf
        return surf

    cache_path = os.path.join(CACHE_DIR, f"{key}.png")
    if os.path.exists(cache_path):
        try:
            surf = pygame.image.load(cache_path).convert()
            if len(_loaded_surfaces) >= _LRU_MAX:
                oldest = next(iter(_loaded_surfaces))
                del _loaded_surfaces[oldest]
            _loaded_surfaces[key] = surf
            return surf
        except (pygame.error, Exception):
            try:
                os.remove(cache_path)
            except OSError:
                pass
            return None

    with _queued_lock:
        if key not in _queued_tiles:
            _queued_tiles.add(key)
            _tile_queue.put((z, x, y))

    return None