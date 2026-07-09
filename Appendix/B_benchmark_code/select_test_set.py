"""Select 40 stratified basemaps from the TEST split for benchmark evaluation.

Stratification goals:
- All 10 tile providers represented (each at least once, avg ~4 per provider)
- All 12 test locations covered
- Variation across zoom levels (overview: z4/z6, regional: z8/z10, local: z12/z14)
- Variation across land cover types (urban, rural, coast, desert, mountain)
- Variation across climate zones

This ensures the benchmark is not biased toward any basemap style or geography.
"""

import csv
import json
import random
from pathlib import Path

# Paths
ROOT = Path(__file__).resolve().parent.parent.parent
LOCATIONS_CSV = ROOT / "data" / "locations.csv"
BASEMAPS_DIR = ROOT / "data" / "processed" / "basemaps"
OUTPUT_JSON = Path(__file__).parent / "output" / "test_set.json"

# All 10 providers grouped by visual character
PROVIDER_GROUPS = {
    "dark_abstract": ["carto_dark"],
    "light_abstract": ["carto_positron", "esri_lightgray"],
    "street_standard": ["osm", "carto_voyager"],
    "satellite": ["esri_imagery"],
    "terrain": ["otm", "stadia_terrain"],
    "artistic": ["stadia_watercolor", "stadia_toner"],
}

ZOOM_LEVELS = {
    "overview": ["z4", "z6"],
    "regional": ["z8", "z10"],
    "local": ["z12", "z14"],
}


def _basemap_path(provider: str, loc_id: str, zoom: str) -> Path:
    return BASEMAPS_DIR / f"{provider}_{loc_id}_{zoom}.png"


def _basemap_exists(provider: str, loc_id: str, zoom: str) -> bool:
    return _basemap_path(provider, loc_id, zoom).exists()


def load_test_locations() -> list:
    """Load all locations with split=='test' from the metadata CSV."""
    with open(LOCATIONS_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [row for row in reader if row["split"].strip() == "test"]


def select_stratified_40(seed: int = 42) -> list:
    """Select 40 basemaps via deterministic stratified sampling.

    Strategy:
        For each provider group, pick basemaps covering multiple locations
        and zoom ranges. Target distribution:
            dark_abstract:     4  (1 provider * 4)
            light_abstract:    8  (2 providers * 4)
            street_standard:   8  (2 providers * 4)
            satellite:         4  (1 provider * 4)
            terrain:           8  (2 providers * 4)
            artistic:          8  (2 providers * 4)
            ---
            total:             40
    """
    rng = random.Random(seed)
    test_locs = load_test_locations()

    # Flatten: list of all providers (10 total)
    all_providers = [p for group in PROVIDER_GROUPS.values() for p in group]

    selected = []
    seen = set()

    for provider in all_providers:
        # For each provider, aim for 4 basemaps across different
        # locations and zoom ranges
        provider_picks = []
        loc_pool = list(test_locs)
        rng.shuffle(loc_pool)

        # Try to cover at least 3 different zoom ranges per provider
        zoom_range_cycle = ["overview", "regional", "local", "regional"]

        zi = 0
        for loc in loc_pool:
            if len(provider_picks) >= 4:
                break

            zoom_range = zoom_range_cycle[zi % len(zoom_range_cycle)]
            zoom_options = list(ZOOM_LEVELS[zoom_range])
            rng.shuffle(zoom_options)

            for zoom in zoom_options:
                key = (provider, loc["id"], zoom)
                if key in seen:
                    continue
                if not _basemap_exists(provider, loc["id"], zoom):
                    continue

                provider_picks.append({
                    "provider": provider,
                    "location_id": loc["id"],
                    "location_name": loc["name"],
                    "climate_zone": loc["climate_zone"],
                    "land_cover": loc["land_cover"],
                    "zoom": zoom,
                    "zoom_range": zoom_range,
                    "basemap_filename": f"{provider}_{loc['id']}_{zoom}.png",
                })
                seen.add(key)
                zi += 1
                break

        selected.extend(provider_picks)

    return selected


def write_test_set(selected: list) -> None:
    """Write test set metadata as JSON for downstream use."""
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    # Summary statistics for transparency
    from collections import Counter
    providers = Counter(s["provider"] for s in selected)
    zooms = Counter(s["zoom_range"] for s in selected)
    land_covers = Counter(s["land_cover"] for s in selected)
    climates = Counter(s["climate_zone"] for s in selected)
    locations = Counter(s["location_name"] for s in selected)

    payload = {
        "n_basemaps": len(selected),
        "seed": 42,
        "summary": {
            "providers": dict(providers),
            "zoom_ranges": dict(zooms),
            "land_covers": dict(land_covers),
            "climate_zones": dict(climates),
            "locations": dict(locations),
        },
        "basemaps": selected,
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(selected)} basemaps to {OUTPUT_JSON}")
    print(f"\nDistribution by provider:")
    for p, n in sorted(providers.items()):
        print(f"  {p:25s}  {n}")
    print(f"\nDistribution by zoom range:")
    for z, n in zooms.items():
        print(f"  {z:12s}  {n}")
    print(f"\nDistribution by land cover:")
    for lc, n in land_covers.items():
        print(f"  {lc:12s}  {n}")
    print(f"\nDistribution by climate zone:")
    for cz, n in climates.items():
        print(f"  {cz:20s}  {n}")


if __name__ == "__main__":
    selected = select_stratified_40()
    write_test_set(selected)
