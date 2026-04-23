# Assets

This folder is the local-first asset pipeline for the Phase 3B.5 race-control UI.

It is intentionally simple:

- `asset_manifest.json` maps teams and drivers to optional assets and colors
- `team_logos/` holds optional team logos
- `driver_photos/` holds optional driver photos

No official assets are downloaded automatically.

## Manifest

The app reads `assets/asset_manifest.json` and supports:

- local relative file paths inside this repo
- manually supplied external URLs
- `null` values for missing assets

Example:

```json
{
  "teams": {
    "Ferrari": {
      "logo": "assets/team_logos/ferrari.png",
      "primary_color": "#DC0000"
    }
  },
  "drivers": {
    "LEC": {
      "photo": "assets/driver_photos/lec.png",
      "team": "Ferrari"
    }
  }
}
```

## Placement

- Put team logos in `assets/team_logos/`
- Put driver photos in `assets/driver_photos/`
- Reference them from `asset_manifest.json`

The app inlines local files as data URIs before sending them to Streamlit/native HTML or the custom component, so missing files fail quietly instead of breaking the board.

## Fallbacks

- Missing team logo: team name still renders cleanly
- Missing driver photo: selected-car header stays text-only
- Missing team color: the UI falls back to the default board cyan

## Team colors

`primary_color` is used for selected-state accents such as:

- timing-row highlight
- selected-car header accent
- pit-call card border/glow
- selected track-marker halo

The dark tactical base theme remains the default background; team colors are accents only.
