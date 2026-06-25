"""
FitForMe FastAPI
==================
A RESTful API wrapper around the AI outfit generation logic from main.py.

Endpoints
---------
GET  /                              � Health check / API info
GET  /events                        � All event categories & events
GET  /events/{category_key}         � Events in a specific category
GET  /styles                        � Style / config reference data
GET  /preferences                   � Get currently saved preferences
POST /preferences                   � Save / update preferences
GET  /wardrobe                      � List all wardrobe items
POST /wardrobe                      � Add a wardrobe item (multipart image upload)
GET  /wardrobe/{item_id}            � Get a single wardrobe item
DELETE /wardrobe/{item_id}          � Remove a wardrobe item
POST /outfit/generate               � Generate a new outfit (main + breakdown images)
POST /outfit/regenerate             � Regenerate an outfit with change notes
GET  /outfits                       � List all saved outfit records
GET  /outfits/{outfit_id}           � Get a single outfit record
DELETE /outfits/{outfit_id}         � Delete an outfit record and its images
GET  /images/{filename}             � Serve generated images
"""

import os
import re
import base64
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator

# ---------------------------------------------------------------------------
# Import all core helpers from main.py � single source of truth.
# ---------------------------------------------------------------------------
from main import (
    DATA_DIR, OUTPUT_DIR, OUTFITS_DIR, WARDROBE_DIR,
    PREFERENCES_FILE, WARDROBE_FILE, OUTFITS_FILE,
    EVENT_CATEGORIES, WARDROBE_CATEGORIES, STYLE_OPTIONS,
    COLOR_PALETTES, DEFAULT_PREFERENCES, IMAGE_MODEL,
    ensure_dirs, load_json, save_json, now_stamp, slugify,
    image_to_b64, guess_mime, load_llm, load_image_client,
    load_wardrobe_items, find_event_matches, build_wardrobe_refs,
    build_base_prompt, build_breakdown_from_prompt, refine_prompt_with_llm,
    generate_main_image, generate_breakdown_images, save_image_bytes,
    get_latest_preferences,
)

# ---------------------------------------------------------------------------
# App bootstrap
# ---------------------------------------------------------------------------
ensure_dirs()

app = FastAPI(
    title="FitForMe AI API",
    description=(
        "AI-powered fashion outfit generator. "
        "Generate full-body outfit images and individual item product shots "
        "for any event using Google Gemini."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class PreferencesIn(BaseModel):
    gender: str
    age_range: str
    kids_size: Optional[str] = None
    vibe: str
    colors: Union[str, List[str]]
    fit: str
    price_level: str

    @field_validator("gender")
    @classmethod
    def validate_gender(cls, v: str) -> str:
        allowed = {"male", "female", "prefer not to say"}
        if v.lower() not in allowed:
            raise ValueError(f"gender must be one of {allowed}")
        return v.lower()

    @field_validator("fit")
    @classmethod
    def validate_fit(cls, v: str) -> str:
        allowed = {"Tailored", "Regular", "Oversized"}
        if v not in allowed:
            raise ValueError(f"fit must be one of {allowed}")
        return v

    @field_validator("price_level")
    @classmethod
    def validate_price_level(cls, v: str) -> str:
        allowed = {"Affordable", "Mid-Range", "Luxury"}
        if v not in allowed:
            raise ValueError(f"price_level must be one of {allowed}")
        return v


class GenerateOutfitIn(BaseModel):
    event_category_key: str
    event: int
    outfit_name: Optional[str] = ""
    user_prompt: str
    source: str = "Outside"
    gender: Optional[str] = None
    style: Optional[str] = None
    fit: Optional[str] = None
    price_level: Optional[str] = None
    colors: Optional[Union[str, List[str]]] = None

    @field_validator("source")
    @classmethod
    def validate_source(cls, v: str) -> str:
        if v not in {"Outside", "Wardrobe"}:
            raise ValueError('source must be "Outside" or "Wardrobe"')
        return v

    @field_validator("event_category_key")
    @classmethod
    def validate_category_key(cls, v: str) -> str:
        if v not in EVENT_CATEGORIES:
            raise ValueError(
                f"event_category_key must be one of {list(EVENT_CATEGORIES.keys())}"
            )
        return v


class RegenerateOutfitIn(BaseModel):
    outfit_id: str
    changes: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _image_url(filename: str) -> str:
    return f"/images/{filename}"


def _outfit_response(record: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(record)
    mp = record.get("main_image_path")
    out["main_image_url"] = _image_url(Path(mp).name) if mp and Path(mp).exists() else None

    bi: Dict[str, str] = record.get("breakdown_images", {})
    urls: Dict[str, Optional[str]] = {}
    for role, path in bi.items():
        urls[role] = _image_url(Path(path).name) if path and Path(path).exists() else None
    out["breakdown_image_urls"] = urls

    stripped_refs = []
    for r in record.get("wardrobe_references", []):
        r2 = dict(r)
        r2.pop("image_base64", None)
        r2.pop("image_b64", None)
        stripped_refs.append(r2)
    out["wardrobe_references"] = stripped_refs
    return out


def _wardrobe_item_response(item: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(item)
    out.pop("image_base64", None)
    out.pop("image_b64", None)
    ip = item.get("image_path")
    out["image_url"] = _image_url(Path(ip).name) if ip and Path(ip).exists() else None
    return out


def _run_outfit_generation(req: GenerateOutfitIn) -> Dict[str, Any]:
    category_data = EVENT_CATEGORIES[req.event_category_key]
    if req.event < 0 or req.event >= len(category_data["events"]):
        raise ValueError(
            f"Event ID '{req.event}' not found in category '{category_data['name']}'. "
            f"Valid event IDs: 0 to {len(category_data['events'])-1}"
        )
    event_name = category_data["events"][req.event]
    event_category_name = category_data["name"]

    saved_prefs = get_latest_preferences()
    data = {
        "gender":      req.gender      or saved_prefs.get("gender", "neutral"),
        "event":       event_name,
        "style":       req.style       or str(saved_prefs.get("vibe", "minimal casual")),
        "fit":         req.fit         or saved_prefs.get("fit", "Regular"),
        "price_level": req.price_level or saved_prefs.get("price_level", "Mid-Range"),
        "colors":      req.colors      or saved_prefs.get("colors", "Neutrals"),
        "user_prompt": req.user_prompt,
    }

    used_wardrobe_items: List[Dict] = []
    if req.source == "Wardrobe":
        wardrobe_items = load_wardrobe_items()
        matched = find_event_matches(wardrobe_items, event_name)
        if matched:
            _, used_wardrobe_items = build_wardrobe_refs(
                matched, event_name, req.user_prompt, saved_prefs
            )

    breakdown = build_breakdown_from_prompt(
        req.user_prompt,
        used_wardrobe_items if req.source == "Wardrobe" else None,
    )
    base_prompt = build_base_prompt(
        data, breakdown,
        wardrobe_note=breakdown.get("source_note") if req.source == "Wardrobe" else None,
    )

    image_client = load_image_client()
    main_bytes = generate_main_image(
        image_client, base_prompt,
        used_wardrobe_items if req.source == "Wardrobe" else None,
    )

    outfit_id = f"outfit_{now_stamp()}_{slugify(req.outfit_name or event_name)}"
    main_image_path = OUTFITS_DIR / f"{outfit_id}_main.png"
    save_image_bytes(main_bytes, main_image_path)

    try:
        breakdown_image_paths = generate_breakdown_images(
            image_client, breakdown, outfit_id, main_image_bytes=main_bytes
        )
    except Exception:
        breakdown_image_paths = {}

    outfit_record = {
        "id":                  outfit_id,
        "outfit_name":         req.outfit_name or None,
        "event_category":      event_category_name,
        "event":               event_name,
        "prompt":              req.user_prompt,
        "final_image_prompt":  base_prompt,
        "source":              req.source,
        "main_image_path":     str(main_image_path),
        "breakdown_items":     breakdown,
        "breakdown_images":    breakdown_image_paths,
        "wardrobe_references": used_wardrobe_items,
        "created_at":          now_stamp(),
    }

    outfits = load_json(OUTFITS_FILE, [])
    outfits.append(outfit_record)
    save_json(OUTFITS_FILE, outfits)
    return outfit_record


# ===========================================================================
# Routes � General
# ===========================================================================

@app.get("/", tags=["General"])
def root() -> Dict[str, Any]:
    """Health check and API info."""
    return {
        "status":  "ok",
        "service": "FitForMe AI API",
        "version": "1.0.0",
        "docs":    "/docs",
    }


# ===========================================================================
# Routes � Events
# ===========================================================================

@app.get("/events", tags=["Events"])
def get_events() -> Dict[str, Any]:
    """List all event categories and their events."""
    return {
        "event_categories": {
            key: {"key": key, "name": data["name"], "events": data["events"]}
            for key, data in EVENT_CATEGORIES.items()
        }
    }


@app.get("/events/{category_key}", tags=["Events"])
def get_events_in_category(category_key: str) -> Dict[str, Any]:
    """Return events for a specific category key, e.g. '3'."""
    if category_key not in EVENT_CATEGORIES:
        raise HTTPException(
            status_code=404,
            detail=f"Category key '{category_key}' not found. Valid keys: {list(EVENT_CATEGORIES.keys())}",
        )
    data = EVENT_CATEGORIES[category_key]
    return {"key": category_key, "name": data["name"], "events": data["events"]}


# ===========================================================================
# Routes � Reference / Style config
# ===========================================================================

@app.get("/styles", tags=["Reference"])
def get_styles() -> Dict[str, Any]:
    """Return available style options, palettes, fits and price levels."""
    return {
        "style_options":       STYLE_OPTIONS,
        "color_palettes":      COLOR_PALETTES,
        "fits":                ["Tailored", "Regular", "Oversized"],
        "price_levels":        ["Affordable", "Mid-Range", "Luxury"],
        "genders":             ["male", "female", "prefer not to say"],
        "age_ranges":          ["10�17", "18�24", "25�34", "35�44", "45�64", "65+"],
        "wardrobe_categories": WARDROBE_CATEGORIES,
    }


# ===========================================================================
# Routes � Preferences
# ===========================================================================

@app.get("/preferences", tags=["Preferences"])
def get_preferences() -> Dict[str, Any]:
    """Return currently saved user preferences."""
    return {"preferences": get_latest_preferences()}


@app.post("/preferences", tags=["Preferences"])
def save_preferences(body: PreferencesIn) -> Dict[str, Any]:
    """
    Save (overwrite) user preferences.

    - **gender**: `male` | `female` | `prefer not to say`
    - **age_range**: one of `10�17`, `18�24`, `25�34`, `35�44`, `45�64`, `65+`
    - **kids_size**: `S` | `M` | `L` | `null`
    - **vibe**: e.g. `Streetwear`, `Elevated Casual`, `Sharp Tailored` �
    - **colors**: palette name (string) OR list of HEX codes
    - **fit**: `Tailored` | `Regular` | `Oversized`
    - **price_level**: `Affordable` | `Mid-Range` | `Luxury`
    """
    prefs = {
        "gender":      body.gender,
        "age_range":   body.age_range,
        "kids_size":   body.kids_size,
        "vibe":        body.vibe,
        "colors":      body.colors,
        "fit":         body.fit,
        "price_level": body.price_level,
        "updated_at":  now_stamp(),
    }
    save_json(PREFERENCES_FILE, prefs)
    return {"message": "Preferences saved successfully.", "preferences": prefs}


# ===========================================================================
# Routes � Wardrobe
# ===========================================================================

@app.get("/wardrobe", tags=["Wardrobe"])
def list_wardrobe(
    category: Optional[str] = None,
    event: Optional[str] = None,
) -> Dict[str, Any]:
    """
    List wardrobe items. Optionally filter by `category` and/or `event`.

    - **category**: `Tops` | `Bottoms` | `Shoes` | `Outerwear` | `Accessories`
    - **event**: exact event name, e.g. `Glastonbury`
    """
    items = load_wardrobe_items()
    if category:
        items = [i for i in items if i.get("category", "").lower() == category.lower()]
    if event:
        items = [
            i for i in items
            if any(e.lower() == event.lower() for e in i.get("events", []))
        ]
    return {"count": len(items), "wardrobe": [_wardrobe_item_response(i) for i in items]}


@app.post("/wardrobe", tags=["Wardrobe"])
async def add_wardrobe_item(
    title:              str        = Form(..., description="Item title"),
    subline:            str        = Form(..., description="Brief description / subline"),
    category:           str        = Form(..., description="Tops | Bottoms | Outerwear | Shoes | Accessories"),
    event_category_key: str        = Form(..., description="Event category key, e.g. '3'"),
    event:              int        = Form(..., description="Specific event ID (index)"),
    image:              UploadFile = File(..., description="Clothing item image"),
) -> Dict[str, Any]:
    """
    Add a new wardrobe item.  Send as **multipart/form-data** with the image attached.
    """
    if category not in WARDROBE_CATEGORIES:
        raise HTTPException(status_code=422, detail=f"category must be one of {WARDROBE_CATEGORIES}")
    if event_category_key not in EVENT_CATEGORIES:
        raise HTTPException(
            status_code=422,
            detail=f"event_category_key must be one of {list(EVENT_CATEGORIES.keys())}",
        )
    category_data = EVENT_CATEGORIES[event_category_key]
    if event < 0 or event >= len(category_data["events"]):
        raise HTTPException(
            status_code=422,
            detail=f"Event ID '{event}' not in '{category_data['name']}'. Valid IDs: 0 to {len(category_data['events'])-1}",
        )
    event_name = category_data["events"][event]

    suffix   = Path(image.filename or "upload.png").suffix or ".png"
    item_id  = f"wardrobe_{now_stamp()}_{slugify(title)}"
    saved_img = WARDROBE_DIR / f"{item_id}{suffix}"

    try:
        contents = await image.read()
        with open(saved_img, "wb") as f:
            f.write(contents)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to save image: {exc}")

    try:
        image_base64 = image_to_b64(saved_img)
    except Exception:
        image_base64 = None

    item = {
        "id":             item_id,
        "image_path":     str(saved_img),
        "image_base64":   image_base64,
        "title":          title,
        "subline":        subline,
        "category":       category,
        "events":         [event_name],
        "event_category": category_data["name"],
        "created_at":     now_stamp(),
    }

    wardrobe = load_json(WARDROBE_FILE, [])
    wardrobe.append(item)
    save_json(WARDROBE_FILE, wardrobe)
    return {"message": "Wardrobe item added successfully.", "item": _wardrobe_item_response(item)}


@app.get("/wardrobe/{item_id}", tags=["Wardrobe"])
def get_wardrobe_item(item_id: str) -> Dict[str, Any]:
    """Get a single wardrobe item by ID."""
    for item in load_wardrobe_items():
        if item.get("id") == item_id:
            return {"item": _wardrobe_item_response(item)}
    raise HTTPException(status_code=404, detail=f"Wardrobe item '{item_id}' not found.")


@app.delete("/wardrobe/{item_id}", tags=["Wardrobe"])
def delete_wardrobe_item(item_id: str) -> Dict[str, Any]:
    """Delete a wardrobe item and its image file."""
    wardrobe = load_json(WARDROBE_FILE, [])
    found = None
    remaining = []
    for item in wardrobe:
        if item.get("id") == item_id:
            found = item
        else:
            remaining.append(item)
    if not found:
        raise HTTPException(status_code=404, detail=f"Wardrobe item '{item_id}' not found.")
    ip = found.get("image_path")
    if ip and Path(ip).exists():
        try:
            Path(ip).unlink()
        except Exception:
            pass
    save_json(WARDROBE_FILE, remaining)
    return {"message": f"Wardrobe item '{item_id}' deleted successfully."}


# ===========================================================================
# Routes � Outfit generation
# ===========================================================================

@app.post("/outfit/generate", tags=["Outfit"])
def generate_outfit(body: GenerateOutfitIn) -> Dict[str, Any]:
    """
    Generate a new AI outfit.

    1. Builds an outfit breakdown from the user's `user_prompt`.
    2. Generates a **full-body fashion main image** (realistic smiling model).
    3. Generates individual **studio product-shot images** per item.
    4. Persists the record and returns all image URLs.

    **Required:**
    - `event_category_key` � category key `"1"` � `"8"` (see `GET /events`)
    - `event` � exact event name from that category
    - `user_prompt` � describe the outfit

    **Optional overrides** (fall back to saved preferences):
    `outfit_name`, `source`, `gender`, `style`, `fit`, `price_level`, `colors`
    """
    try:
        record = _run_outfit_generation(body)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Outfit generation failed: {exc}")
    return {"message": "Outfit generated successfully.", "outfit": _outfit_response(record)}


@app.post("/outfit/regenerate", tags=["Outfit"])
def regenerate_outfit(body: RegenerateOutfitIn) -> Dict[str, Any]:
    """
    Regenerate an existing outfit with specific change notes.

    The original main image is sent as a visual reference alongside the
    refined prompt, so the model understands what to keep and what to change.

    **Required:**
    - `outfit_id` � ID of the outfit to base the regeneration on
    - `changes` � what should change, e.g. `"swap the jacket for a trench coat"`
    """
    outfits = load_json(OUTFITS_FILE, [])
    original = next((o for o in outfits if o.get("id") == body.outfit_id), None)
    if not original:
        raise HTTPException(status_code=404, detail=f"Outfit '{body.outfit_id}' not found.")

    original_prompt  = original.get("prompt", "")
    regen_user_prompt = f"{original_prompt}. Changes requested: {body.changes}"

    saved_prefs = get_latest_preferences()
    data = {
        "gender":      saved_prefs.get("gender", "neutral"),
        "event":       original["event"],
        "style":       str(saved_prefs.get("vibe", "minimal casual")),
        "fit":         saved_prefs.get("fit", "Regular"),
        "price_level": saved_prefs.get("price_level", "Mid-Range"),
        "colors":      saved_prefs.get("colors", "Neutrals"),
        "user_prompt": regen_user_prompt,
    }

    regen_breakdown  = build_breakdown_from_prompt(regen_user_prompt)
    regen_base_prompt = build_base_prompt(data, regen_breakdown)

    try:
        llm = load_llm()
        regen_prompt = refine_prompt_with_llm(llm, regen_base_prompt)
    except Exception:
        regen_prompt = regen_base_prompt

    image_client = load_image_client()
    inputs: List[Dict] = [{"type": "text", "text": regen_prompt}]
    ref_bytes = None
    original_main = original.get("main_image_path")
    if original_main and Path(original_main).exists():
        try:
            with open(original_main, "rb") as f:
                ref_bytes = f.read()
            inputs.append({
                "type":      "image",
                "data":      base64.b64encode(ref_bytes).decode("utf-8"),
                "mime_type": "image/png",
            })
        except Exception:
            pass

    try:
        interaction = image_client.interactions.create(model=IMAGE_MODEL, input=inputs)
        regen_bytes = base64.b64decode(interaction.output_image.data)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Regeneration failed: {exc}")

    outfit_id = body.outfit_id
    match = re.search(r"_v(\d+)$", outfit_id)
    if match:
        version  = int(match.group(1)) + 1
        regen_id = re.sub(r"_v\d+$", f"_v{version}", outfit_id)
    else:
        regen_id = f"{outfit_id}_v2"

    regen_main_path = OUTFITS_DIR / f"{regen_id}_main.png"
    save_image_bytes(regen_bytes, regen_main_path)

    try:
        regen_breakdown_paths = generate_breakdown_images(
            image_client, regen_breakdown, regen_id, main_image_bytes=regen_bytes
        )
    except Exception:
        regen_breakdown_paths = {}

    regen_record = dict(original)
    regen_record.update({
        "id":                 regen_id,
        "prompt":             regen_user_prompt,
        "final_image_prompt": regen_prompt,
        "main_image_path":    str(regen_main_path),
        "breakdown_items":    regen_breakdown,
        "breakdown_images":   regen_breakdown_paths,
        "created_at":         now_stamp(),
        "regenerated_from":   outfit_id,
        "regeneration_note":  body.changes,
    })

    outfits = load_json(OUTFITS_FILE, [])
    outfits.append(regen_record)
    save_json(OUTFITS_FILE, outfits)
    return {"message": "Outfit regenerated successfully.", "outfit": _outfit_response(regen_record)}


# ===========================================================================
# Routes � Outfit records
# ===========================================================================

@app.get("/outfits", tags=["Outfit"])
def list_outfits(
    event:  Optional[str] = None,
    limit:  int = 50,
    offset: int = 0,
) -> Dict[str, Any]:
    """
    List all saved outfits (newest first).

    Query params: `event` (filter), `limit`, `offset` (pagination).
    """
    outfits: List[Dict] = list(reversed(load_json(OUTFITS_FILE, [])))
    if event:
        outfits = [o for o in outfits if o.get("event", "").lower() == event.lower()]
    total = len(outfits)
    page  = outfits[offset: offset + limit]
    return {
        "total":   total,
        "limit":   limit,
        "offset":  offset,
        "outfits": [_outfit_response(o) for o in page],
    }


@app.get("/outfits/{outfit_id}", tags=["Outfit"])
def get_outfit(outfit_id: str) -> Dict[str, Any]:
    """Get a single outfit record by ID."""
    for outfit in load_json(OUTFITS_FILE, []):
        if outfit.get("id") == outfit_id:
            return {"outfit": _outfit_response(outfit)}
    raise HTTPException(status_code=404, detail=f"Outfit '{outfit_id}' not found.")


@app.delete("/outfits/{outfit_id}", tags=["Outfit"])
def delete_outfit(outfit_id: str, delete_images: bool = True) -> Dict[str, Any]:
    """
    Delete an outfit record.  Pass `?delete_images=false` to keep image files.
    """
    outfits  = load_json(OUTFITS_FILE, [])
    found    = None
    remaining = []
    for o in outfits:
        if o.get("id") == outfit_id:
            found = o
        else:
            remaining.append(o)
    if not found:
        raise HTTPException(status_code=404, detail=f"Outfit '{outfit_id}' not found.")
    if delete_images:
        for path in [found.get("main_image_path")] + list(found.get("breakdown_images", {}).values()):
            if path and Path(path).exists():
                try:
                    Path(path).unlink()
                except Exception:
                    pass
    save_json(OUTFITS_FILE, remaining)
    return {"message": f"Outfit '{outfit_id}' deleted successfully."}


# ===========================================================================
# Routes � Image serving
# ===========================================================================

@app.get("/images/{filename}", tags=["Images"])
def serve_image(filename: str) -> FileResponse:
    """
    Serve a generated image by filename.
    Searches both `outputs/outfits/` and `outputs/wardrobe/`.
    """
    for directory in (OUTFITS_DIR, WARDROBE_DIR):
        candidate = directory / filename
        if candidate.exists():
            return FileResponse(str(candidate), media_type=guess_mime(candidate))
    raise HTTPException(status_code=404, detail=f"Image '{filename}' not found.")


# ===========================================================================
# Entrypoint
# ===========================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
