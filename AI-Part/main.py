import os
import json
import base64
import shutil
from pathlib import Path
from datetime import datetime
from mimetypes import guess_type

from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont

from langchain_google_genai import ChatGoogleGenerativeAI
from google import genai


# =========================================================
# CONFIG
# =========================================================

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTFITS_DIR = OUTPUT_DIR / "outfits"
WARDROBE_DIR = OUTPUT_DIR / "wardrobe"

PREFERENCES_FILE = DATA_DIR / "preferences.json"
WARDROBE_FILE = DATA_DIR / "wardrobe.json"
OUTFITS_FILE = DATA_DIR / "outfits.json"

TEXT_MODEL = "gemini-2.5-flash"
IMAGE_MODEL = "gemini-2.5-flash-image"

DEFAULT_PREFERENCES = {
    "gender": "neutral",
    "age_range": "18–24",
    "kids_size": None,
    "vibe": "minimal casual",
    "colors": ["Neutrals"],
    "fit": "Regular",
    "price_level": "Mid-Range",
}

EVENT_CATEGORIES = {
    "1": {
        "name": "Sports — Domestic",
        "events": [
            "Derby", "F1", "Super Bowl", "The Masters", "Ryder Cup",
            "US Open Golf", "Wimbledon", "US Open Tennis", "NBA Finals",
            "World Cup", "College Football Championship", "Daytona 500",
            "Preakness", "Belmont Stakes", "March Madness", "NFL Draft",
            "MLB World Series", "NHL Stanley Cup", "Boxing", "WWE", "UFC"
        ],
    },
    "2": {
        "name": "Sports — International",
        "events": [
            "French Open", "Copa America", "Tour de France", "Premier League",
            "Australian Open", "Ryder Cup Europe", "Rugby World Cup",
            "Six Nations Rugby", "Champions League Final", "Cricket World Cup"
        ],
    },
    "3": {
        "name": "Music & Entertainment",
        "events": [
            "Stadium Concerts", "Coachella", "Lollapalooza", "Glastonbury",
            "Tomorrowland", "Jazz Festivals", "Country Music Festivals",
            "Opera", "Broadway", "Comedy Shows", "Award Shows", "BET Awards", "Met Gala"
        ],
    },
    "4": {
        "name": "Cultural & International",
        "events": [
            "Oktoberfest", "Carnival Rio", "Mardi Gras", "New Year's Eve",
            "St. Patrick's Day", "Pride Parade", "Running of the Bulls",
            "Monaco Grand Prix", "Cannes Film Festival", "Fashion Week"
        ],
    },
    "5": {
        "name": "Professional & Career",
        "events": [
            "Job Interview", "Industry Summit", "Work Event", "Corporate Meeting",
            "Networking Event", "Business Conference"
        ],
    },
    "6": {
        "name": "Dating & Social",
        "events": [
            "Date Night", "First Date", "Cocktail Party", "Dinner Party",
            "Rooftop Brunch", "White Party", "Garden Party", "Watch Party", "Happy Hour"
        ],
    },
    "7": {
        "name": "Casual & Everyday",
        "events": [
            "Casual Day Out", "Coffee Run", "Shopping Day", "Weekend Hangout",
            "Errands", "Family Gathering"
        ],
    },
    "8": {
        "name": "Celebrations & Parties",
        "events": [
            "Birthday Party", "House Party", "Pool Party", "Yacht Party",
            "Holiday Party", "Graduation Party", "Retirement Party"
        ],
    },
}

WARDROBE_CATEGORIES = ["Tops", "Bottoms", "Outerwear", "Shoes", "Accessories"]
ROLE_MAP = {
    "top": "Tops",
    "tops": "Tops",
    "bottom": "Bottoms",
    "bottoms": "Bottoms",
    "outerwear": "Outerwear",
    "outwear": "Outerwear",
    "shoes": "Shoes",
    "accessories": "Accessories",
    "accessory": "Accessories",
}

STYLE_OPTIONS = {
    "women": ["Streetwear", "Elevated Casual", "Business Chic", "Maximalist"],
    "men": ["Street & Hype", "Sharp Tailored", "Rugged Heritage", "Athleisure"],
}

COLOR_PALETTES = [
    "Neutrals", "Earth", "Mono Carbon", "Pastels",
    "Forest", "Brights", "Denim & Indigo", "Noir & Gold"
]


# =========================================================
# HELPERS
# =========================================================

def ensure_dirs():
    DATA_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)
    OUTFITS_DIR.mkdir(exist_ok=True)
    WARDROBE_DIR.mkdir(exist_ok=True)


def load_json(path, default):
    if not path.exists():
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def now_stamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def slugify(text):
    text = text.lower().strip()
    out = []
    for ch in text:
        if ch.isalnum():
            out.append(ch)
        else:
            out.append("_")
    slug = "".join(out)
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_") or "item"


def ask_text(prompt, allow_empty=False):
    while True:
        value = input(prompt).strip()
        if value or allow_empty:
            return value
        print("This field is required.")


def ask_yes_no(prompt):
    while True:
        value = input(prompt + " (y/n): ").strip().lower()
        if value in ("y", "yes"):
            return True
        if value in ("n", "no"):
            return False
        print("Please enter y or n.")


def choose_from_list(prompt, options):
    while True:
        print(prompt)
        for idx, opt in enumerate(options, start=1):
            print(f"{idx}. {opt}")
        choice = input("Choose number: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(options):
            return options[int(choice) - 1]
        print("Invalid choice. Try again.")


def choose_multiple_from_list(prompt, options):
    while True:
        print(prompt)
        for idx, opt in enumerate(options, start=1):
            print(f"{idx}. {opt}")
        raw = input("Choose numbers separated by commas: ").strip()
        if not raw:
            print("Please choose at least one option.")
            continue
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        if all(p.isdigit() and 1 <= int(p) <= len(options) for p in parts):
            uniq = []
            for p in parts:
                val = options[int(p) - 1]
                if val not in uniq:
                    uniq.append(val)
            return uniq
        print("Invalid choice. Try again.")


def get_latest_preferences():
    data = load_json(PREFERENCES_FILE, None)
    if not data:
        return DEFAULT_PREFERENCES.copy()
    return data


def normalize_category(value):
    if not value:
        return None
    return ROLE_MAP.get(value.strip().lower(), value.strip().title())


def image_to_b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def guess_mime(path):
    mime, _ = guess_type(str(path))
    return mime or "image/png"


def safe_font(size=22):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        return ImageFont.load_default()


def load_llm():
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    return ChatGoogleGenerativeAI(
        model=TEXT_MODEL,
        temperature=0.25,
        max_retries=2,
        api_key=api_key,
    )


def load_image_client():
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if api_key:
        return genai.Client(api_key=api_key)
    return genai.Client()


# =========================================================
# INPUT FLOW
# =========================================================

def add_preferences():
    print("\n=== ADD PREFERENCES ===")

    gender = choose_from_list("Select gender:", ["male", "female", "prefer not to say"])
    age_range = choose_from_list(
        "Select age range:",
        ["10–17", "18–24", "25–34", "35–44", "45–64", "65+"]
    )
    kids_size = choose_from_list(
        "Dressing kids (optional):",
        ["Skip", "S", "M", "L"]
    )
    if kids_size == "Skip":
        kids_size = None

    if gender == "female":
        vibe = choose_from_list("Select default vibe:", STYLE_OPTIONS["women"])
    elif gender == "male":
        vibe = choose_from_list("Select default vibe:", STYLE_OPTIONS["men"])
    else:
        vibe = choose_from_list(
            "Select default vibe:",
            STYLE_OPTIONS["women"] + STYLE_OPTIONS["men"]
        )

    print("Color preference:")
    color_mode = choose_from_list("Choose color mode:", ["Predefined palette", "Custom HEX codes"])
    if color_mode == "Predefined palette":
        colors = choose_from_list("Select palette:", COLOR_PALETTES)
    else:
        colors = ask_text("Enter custom HEX color codes separated by commas: ")

    fit = choose_from_list("Select preferred fit:", ["Tailored", "Regular", "Oversized"])
    price_level = choose_from_list("Select price level:", ["Affordable", "Mid-Range", "Luxury"])

    prefs = {
        "gender": gender,
        "age_range": age_range,
        "kids_size": kids_size,
        "vibe": vibe,
        "colors": colors,
        "fit": fit,
        "price_level": price_level,
        "updated_at": now_stamp(),
    }
    save_json(PREFERENCES_FILE, prefs)
    print("\nSaved Preference:")
    print(json.dumps(prefs, indent=2, ensure_ascii=False))


def select_event():
    print("\n=== SELECT EVENT CATEGORY ===")
    category_keys = list(EVENT_CATEGORIES.keys())
    for k in category_keys:
        print(f"{k}. {EVENT_CATEGORIES[k]['name']}")
    while True:
        cat_choice = input("Choose category number: ").strip()
        if cat_choice in EVENT_CATEGORIES:
            category = EVENT_CATEGORIES[cat_choice]["name"]
            events = EVENT_CATEGORIES[cat_choice]["events"]
            break
        print("Invalid category. Try again.")

    event = choose_from_list(f"Select event under {category}:", events)
    return category, event


def add_wardrobe():
    print("\n=== ADD WARDROBE ===")
    image_path = Path(ask_text("Enter local image path: "))
    while not image_path.exists():
        print("File not found. Try again.")
        image_path = Path(ask_text("Enter local image path: "))

    title = ask_text("Title: ")
    subline = ask_text("Subline: ")
    category = choose_from_list("Select wardrobe category:", WARDROBE_CATEGORIES)

    print("Select event category for this wardrobe item:")
    event_category, event = select_event()
    selected_events = [event]

    item_id = f"wardrobe_{now_stamp()}_{slugify(title)}"
    ext = image_path.suffix if image_path.suffix else ".png"
    saved_image = WARDROBE_DIR / f"{item_id}{ext}"
    shutil.copy2(image_path, saved_image)
    # Save base64 version (important for AI reference)
    try:
        image_base64 = image_to_b64(saved_image)
    except Exception:
        image_base64 = None

    wardrobe = load_json(WARDROBE_FILE, [])
    item = {
        "id": item_id,
        "image_path": str(saved_image),
        "image_base64": image_base64,
        "title": title,
        "subline": subline,
        "category": category,
        "events": selected_events,
        "event_category": event_category,
        "created_at": now_stamp(),
    }
    wardrobe.append(item)
    save_json(WARDROBE_FILE, wardrobe)

    print("\nWardrobe item saved:")
    print(json.dumps(item, indent=2, ensure_ascii=False))


# =========================================================
# WARDROBE MATCHING
# =========================================================

def load_wardrobe_items():
    return load_json(WARDROBE_FILE, [])


def find_event_matches(wardrobe_items, event_name):
    matches = []
    for item in wardrobe_items:
        events = item.get("events", [])
        if any(str(e).lower() == event_name.lower() for e in events):
            matches.append(item)
    return matches


def score_item(item, prompt, prefs):
    score = 0
    text = f"{item.get('title', '')} {item.get('subline', '')}".lower()
    prompt_words = set(re.findall(r"[a-zA-Z]+", prompt.lower()))
    text_words = set(re.findall(r"[a-zA-Z]+", text))

    score += len(prompt_words & text_words)

    vibe = str(prefs.get("vibe", "")).lower()
    colors = str(prefs.get("colors", "")).lower()
    fit = str(prefs.get("fit", "")).lower()
    price = str(prefs.get("price_level", "")).lower()

    for token in [vibe, colors, fit, price]:
        if token and token in text:
            score += 3

    return score


def pick_best_item(items, prompt, prefs):
    if not items:
        return None
    ranked = sorted(items, key=lambda x: score_item(x, prompt, prefs), reverse=True)
    return ranked[0]


def build_wardrobe_refs(wardrobe_items, selected_event, prompt, prefs):
    refs = []
    used_items = []
    for role in WARDROBE_CATEGORIES:
        role_items = [
            item for item in wardrobe_items
            if item.get("category") == role and any(e.lower() == selected_event.lower() for e in item.get("events", []))
        ]
        best = pick_best_item(role_items, prompt, prefs)
        if best:
            refs.append(best["image_path"])
            used_items.append(best)
    return refs, used_items


# =========================================================
# PROMPT ENGINEERING
# =========================================================

def build_base_prompt(data, breakdown, wardrobe_note=None):
    colors = data["colors"] if isinstance(data["colors"], str) else ", ".join(data["colors"])
    prompt = f"""
Full body fashion photo of a {data['gender']} wearing a {data['style']} outfit for {data['event']}.
Outfit includes {breakdown.get('top', 'a suitable top')}, {breakdown.get('bottom', 'a suitable bottom')}, {breakdown.get('shoes', 'appropriate shoes')}, {breakdown.get('outerwear', 'no outerwear if not needed')}, {breakdown.get('accessories', 'minimal accessories if needed')}.
Color palette: {colors}.
Fit: {data['fit']}.
Price level: {data['price_level']}.
The outfit must be realistic, modern, polished, and event-appropriate.
Background: realistic {data['event']} setting.
High-detail professional fashion photography, full body, clean composition, sharp focus.
""".strip()

    if wardrobe_note:
        prompt += f"\nUse provided wardrobe items as reference: {wardrobe_note}"

    prompt += f"\nUser styling note: {data['user_prompt']}"
    return prompt


def refine_prompt_with_llm(llm, prompt_text):
    try:
        msg = (
            "Rewrite this outfit image prompt to be clean, clear, and concise. "
            "Keep all important details. Keep it under 150 words. "
            "Return only the prompt text.\n\n"
            f"{prompt_text}"
        )
        result = llm.invoke(msg)
        text = getattr(result, "content", str(result)).strip()
        return text if text else prompt_text
    except Exception:
        return prompt_text


# =========================================================
# IMAGE GENERATION
# =========================================================

def generate_main_image(image_client, prompt_text, wardrobe_items=None):
    inputs = [{"type": "text", "text": prompt_text}]

    # Use stored base64 directly from wardrobe items when provided
    if wardrobe_items:
        for item in wardrobe_items:
            b64 = item.get("image_base64") or item.get("image_b64")
            if b64:
                inputs.append({
                    "type": "image",
                    "data": b64,
                    "mime_type": "image/png",
                })

    interaction = image_client.interactions.create(
        model=IMAGE_MODEL,
        input=inputs,
    )

    if not getattr(interaction, "output_image", None):
        raise RuntimeError("No image returned by the model.")

    image_b64 = interaction.output_image.data
    return base64.b64decode(image_b64)


def save_image_bytes(image_bytes, path):
    with open(path, "wb") as f:
        f.write(image_bytes)


def create_breakdown_image(breakdown, save_path, title="Outfit Breakdown"):
    canvas = Image.new("RGB", (1200, 900), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = safe_font(36)
    body_font = safe_font(26)

    draw.text((50, 40), title, fill="black", font=title_font)

    lines = [
        f"Top: {breakdown.get('top', 'N/A')}",
        f"Bottom: {breakdown.get('bottom', 'N/A')}",
        f"Shoes: {breakdown.get('shoes', 'N/A')}",
        f"Outerwear: {breakdown.get('outerwear', 'N/A')}",
        f"Accessories: {breakdown.get('accessories', 'N/A')}",
    ]

    y = 130
    for line in lines:
        draw.text((60, y), line, fill="black", font=body_font)
        y += 90

    canvas.save(save_path)


def generate_item_image(image_client, item_name, base_prompt="", reference_image_bytes=None):
    """
    Generate individual clothing item image (product shot style).
    """
    prompt = f"""
Studio product shot of {item_name}.
No human model.
Clean white background.
Highly detailed clothing photography.
Focus only on the item.
Keep same style as outfit.
{base_prompt}
""".strip()

    inputs = [{"type": "text", "text": prompt}]
    # attach reference main image so generated product shot matches main image
    if reference_image_bytes:
        inputs.append({
            "type": "image",
            "data": base64.b64encode(reference_image_bytes).decode("utf-8"),
            "mime_type": "image/png",
        })

    interaction = image_client.interactions.create(
        model=IMAGE_MODEL,
        input=inputs,
    )

    if not getattr(interaction, "output_image", None):
        raise RuntimeError("No image returned for item generation.")

    return base64.b64decode(interaction.output_image.data)


def generate_breakdown_images(image_client, breakdown, outfit_id, main_image_bytes=None):
    paths = {}

    # Determine which breakdown items we'll actually generate
    to_generate = []
    for key, value in breakdown.items():
        if not value or key == "source_note":
            continue
        if isinstance(value, str) and value.strip().lower() == "no outerwear if not needed":
            continue
        to_generate.append((key, value))

    if not to_generate:
        print("No breakdown items to generate.")
        return paths

    print("\nPlanned breakdown images to generate:")
    for key, value in to_generate:
        print(f"- {key}: {value}")

    # Generate each breakdown item, sending the main image each time for consistency
    for key, value in to_generate:
        try:
            img_bytes = generate_item_image(image_client, value, "", reference_image_bytes=main_image_bytes)
            path = OUTFITS_DIR / f"{outfit_id}_{key}.png"
            with open(path, "wb") as f:
                f.write(img_bytes)
            paths[key] = str(path)
        except Exception as e:
            print(f"Failed to generate {key}: {e}")

    return paths


# =========================================================
# OUTFIT GENERATION
# =========================================================

def ask_style_mode():
    return choose_from_list("Select style mode:", ["New Style", "Saved Preference"])


def ask_source():
    return choose_from_list("Select outfit source:", ["Wardrobe", "Outside"])


def build_breakdown_from_prompt(user_prompt, source_items=None):
    text = user_prompt.lower()

    breakdown = {
        "top": "a suitable top",
        "bottom": "a suitable bottom",
        "shoes": "appropriate shoes",
        "outerwear": "no outerwear if not needed",
        "accessories": "minimal accessories if needed",
    }

    if "shirt" in text or "tee" in text or "t-shirt" in text or "tshirt" in text:
        breakdown["top"] = "a shirt or t-shirt"
    if "blouse" in text:
        breakdown["top"] = "a blouse"
    if "hoodie" in text:
        breakdown["top"] = "a hoodie"

    if "pant" in text or "trouser" in text or "jean" in text or "jeans" in text:
        breakdown["bottom"] = "pants or jeans"
    if "skirt" in text:
        breakdown["bottom"] = "a skirt"
    if "short" in text:
        breakdown["bottom"] = "shorts"

    if "sneaker" in text:
        breakdown["shoes"] = "sneakers"
    if "boot" in text:
        breakdown["shoes"] = "boots"
    if "heel" in text:
        breakdown["shoes"] = "heels"

    if "blazer" in text:
        breakdown["outerwear"] = "a blazer"
    if "jacket" in text:
        breakdown["outerwear"] = "a jacket"
    if "coat" in text:
        breakdown["outerwear"] = "a coat"

    if "watch" in text:
        breakdown["accessories"] = "a watch"
    if "bag" in text:
        breakdown["accessories"] = "a bag"
    if "sunglass" in text:
        breakdown["accessories"] = "sunglasses"
    if "jewelry" in text or "jewellery" in text:
        breakdown["accessories"] = "jewelry"

    if source_items:
        titles = ", ".join([item.get("title", "") for item in source_items if item.get("title")])
        if titles:
            breakdown["source_note"] = f"Wardrobe references: {titles}"

    return breakdown


def generate_outfit():
    print("\n=== GENERATE OUTFIT ===")
    event_category, event = select_event()

    outfit_name = ask_text("Outfit name (optional, press Enter to skip): ", allow_empty=True)
    user_prompt = ask_text("Describe the outfit clearly: ")

    style_mode = ask_style_mode()
    saved_prefs = get_latest_preferences()

    if style_mode == "New Style":
        style = str(saved_prefs.get("vibe", "minimal casual"))
    else:
        style = str(saved_prefs.get("vibe", "minimal casual"))

    source = ask_source()

    wardrobe_items = load_wardrobe_items()
    reference_images = []
    used_wardrobe_items = []

    if source == "Wardrobe":
        matched = find_event_matches(wardrobe_items, event)
        if not matched:
            print("\nNo wardrobe items found for this event.")
            if not ask_yes_no("Do you want to continue with outside generation using the prompt only?"):
                print("Stopped.")
                return
            source = "Outside"
        else:
            ref_images, used_wardrobe_items = build_wardrobe_refs(matched, event, user_prompt, saved_prefs)
            reference_images = ref_images
            print("\nMatched wardrobe items:")
            for item in used_wardrobe_items:
                print(f"- {item.get('category')}: {item.get('title')}")

            needed_roles = []
            present_roles = {item.get("category") for item in used_wardrobe_items}
            for role in ["Tops", "Bottoms", "Shoes"]:
                if role not in present_roles:
                    needed_roles.append(role)

            if needed_roles:
                print(f"\nMissing required items: {', '.join(needed_roles)}")
                if not ask_yes_no("Can I generate the missing item(s) too?"):
                    print("Stopped.")
                    return
            # We still generate the full outfit image, but wardrobe items are used as references.

    data = {
        "gender": saved_prefs.get("gender", "neutral"),
        "event": event,
        "style": style,
        "fit": saved_prefs.get("fit", "Regular"),
        "price_level": saved_prefs.get("price_level", "Mid-Range"),
        "colors": saved_prefs.get("colors", "Neutrals"),
        "user_prompt": user_prompt,
    }

    breakdown = build_breakdown_from_prompt(user_prompt, used_wardrobe_items if source == "Wardrobe" else None)
    base_prompt = build_base_prompt(data, breakdown, wardrobe_note=breakdown.get("source_note") if source == "Wardrobe" else None)

    llm = load_llm()
    final_prompt = refine_prompt_with_llm(llm, base_prompt)

    print("\n--- FINAL IMAGE PROMPT ---")
    print(final_prompt)
    print("--------------------------")

    image_client = load_image_client()
    try:
        main_bytes = generate_main_image(
            image_client,
            final_prompt,
            used_wardrobe_items if source == "Wardrobe" else None
        )
    except Exception as e:
        print(f"\nImage generation failed: {e}")
        return

    outfit_id = f"outfit_{now_stamp()}_{slugify(outfit_name or event)}"
    main_image_path = OUTFITS_DIR / f"{outfit_id}_main.png"
    save_image_bytes(main_bytes, main_image_path)

    # Generate individual breakdown item images (product shots).
    # We pass the main image bytes as a reference so each item matches the generated main image.
    try:
        breakdown_image_paths = generate_breakdown_images(image_client, breakdown, outfit_id, main_image_bytes=main_bytes)
    except Exception as e:
        print(f"Failed to generate breakdown images: {e}")
        breakdown_image_paths = {}

    outfit_record = {
        "id": outfit_id,
        "outfit_name": outfit_name or None,
        "event_category": event_category,
        "event": event,
        "prompt": user_prompt,
        "final_image_prompt": final_prompt,
        "style_mode": style_mode,
        "source": source,
        "main_image_path": str(main_image_path),
        "breakdown_items": breakdown,
        "breakdown_images": breakdown_image_paths,
        "wardrobe_references": used_wardrobe_items,
        "created_at": now_stamp(),
    }

    outfits = load_json(OUTFITS_FILE, [])
    outfits.append(outfit_record)
    save_json(OUTFITS_FILE, outfits)

    print("\nOutfit generated and saved.")
    print(f"Main image: {main_image_path}")
    print(f"Breakdown images: {breakdown_image_paths}")

    while True:
        print("\nPost-output options:")
        print("1. Regenerate")
        print("2. Save & Exit")
        print("3. Generate Another Outfit")
        choice = input("Choose option: ").strip()

        if choice == "1":
            changes = ask_text("What changes do you need in the outfit? ")
            regen_data = dict(data)
            regen_data["user_prompt"] = f"{user_prompt}. Changes requested: {changes}"
            regen_breakdown = build_breakdown_from_prompt(regen_data["user_prompt"], used_wardrobe_items if source == "Wardrobe" else None)
            regen_base_prompt = build_base_prompt(
                regen_data,
                regen_breakdown,
                wardrobe_note=regen_breakdown.get("source_note") if source == "Wardrobe" else None,
            )
            regen_prompt = refine_prompt_with_llm(llm, regen_base_prompt)

            try:
                ref_bytes = None
                if Path(main_image_path).exists():
                    with open(main_image_path, "rb") as f:
                        ref_bytes = f.read()

                inputs = [{"type": "text", "text": regen_prompt}]
                if ref_bytes:
                    inputs.append({
                        "type": "image",
                        "data": base64.b64encode(ref_bytes).decode("utf-8"),
                        "mime_type": "image/png",
                    })
                for ref in reference_images:
                    ref_path = Path(ref)
                    if ref_path.exists():
                        inputs.append({
                            "type": "image",
                            "data": image_to_b64(ref_path),
                            "mime_type": guess_mime(ref_path),
                        })

                interaction = image_client.interactions.create(
                    model=IMAGE_MODEL,
                    input=inputs,
                )
                regen_bytes = base64.b64decode(interaction.output_image.data)

                regen_id = f"{outfit_id}_v2"
                regen_main_path = OUTFITS_DIR / f"{regen_id}_main.png"

                save_image_bytes(regen_bytes, regen_main_path)

                # generate per-item breakdown images for the regenerated outfit
                try:
                    regen_breakdown_paths = generate_breakdown_images(image_client, regen_breakdown, regen_id, main_image_bytes=regen_bytes)
                except Exception as e:
                    print(f"Failed to generate regen breakdown images: {e}")
                    regen_breakdown_paths = {}

                regen_record = dict(outfit_record)
                regen_record.update({
                    "id": regen_id,
                    "prompt": regen_data["user_prompt"],
                    "final_image_prompt": regen_prompt,
                    "main_image_path": str(regen_main_path),
                    "breakdown_items": regen_breakdown,
                    "breakdown_images": regen_breakdown_paths,
                    "created_at": now_stamp(),
                    "regenerated_from": outfit_id,
                    "regeneration_note": changes,
                })

                outfits = load_json(OUTFITS_FILE, [])
                outfits.append(regen_record)
                save_json(OUTFITS_FILE, outfits)

                print("\nRegenerated outfit saved.")
                print(f"Main image: {regen_main_path}")
                print(f"Breakdown images: {regen_breakdown_paths}")
                return
            except Exception as e:
                print(f"Regeneration failed: {e}")
                return

        if choice == "2":
            return

        if choice == "3":
            generate_outfit()
            return

        print("Invalid choice.")


# =========================================================
# MAIN LOOP
# =========================================================

def main():
    ensure_dirs()

    while True:
        print("\n==============================")
        print("AI OUTFIT GENERATOR")
        print("==============================")
        print("1. Add Preferences")
        print("2. Generate Outfit")
        print("3. Add Wardrobe")
        print("4. Exit")

        choice = input("Choose an option: ").strip()

        if choice == "1":
            add_preferences()
        elif choice == "2":
            generate_outfit()
        elif choice == "3":
            add_wardrobe()
        elif choice == "4":
            print("Goodbye.")
            break
        else:
            print("Invalid choice. Try again.")


if __name__ == "__main__":
    main()