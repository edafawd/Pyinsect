import json
import urllib.request
from pathlib import Path
import random
import shutil
from time import sleep

# ========== SETTINGS ==========
IMAGES_PER_INSECT = 30          # images per insect (keep this low to stay safe)
DELAY_BETWEEN_INSECTS = 3       # seconds pause between each insect
DELAY_BETWEEN_IMAGES = 0.6      # seconds pause between each photo
# ==============================

# List of 100 insects (common + some invasive ones)
insects = [
    "japanese beetle", "spotted lanternfly", "brown marmorated stink bug", "emerald ash borer",
    "asian longhorned beetle", "spongy moth", "monarch butterfly", "honey bee",
    "bumblebee", "ladybug", "praying mantis", "dragonfly", "damselfly",
    "crane fly", "mosquito", "house fly", "horse fly", "robber fly",
    "hoverfly", "paper wasp", "yellowjacket", "hornet", "carpenter ant",
    "fire ant", "black ant", "termite", "cockroach", "cricket",
    "grasshopper", "katydid", "cicada", "aphid", "leafhopper",
    "stink bug", "assassin bug", "bed bug", "water strider", "backswimmer",
    "diving beetle", "ground beetle", "tiger beetle", "click beetle", "firefly",
    "june beetle", "rhinoceros beetle", "stag beetle", "weevil", "longhorn beetle",
    "blister beetle", "soldier beetle", "carpet beetle", "dermestid beetle", "mealworm",
    "swallowtail butterfly", "cabbage white", "painted lady", "red admiral", "mourning cloak",
    "fritillary", "hairstreak", "skipper butterfly", "luna moth", "polyphemus moth",
    "io moth", "cecropia moth", "hawk moth", "sphinx moth", "tiger moth",
    "underwing moth", "inchworm", "tent caterpillar", "bagworm", "cutworm",
    "army worm", "corn earworm", "codling moth", "diamondback moth", "clothes moth",
    "silverfish", "firebrat", "earwig", "booklouse", "thrips",
    "lacewing", "antlion", "dobsonfly", "fishfly", "scorpionfly",
    "hangingfly", "caddisfly", "mayfly", "stonefly", "springtail",
    "bristletail", "webspinner", "zorapteran", "stick insect", "leaf insect"
]

headers = {"User-Agent": "InsectDownloader/1.0 (personal project)"}
temp_base = Path("temp_insects")
temp_base.mkdir(exist_ok=True)

print(f"Starting download for {len(insects)} insects...\n")

for i, insect in enumerate(insects, 1):
    print(f"[{i}/{len(insects)}] {insect}")

    folder_name = insect.lower().replace(" ", "_")
    temp_dir = temp_base / folder_name
    temp_dir.mkdir(exist_ok=True)

    api_url = (
        f"https://api.inaturalist.org/v1/observations"
        f"?taxon_name={insect.replace(' ', '%20')}"
        f"&photos=true&per_page=200&quality_grade=research"
    )

    try:
        req = urllib.request.Request(api_url, headers=headers)
        with urllib.request.urlopen(req, timeout=30) as response:
            data = json.loads(response.read().decode())
            results = data.get("results", [])

            count = 0
            for obs in results:
                if count >= IMAGES_PER_INSECT:
                    break

                photos = obs.get("photos", [])
                if not photos:
                    continue

                photo_url = photos[0].get("url", "").replace("square", "medium")
                if not photo_url:
                    continue

                try:
                    img_req = urllib.request.Request(photo_url, headers=headers)
                    with urllib.request.urlopen(img_req, timeout=20) as img_resp:
                        filename = temp_dir / f"{folder_name}_{count+1}.jpg"
                        with open(filename, "wb") as f:
                            f.write(img_resp.read())
                        count += 1
                        print(f"   ✓ {count}/{IMAGES_PER_INSECT}")
                except Exception:
                    pass

                sleep(DELAY_BETWEEN_IMAGES)

        # Organize into train / val
        all_images = list(temp_dir.glob("*.jpg"))
        if len(all_images) < 5:
            print(f"   ⚠ Only {len(all_images)} images — skipped")
            continue

        random.shuffle(all_images)
        split = int(len(all_images) * 0.8)

        train_dir = Path("data/train") / folder_name
        val_dir = Path("data/val") / folder_name
        train_dir.mkdir(parents=True, exist_ok=True)
        val_dir.mkdir(parents=True, exist_ok=True)

        for img in all_images[:split]:
            shutil.copy(img, train_dir / img.name)
        for img in all_images[split:]:
            shutil.copy(img, val_dir / img.name)

        print(f"   → Saved {split} train + {len(all_images)-split} val")

    except Exception as e:
        print(f"   ✗ Error: {e}")

    sleep(DELAY_BETWEEN_INSECTS)

# Cleanup
if temp_base.exists():
    shutil.rmtree(temp_base)

print("\nFinished! Check the 'data' folder.")
