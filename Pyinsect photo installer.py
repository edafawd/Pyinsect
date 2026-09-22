import json
import random
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

# ========== 200 INSECTS ==========
INSECTS = [
    "ambrosia beetle",
    "american cockroach",
    "american lady",
    "anise swallowtail",
    "antlion",
    "aphid",
    "argentine ant",
    "army worm",
    "asian giant hornet",
    "asian lady beetle",
    "asian longhorned beetle",
    "assassin bug",
    "backswimmer",
    "bagworm",
    "bald faced hornet",
    "balsam woolly adelgid",
    "bark beetle",
    "bean leaf beetle",
    "bed bug",
    "bee fly",
    "black ant",
    "black fly",
    "black swallowtail",
    "blister beetle",
    "blow fly",
    "boll weevil",
    "booklouse",
    "boxelder bug",
    "bronze birch borer",
    "brown banded cockroach",
    "brown lacewing",
    "brown marmorated stink bug",
    "buckeye butterfly",
    "bumblebee",
    "cabbage white",
    "caddisfly",
    "carpenter ant",
    "carpenter bee",
    "carpet beetle",
    "carrion beetle",
    "cecropia moth",
    "cereal leaf beetle",
    "cicada",
    "cicada killer",
    "clearwing moth",
    "click beetle",
    "clothes moth",
    "clouded sulphur",
    "cockroach",
    "codling moth",
    "colorado potato beetle",
    "comma butterfly",
    "common ringlet",
    "convergent lady beetle",
    "corn earworm",
    "crane fly",
    "cricket",
    "cucumber beetle",
    "cutworm",
    "damsel bug",
    "damselfly",
    "darkling beetle",
    "deer fly",
    "diamondback moth",
    "diving beetle",
    "dobsonfly",
    "douglas fir beetle",
    "douglas fir tussock moth",
    "dragonfly",
    "dung beetle",
    "earwig",
    "elm leaf beetle",
    "emerald ash borer",
    "european paper wasp",
    "fall webworm",
    "field cricket",
    "fire ant",
    "firebrat",
    "firefly",
    "fishfly",
    "flea beetle",
    "flesh fly",
    "forest tent caterpillar",
    "formosan termite",
    "fritillary",
    "fruit fly",
    "fungus gnat",
    "gall wasp",
    "german cockroach",
    "giant swallowtail",
    "giant water bug",
    "grasshopper",
    "gray hairstreak",
    "green lacewing",
    "green june beetle",
    "ground beetle",
    "hairstreak",
    "hangingfly",
    "harlequin bug",
    "harvester ant",
    "hawk moth",
    "hemlock woolly adelgid",
    "honey bee",
    "hornet",
    "horse fly",
    "house fly",
    "hoverfly",
    "hummingbird moth",
    "imperial moth",
    "inchworm",
    "io moth",
    "japanese beetle",
    "jerusalem cricket",
    "june beetle",
    "katydid",
    "lace bug",
    "lacewing",
    "ladybug",
    "leaf insect",
    "leafcutter bee",
    "leafhopper",
    "locust borer",
    "longhorn beetle",
    "luna moth",
    "mason bee",
    "mayfly",
    "mealybug",
    "mealworm",
    "mexican bean beetle",
    "midge",
    "migratory grasshopper",
    "milkweed bug",
    "mole cricket",
    "monarch butterfly",
    "mosquito",
    "mountain pine beetle",
    "mourning cloak",
    "mud dauber",
    "oriental beetle",
    "oriental cockroach",
    "painted lady",
    "paper wasp",
    "pavement ant",
    "periodical cicada",
    "pine sawfly",
    "pipevine swallowtail",
    "plant bug",
    "polyphemus moth",
    "powderpost beetle",
    "praying mantis",
    "question mark butterfly",
    "red admiral",
    "red imported fire ant",
    "rhinoceros beetle",
    "rice weevil",
    "robber fly",
    "rose chafer",
    "rove beetle",
    "sawfly",
    "scale insect",
    "scorpionfly",
    "silverfish",
    "silver spotted skipper",
    "skipper butterfly",
    "soldier beetle",
    "soldier fly",
    "sphinx moth",
    "spider wasp",
    "spittlebug",
    "spongy moth",
    "spotted lanternfly",
    "springtail",
    "spruce beetle",
    "squash bug",
    "stag beetle",
    "stick insect",
    "stink bug",
    "stonefly",
    "swallowtail butterfly",
    "sweat bee",
    "tachinid fly",
    "tent caterpillar",
    "termite",
    "thrips",
    "tiger beetle",
    "tiger moth",
    "tomato hornworm",
    "treehopper",
    "tussock moth",
    "underwing moth",
    "velvet ant",
    "viceroy butterfly",
    "walking stick",
    "water boatman",
    "water strider",
    "weevil",
    "western pine beetle",
    "western tiger swallowtail",
    "whitefly",
    "yellowjacket"
]

IMAGES_PER_INSECT = 400
MAX_API_PAGES = 5          # 5 x 200 observations = 1000 IDs max
TRAIN_RATIO = 0.8
PHOTO_SIZE = "medium"
SLEEP_API = 1.0
SLEEP_S3 = 0.08
# ==============================

HEADERS = {"User-Agent": "PyinsectTrainer/1.0 (personal ML project)"}
DATA = Path("data")

def get_json(url):
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=40) as r:
        return json.loads(r.read().decode())

def folder_name(name):
    return name.lower().replace(" ", "_")

def aws_photo_urls(photo_id, ext):
    ext = (ext or "jpg").lstrip(".").lower()
    exts = ["jpeg", "jpg"] if ext == "jpeg" else [ext, "jpg", "jpeg"]
    urls = []
    for e in exts:
        u = f"https://inaturalist-open-data.s3.amazonaws.com/photos/{photo_id}/{PHOTO_SIZE}.{e}"
        if u not in urls:
            urls.append(u)
    return urls

def download_file(url, dest):
    req = urllib.request.Request(url, headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            data = r.read()
        if len(data) < 2000:
            return False
        dest.write_bytes(data)
        return True
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return False
        raise

def resolve_taxon(name):
    q = urllib.parse.quote(name)
    url = f"https://api.inaturalist.org/v1/taxa/autocomplete?q={q}&is_active=true"
    results = get_json(url).get("results", [])
    if not results:
        return None
    for t in results:
        if t.get("name", "").lower() == name.lower() or t.get("preferred_common_name", "").lower() == name.lower():
            return t
    return results[0]

def fetch_photo_ids(taxon_id, need):
    photos = []
    seen = set()
    for page in range(1, MAX_API_PAGES + 1):
        url = (
            "https://api.inaturalist.org/v1/observations"
            f"?taxon_id={taxon_id}"
            "&photos=true"
            "&quality_grade=research"
            "&photo_license=cc-by,cc-by-nc,cc-by-sa,cc-by-nd,cc-by-nc-sa,cc-by-nc-nd,cc0"
            "&per_page=200"
            f"&page={page}"
        )
        print(f"    API page {page}...")
        results = get_json(url).get("results", [])
        time.sleep(SLEEP_API)
        if not results:
            break
        for o in results:
            # 1 photo per observation, less duplicate junk
            p = (o.get("photos") or [None])[0]
            if not p:
                continue
            pid = p.get("id")
            if not pid or pid in seen:
                continue
            img_url = p.get("url") or ""
            if "static.inaturalist.org" in img_url and "inaturalist-open-data" not in img_url:
                continue
            seen.add(pid)
            ext = img_url.rsplit(".", 1)[-1].split("?")[0] if img_url else "jpg"
            photos.append((pid, ext))
            if len(photos) >= need:
                return photos
    return photos

print(f"100 insects x {IMAGES_PER_INSECT} photos")
print("Photos from AWS. API only lists IDs.\n")

for n, insect in enumerate(INSECTS, 1):
    print("=" * 50)
    print(f"[{n}/{len(INSECTS)}] {insect}")
    try:
        taxon = resolve_taxon(insect)
        time.sleep(SLEEP_API)
        if not taxon:
            print("  no taxon, skip")
            continue

        taxon_id = taxon["id"]
        print(f"  taxon_id={taxon_id}  {taxon.get('name')}")

        photos = fetch_photo_ids(taxon_id, IMAGES_PER_INSECT)
        print(f"  photo IDs: {len(photos)}")

        temp = Path("temp_aws") / folder_name(insect)
        temp.mkdir(parents=True, exist_ok=True)
        saved = []

        for pid, ext in photos:
            if len(saved) >= IMAGES_PER_INSECT:
                break
            dest = temp / f"{folder_name(insect)}_{len(saved)+1}.jpg"
            ok = False
            for url in aws_photo_urls(pid, ext):
                if download_file(url, dest):
                    ok = True
                    break
            if ok:
                saved.append(dest)
                if len(saved) % 25 == 0 or len(saved) == IMAGES_PER_INSECT:
                    print(f"  AWS {len(saved)}/{IMAGES_PER_INSECT}")
            time.sleep(SLEEP_S3)

        if len(saved) < 10:
            print("  too few images, skip")
            continue

        random.shuffle(saved)
        split = int(len(saved) * TRAIN_RATIO)
        cls = folder_name(insect)
        train_dir = DATA / "train" / cls
        val_dir = DATA / "val" / cls
        train_dir.mkdir(parents=True, exist_ok=True)
        val_dir.mkdir(parents=True, exist_ok=True)

        for p in saved[:split]:
            shutil.copy(p, train_dir / p.name)
        for p in saved[split:]:
            shutil.copy(p, val_dir / p.name)

        print(f"  DONE train={split}  val={len(saved)-split}")

    except Exception as e:
        print(f"  error: {e}")
        time.sleep(3)

if Path("temp_aws").exists():
    shutil.rmtree("temp_aws")

print("\nAll done. Check data/train and data/val")
