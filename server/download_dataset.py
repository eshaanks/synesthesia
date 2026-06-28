from icrawler.builtin import BingImageCrawler
import os
import time

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'full_dataset')

categories = {
    '01_vast_sky': [
        'enormous open sky clouds minimal landscape',
        'wide sky horizon empty prairie photography',
        'dramatic cloudscape sky fill frame',
        'open sky wide angle minimal ground',
    ],
    '02_lone_figure': [
        'lone person silhouette vast mountain',
        'single figure fog empty landscape',
        'solitary human desert wide shot',
        'lone figure ocean shore minimalist',
    ],
    '03_dark_water': [
        'dark still lake surface night',
        'deep dark ocean aerial looking down',
        'black calm river surface moody',
        'dark fjord water Norway atmospheric',
    ],
    '04_mist_fog': [
        'thick morning fog forest valley',
        'dense mist mountain landscape',
        'fog rolling hills soft diffuse light',
        'misty lake dawn atmospheric',
    ],
    '05_fire_flame': [
        'close up flame fire detail orange macro',
        'campfire flame macro warm dark background',
        'wildfire flames close texture detail',
        'candle flame macro isolated dark',
    ],
    '06_ice_frost': [
        'frost crystal macro ice detail close',
        'frozen lake ice texture close up',
        'ice crystal formation blue white macro',
        'frost pattern window glass macro',
    ],
    '07_root_soil': [
        'tree roots soil texture close up macro',
        'underground roots earth detail',
        'bare roots exposed ground photography',
        'soil earth texture macro close',
    ],
    '08_canopy_above': [
        'looking up forest canopy sunlight rays',
        'tree canopy from below green light',
        'forest sunbeam through trees looking up',
        'bamboo canopy straight up photography',
    ],
    '09_rust_decay': [
        'rust metal texture close up macro',
        'peeling paint weathered surface detail',
        'oxidized metal surface macro photography',
        'decay weathered wood texture close',
    ],
    '10_water_surface': [
        'water surface light refraction close up',
        'looking below water surface light rays',
        'water surface ripple macro detail',
        'pool water surface light caustics',
    ],
    '11_skin_fabric': [
        'fabric weave thread macro close up',
        'textile texture macro photography',
        'woven fabric detail close up macro',
        'linen cotton texture extreme macro',
    ],
    '12_stone_crack': [
        'cracked dry earth texture aerial',
        'rock face texture close up detail',
        'stone crack fracture pattern macro',
        'dry cracked mud desert texture',
    ],
    '13_wave_crash': [
        'wave crashing shore impact white water',
        'ocean wave moment of impact photography',
        'wave crash rocks dramatic water force',
        'shore wave break aerial photography',
    ],
    '14_reflection': [
        'perfect mirror lake reflection symmetry',
        'still water mountain reflection dawn',
        'tree reflection calm water symmetry',
        'perfect reflection lake minimal',
    ],
    '15_murmuration': [
        'starling murmuration pattern sky',
        'birds flock formation aerial pattern',
        'murmuration starlings dusk sky',
        'bird flock sky movement pattern',
    ],
    '16_city_night': [
        'city lights night aerial photography',
        'city street night warm light',
        'urban night lights bokeh cityscape',
        'city aerial night glow warm',
    ],
}

# 4 queries x 32 images = ~128 per category attempt
# with failures expected ~30 clean per category = ~480 total
IMAGES_PER_QUERY = 32

for category, queries in categories.items():
    cat_dir = os.path.join(BASE, category)
    os.makedirs(cat_dir, exist_ok=True)
    print(f"\n[{category}] downloading...")
    for query in queries:
        try:
            crawler = BingImageCrawler(
                storage={'root_dir': cat_dir},
                feeder_threads=1,
                parser_threads=1,
                downloader_threads=3,
            )
            crawler.crawl(
                keyword=query,
                max_num=IMAGES_PER_QUERY,
                min_size=(1920, 1080),
                file_idx_offset='auto',
            )
            print(f"  done: {query}")
            time.sleep(2)
        except Exception as e:
            print(f"  failed: {query} — {e}")
    count = len([f for f in os.listdir(cat_dir) if f.endswith(('.jpg','.jpeg','.png'))])
    print(f"  [{category}] total so far: {count} images")

print("\nall done — final counts:")
for category in categories:
    cat_dir = os.path.join(BASE, category)
    count = len([f for f in os.listdir(cat_dir) if f.endswith(('.jpg','.jpeg','.png'))])
    print(f"  {category}: {count}")
