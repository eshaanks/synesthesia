from icrawler.builtin import BingImageCrawler
import os
import time

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'full_dataset')

new_categories = {
    '04b_coastal_haze': [
        'sea mist harbour morning atmospheric',
        'coastal fog lighthouse moody',
        'ocean haze morning light shore',
        'beach fog early morning calm',
    ],
    '05b_lava_ember': [
        'lava flow volcanic texture close',
        'fire embers glow dark macro',
        'volcano lava close night orange',
        'ember glow fire dark abstract',
    ],
    '07b_moss_ground': [
        'moss covered rock macro detail',
        'forest floor mushroom detail macro',
        'lichen stone texture macro close',
        'fallen leaves forest floor macro',
    ],
    '08b_light_through': [
        'sunbeam cathedral window light rays',
        'light shaft dust particles indoor',
        'rays light through window blinds',
        'dappled light shadow pattern ground',
    ],
    '09b_aged_surface': [
        'old book page texture macro close',
        'aged wood grain texture detail',
        'vintage paper texture macro brown',
        'old wall plaster texture decay',
    ],
    '11b_organic_texture': [
        'tree bark texture macro close detail',
        'snake scale texture macro close',
        'feather detail macro close up',
        'leaf vein texture macro green',
    ],
    '12b_mineral_texture': [
        'granite mineral texture macro close',
        'sandstone texture close detail',
        'marble texture macro abstract pattern',
        'geode crystal interior macro detail',
    ],
    '14b_puddle_reflect': [
        'puddle reflection city street night',
        'wet pavement reflection night lights',
        'rain puddle sky reflection abstract',
        'window glass reflection abstract city',
    ],
    '01b_aurora': [
        'aurora borealis night sky green',
        'northern lights green sky landscape',
        'aurora reflection lake night calm',
        'milky way stars wide landscape night',
    ],
    '02b_shadow_figure': [
        'person shadow long street minimal',
        'silhouette window light indoor dramatic',
        'figure shadow wall abstract light',
        'lone shadow empty street minimal',
    ],
    '06b_snow_detail': [
        'snow crystal branch macro white',
        'fresh snow texture macro surface',
        'snowflake macro detail white close',
        'snow surface texture close detail',
    ],
    '10b_rain_surface': [
        'rain hitting water surface macro',
        'puddle rain ripple close up',
        'raindrops glass window macro detail',
        'water drops surface impact macro',
    ],
}

IMAGES_PER_QUERY = 10

for category, queries in new_categories.items():
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
                min_size=(1280, 720),
                file_idx_offset='auto',
            )
            print(f"  done: {query}")
            time.sleep(1)
        except Exception as e:
            print(f"  failed: {query} — {e}")

    count = len([f for f in os.listdir(cat_dir)
                 if f.lower().endswith(('.jpg','.jpeg','.png'))])
    print(f"  [{category}] total: {count} images")

print("\nall done — full dataset counts:")
total = 0
for d in sorted(os.listdir(BASE)):
    path = os.path.join(BASE, d)
    if os.path.isdir(path):
        count = len([f for f in os.listdir(path)
                     if f.lower().endswith(('.jpg','.jpeg','.png'))])
        total += count
        status = "✓" if count >= 25 else "⚠ LOW"
        print(f"  {d}: {count} {status}")
print(f"\n  TOTAL: {total}")
