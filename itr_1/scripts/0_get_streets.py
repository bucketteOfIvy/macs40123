import overpy
import pandas as pd
import time

# Query only through secondary roadways. More than this is bad.
overpass_query = \
"""
[out:json][timeout:25];
area['name'='City of New York']->.searchArea;
(
  way[highway=motorway](area.searchArea);
  way[highway=trunk](area.searchArea);
  way[highway=primary](area.searchArea);
  way[highway=secondary](area.searchArea);
  way[highway=tertiary](area.searchArea);
//  way[highway=residential](area.searchArea);
//    way[highway](area.searchArea);
);
out body;
>;
out skel qt;
"""
# Make Overpass. Do Query
t0 = time.time()
print("Starting query.")
api = overpy.Overpass()
query = api.query(overpass_query)
print(f'Finished query. Time elapsed: {time.time() - t0}')

t0 = time.time()
print("Beginning flattening")
indices = []
indices_words = []
for way in query.ways:
  tags = way.tags
  words = []
  for k, v in tags.items():
    words.append(f"{k}|{v}")
  indices.append(way.id)
  indices_words.append(words)

pd.DataFrame({'way_id':indices, 'words':indices_words}).to_csv('../data/osm_tags_per_way.csv')



