"""
We start by first downloading CloudSen12 dataset in taco format.
"""
from huggingface_hub import hf_hub_download
from os import path
import tacoreader.v1 as tacoreader

if not path.exists("cloudsen12-l1c.0000.part.taco"):
    dataset1 = hf_hub_download("tacofoundation/CloudSEN12", "cloudsen12-l1c.0000.part.taco", repo_type="dataset", local_dir=".")
    dataset2 = hf_hub_download("tacofoundation/CloudSEN12", "cloudsen12-l1c.0001.part.taco", repo_type="dataset", local_dir=".")

"""
But this might be 1TB file and we are only interested on images
where cloud % is greater than 0. For that we need to create a minitaco
or a tortilla dataset.
"""

import geopandas as gpd
import pandas as pd

# 1. Load the taco dataset
# HINT: Every TACO dataset is a GeoDataFrame if it fullfill stac requirements
dataset = tacoreader.load(["cloudsen12-l1c.0000.part.taco", "cloudsen12-l1c.0001.part.taco"])

# 2. Spatial Query [Only Switzerland]
eligible_countries = ["Switzerland", "Austria", "New Zealand","Canada", "Norway", "Chile"]
subset_sp = dataset[dataset["rai:admin0"].isin(eligible_countries)]

# 3. Temporal Query [Only 2022]
# years = pd.to_datetime(subset_sp["stac:time_start"], unit='s').dt.year
# subset_sp_temporal = subset_sp[years == 2020]

# 4. Filter images that contain cloud shadows
subset_final = subset_sp[subset_sp["cloud_shadow_percentage"] > 0]
print(subset_final.plot())

# 5. Create a new TACO file based on the previous filters
tacoreader.compile(dataframe=subset_final, output="mini.taco", nworkers=4)

# 6. Load your new TACO file
# minitaco = tacoreader.load("mini.taco")

# Final comments: Actually this is a Tortilla files since it does not have COLLECTION properties
# If you want to convert it to TACO, use tacotoolbox.tortilla2taco.
# Read more about TORTILLA here: https://tacofoundation.github.io/specification/tortilla
# Read more about TACO here: https://tacofoundation.github.io/specification/taco
# Sorry we are still in very early alpha, no clean documentation, but we are working hard to make TACO better!