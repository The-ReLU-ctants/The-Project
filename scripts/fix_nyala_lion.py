"""One-off fix: wildlife_datasets crashes writing a None license URL for
NyalaData/LionData after a successful download, so extraction never runs.
Catch that specific bug and finish extraction manually.
"""
import wildlife_datasets.datasets as d

for cls, root in [(d.NyalaData, "data/raw_subset/nyala"), (d.LionData, "data/raw_subset/lion")]:
    print(f"=== {cls.__name__} -> {root} ===")
    try:
        cls.download(root)
    except TypeError as e:
        print(f"  (expected license-write bug, continuing): {e}")
    cls.extract(root)
    print(f"  {cls.__name__} done.")
