"""Training data for the detector (Phase 1).

    sources.py   the public datasets we use: where they come from, licence, checksum
    readers.py   reading YOLO- and COCO-format datasets into one common form
    labels.py    mapping each dataset's label names onto person / helmet / vest
    dedup.py     finding near-duplicate images (so no test image has a twin in training)
    build.py     merging everything into one YOLO dataset, with a report
"""
