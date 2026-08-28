"""
Maps YOLO class names (from data.yaml's `names` list) to the MDP arena's
official Image ID scheme (11-40), used by the RPi / algorithm / Android app.

The YOLO class *index* is not stable across re-exports/re-training (Roboflow
re-alphabetizes `names` on every version bump), so the mapping is keyed by
class *name* instead, looked up via model.names[cls_idx] at inference time.
"""

IMAGE_ID = {
    "one": 11,
    "two": 12,
    "three": 13,
    "four": 14,
    "five": 15,
    "six": 16,
    "seven": 17,
    "eight": 18,
    "nine": 19,
    "A": 20,
    "B": 21,
    "C": 22,
    "D": 23,
    "E": 24,
    "F": 25,
    "G": 26,
    "H": 27,
    "S": 28,
    "T": 29,
    "U": 30,
    "V": 31,
    "W": 32,
    "X": 33,
    "Y": 34,
    "Z": 35,
    "up": 36,
    "down": 37,
    "right": 38,
    "left": 39,
    "circle": 40,  # "Stop"
    # "Bullseye" is a dataset class with no Image ID in the arena reference
    # chart -- decide how it should be reported (or filtered out) before
    # wiring this into the IR server.
}


def get_image_id(class_name: str) -> int:
    try:
        return IMAGE_ID[class_name]
    except KeyError:
        raise KeyError(
            f"No Image ID mapping for class '{class_name}'. "
            f"Known classes: {sorted(IMAGE_ID)}"
        )
