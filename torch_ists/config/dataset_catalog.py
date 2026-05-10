"""Default dataset catalog for packaged scripts."""

from __future__ import annotations

import copy
from typing import List

_DEFAULT_SELECTED_DATASETS = [
    "ArrowHead",
    "ArticularyWordRecognition",
    "BasicMotions",
    "Car",
    "CharacterTrajectories",
    "Coffee",
    "Cricket",
    "ERing",
    "Epilepsy",
    "EthanolConcentration",
    "EyesOpenShut",
    "FingerMovements",
    "GunPoint",
    "Handwriting",
    "Herring",
    "JapaneseVowels",
    "Libras",
    "Lightning2",
    "Lightning7",
    "LSST",
    "Meat",
    "NATOPS",
    "OliveOil",
    "PickupGestureWiimoteZ",
    "RacketSports",
    "Rock",
    "SelfRegulationSCP1",
    "SelfRegulationSCP2",
    "ShakeGestureWiimoteZ",
    "SmoothSubspace",
    "SpokenArabicDigits",
    "ToeSegmentation1",
    "ToeSegmentation2",
    "Trace",
    "Wine",
]


def get_default_selected_datasets() -> List[str]:
    return copy.deepcopy(_DEFAULT_SELECTED_DATASETS)

