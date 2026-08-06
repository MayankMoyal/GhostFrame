"""Prop Configuration -- Category taxonomy, attachment profiles, and pipeline interface."""
import json
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

class PropCategory(Enum):
    HAND_HELD  = "hand_held"
    SHIELD     = "shield"
    HEAD_WEAR  = "head_wear"
    NECK_WEAR  = "neck_wear"
    WRIST_WEAR = "wrist_wear"
    EAR_WEAR   = "ear_wear"
    FACE_WEAR  = "face_wear"
    BODY_WEAR  = "body_wear"

@dataclass(frozen=True)
class AttachmentProfile:
    category: PropCategory
    pivot_strategy: str
    pivot_default: Tuple[float, float]
    body_scale_ref: str
    prop_scale_ref: str
    scale_multiplier: float
    rotation_mode: str
    auto_orient: str
    z_index: int = 1

PROFILES = {
    PropCategory.HAND_HELD: AttachmentProfile(
        category=PropCategory.HAND_HELD, pivot_strategy="handle_bottom",
        pivot_default=(0.5, 0.95), 
        body_scale_ref="palm_width", prop_scale_ref="handle_width", 
        scale_multiplier=1.0,  # Dynamic scaling based on hand size
        rotation_mode="hand_vector", auto_orient="vertical_handle_down", z_index=1),
    PropCategory.SHIELD: AttachmentProfile(
        category=PropCategory.SHIELD, pivot_strategy="center",
        pivot_default=(0.5, 0.5), body_scale_ref="forearm_length",
        prop_scale_ref="height", scale_multiplier=1.5,
        rotation_mode="forearm_angle", auto_orient="as_is", z_index=1),
    PropCategory.HEAD_WEAR: AttachmentProfile(
        category=PropCategory.HEAD_WEAR, pivot_strategy="bottom_center",
        pivot_default=(0.5, 0.95), body_scale_ref="ear_distance",
        prop_scale_ref="width", scale_multiplier=2.5,
        rotation_mode="head_tilt", auto_orient="as_is", z_index=1),
    PropCategory.NECK_WEAR: AttachmentProfile(
        category=PropCategory.NECK_WEAR, pivot_strategy="top_center",
        pivot_default=(0.5, 0.05), body_scale_ref="shoulder_width",
        prop_scale_ref="width", scale_multiplier=0.7,
        rotation_mode="shoulder_tilt", auto_orient="as_is", z_index=1),
    PropCategory.WRIST_WEAR: AttachmentProfile(
        category=PropCategory.WRIST_WEAR, pivot_strategy="center",
        pivot_default=(0.5, 0.5), body_scale_ref="palm_width",
        prop_scale_ref="width", scale_multiplier=0.8,
        rotation_mode="forearm_angle", auto_orient="as_is", z_index=1),
    PropCategory.EAR_WEAR: AttachmentProfile(
        category=PropCategory.EAR_WEAR, pivot_strategy="top_center",
        pivot_default=(0.5, 0.05), body_scale_ref="ear_eye_distance",
        prop_scale_ref="height", scale_multiplier=1.5,
        rotation_mode="head_tilt", auto_orient="as_is", z_index=1),
    PropCategory.FACE_WEAR: AttachmentProfile(
        category=PropCategory.FACE_WEAR, pivot_strategy="center",
        pivot_default=(0.5, 0.5), body_scale_ref="ear_distance",
        prop_scale_ref="width", scale_multiplier=1.3,
        rotation_mode="head_tilt", auto_orient="as_is", z_index=1),
    PropCategory.BODY_WEAR: AttachmentProfile(
        category=PropCategory.BODY_WEAR, pivot_strategy="top_center",
        pivot_default=(0.5, 0.05), body_scale_ref="torso_height",
        prop_scale_ref="height", scale_multiplier=1.0,
        rotation_mode="shoulder_tilt", auto_orient="as_is", z_index=-1),
}

@dataclass
class PropRequest:
    image_path: str
    category: str = "hand_held"
    prompt: str = ""
    target_side: str = "any"

def load_pipeline_json(json_path: str) -> List[PropRequest]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    requests = []
    for entry in data.get("props", []):
        requests.append(PropRequest(
            image_path=entry["image_path"],
            category=entry.get("category", "hand_held"),
            prompt=entry.get("prompt", ""),
            target_side=entry.get("target_side", "any"),
        ))
    return requests

def resolve_category(category_str: str) -> PropCategory:
    key = category_str.strip().lower()
    for cat in PropCategory:
        if cat.value == key:
            return cat
    _ALIASES = {
        "hand": PropCategory.HAND_HELD, "held": PropCategory.HAND_HELD,
        "head": PropCategory.HEAD_WEAR, "hat": PropCategory.HEAD_WEAR,
        "neck": PropCategory.NECK_WEAR, "necklace": PropCategory.NECK_WEAR,
        "wrist": PropCategory.WRIST_WEAR, "bracelet": PropCategory.WRIST_WEAR,
        "ear": PropCategory.EAR_WEAR, "earring": PropCategory.EAR_WEAR,
        "face": PropCategory.FACE_WEAR, "glasses": PropCategory.FACE_WEAR,
        "body": PropCategory.BODY_WEAR, "cape": PropCategory.BODY_WEAR,
    }
    return _ALIASES.get(key, PropCategory.HAND_HELD)

_CATEGORY_KEYWORDS = {
    PropCategory.BODY_WEAR: ["cape", "cloak", "armor", "armour", "vest", "brooch", "chest plate", "breastplate", "tabard"],
    PropCategory.HEAD_WEAR: ["hat", "helmet", "crown", "tiara", "cap", "headband", "turban", "hood", "beanie", "beret", "headgear", "headpiece"],
    PropCategory.NECK_WEAR: ["necklace", "pendant", "chain", "choker", "scarf", "tie", "necktie", "bow tie", "locket", "collar", "amulet", "medallion"],
    PropCategory.WRIST_WEAR: ["bracelet", "watch", "gauntlet", "wristband", "bangle", "cuff", "wrist guard"],
    PropCategory.EAR_WEAR: ["earring", "ear cuff", "ear ring", "stud", "earbud"],
    PropCategory.FACE_WEAR: ["glasses", "sunglasses", "mask", "monocle", "nose ring", "spectacles", "goggles", "visor", "eye patch"],
    PropCategory.SHIELD: ["shield", "buckler", "kite shield", "tower shield"],
    PropCategory.HAND_HELD: ["sword", "staff", "wand", "axe", "hammer", "mace", "dagger", "knife", "torch", "flag", "spear", "trident", "gun", "pistol", "bow", "scepter", "sceptre", "club", "bat", "fan", "umbrella", "lantern", "orb", "flute", "harp", "lute"],
}

def _word_match(keyword: str, text: str) -> bool:
    import re
    return bool(re.search(r'\b' + re.escape(keyword) + r'\b', text))

def classify_from_prompt(prompt: str) -> PropCategory:
    lower = prompt.lower()
    for category, keywords in _CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if _word_match(kw, lower):
                return category
    return PropCategory.HAND_HELD