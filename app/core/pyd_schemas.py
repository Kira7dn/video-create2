from __future__ import annotations

from enum import Enum
from typing import List, Optional, Literal

from pydantic import BaseModel, conlist, constr


class TransitionType(str, Enum):
    fade = "fade"
    fadeblack = "fadeblack"
    fadewhite = "fadewhite"
    cut = "cut"


class Media(BaseModel):
    url: constr(strip_whitespace=True, min_length=1)
    start_delay: Optional[float] = None
    end_delay: Optional[float] = None
    local_path: Optional[str] = None


class VoiceOver(Media):
    content: Optional[str] = None


class Image(Media):
    # Allow empty string for image URL specifically
    url: constr(strip_whitespace=True, min_length=0)


class Transition(BaseModel):
    type: Optional[TransitionType] = None
    duration: Optional[float] = None


class BackgroundMusic(Media):
    url: str = None


class Segment(BaseModel):
    id: constr(strip_whitespace=True, min_length=1)
    voice_over: Optional[VoiceOver] = None
    image: Optional[Image] = None
    video: Optional[Media] = None
    transition_in: Optional[Transition] = None
    transition_out: Optional[Transition] = None


class InputPayload(BaseModel):
    niche: Optional[str] = None
    keywords: Optional[List[str]] = None
    title: Optional[str] = None
    description: Optional[str] = None
    segments: conlist(Segment, min_length=1)
    background_music: Optional[BackgroundMusic] = None
    video_type: Optional[Literal["short", "long"]] = None
