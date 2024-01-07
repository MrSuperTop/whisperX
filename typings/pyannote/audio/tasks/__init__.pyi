from .embedding.arcface import SupervisedRepresentationLearningWithArcFace
from .segmentation.multilabel import MultiLabelSegmentation as MultiLabelSegmentation
from .segmentation.overlapped_speech_detection import (
    OverlappedSpeechDetection as OverlappedSpeechDetection,
)
from .segmentation.speaker_diarization import SpeakerDiarization as SpeakerDiarization
from .segmentation.voice_activity_detection import (
    VoiceActivityDetection as VoiceActivityDetection,
)

Segmentation = SpeakerDiarization
SpeakerEmbedding = SupervisedRepresentationLearningWithArcFace
