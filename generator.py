
#  30 Second Peaceful Classical Piano Generator

import numpy as np
from scipy.io.wavfile import write
from IPython.display import Audio
import math
import random

sample_rate = 44100

# Peaceful tempo longer notes
note_duration = 0.75
measures = 40

# Freq nap
note_freq = {
    "C3":130.81, "D3":146.83, "E3":164.81, "F3":174.61, "G3":196.00,
    "A3":220.00, "B3":246.94,

    "C4":261.63, "D4":293.66, "E4":329.63, "F4":349.23, "G4":392.00,
    "A4":440.00, "B4":493.88,

    "C5":523.25, "D5":587.33, "E5":659.25
}

melody_scale = ["C4","D4","E4","G4","A4","C5","D5","E5"]

left_hand_chords = [
    ["C3","G3","C4"],   # C major open
    ["A3","E4","A4"],   # A minor open
    ["F3","C4","F4"],   # F major open
    ["G3","D4","G4"]    # G major open
]

def synth(freq, dur):
    t = np.linspace(0, dur, int(sample_rate * dur), False)
    envelope = np.exp(-2.5 * t)  # softer decay
    wave = np.sin(2 * math.pi * freq * t) * envelope
    return wave * 0.4

song = np.array([])

for m in range(measures):

    note = random.choice(melody_scale)
    melody = synth(note_freq[note], note_duration)

    chord = random.choice(left_hand_chords)
    lh = np.zeros(len(melody))
    for c in chord:
        lh += synth(note_freq[c], note_duration) * 0.7

    song = np.concatenate((song, melody + lh))

song = song / np.max(np.abs(song))
song = song.astype(np.float32)

write("output.wav", sample_rate, song)
Audio("output.wav")
