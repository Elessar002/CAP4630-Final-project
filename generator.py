

import os
import math
import random

import numpy as np
from scipy.io.wavfile import write

# ====== Global audio settings ======

SAMPLE_RATE = 44100          # CD quality
NOTE_DURATION = 0.75         # seconds per note
NUM_NOTES = 40               # 40 * 0.75s = 30 seconds

# ====== Frequency map for notes (equal temperament) ======

NOTE_FREQ = {
    "C3": 130.81, "D3": 146.83, "E3": 164.81, "F3": 174.61, "G3": 196.00,
    "A3": 220.00, "B3": 246.94,

    "C4": 261.63, "D4": 293.66, "E4": 329.63, "F4": 349.23, "G4": 392.00,
    "A4": 440.00, "B4": 493.88,

    "C5": 523.25, "D5": 587.33, "E5": 659.25
}

# A peaceful C-major pentatonic-ish scale (no harsh dissonances)
MELODY_SCALE = ["C4", "D4", "E4", "G4", "A4", "C5", "D5"]


# ====== Utility functions ======

def envelope(num_samples: int) -> np.ndarray:
    
    t = np.linspace(0, 1, num_samples, endpoint=False)
    # Attack: first 10%, Release: last 20%
    attack = 0.10
    release = 0.20

    env = np.ones_like(t)

    # Attack segment
    attack_samples = int(num_samples * attack)
    if attack_samples > 0:
        env[:attack_samples] = np.linspace(0.0, 1.0, attack_samples)

    # Release segment
    release_samples = int(num_samples * release)
    if release_samples > 0:
        env[-release_samples:] = np.linspace(1.0, 0.0, release_samples)

    return env


def synth_note(frequency: float, duration: float, sample_rate: int) -> np.ndarray:
    
    num_samples = int(duration * sample_rate)
    t = np.linspace(0, duration, num_samples, endpoint=False)

    # Fundamental plus a couple of harmonics with decaying amplitudes
    fundamental = np.sin(2 * math.pi * frequency * t)
    harmonic2 = 0.5 * np.sin(2 * math.pi * 2 * frequency * t)
    harmonic3 = 0.25 * np.sin(2 * math.pi * 3 * frequency * t)

    raw_note = fundamental + harmonic2 + harmonic3

    # Apply envelope
    env = envelope(num_samples)
    note = raw_note * env

    return note


def choose_next_note(prev_note: str | None) -> str:
    
    if prev_note is None:
        return random.choice(MELODY_SCALE)

    # Current index in scale
    idx = MELODY_SCALE.index(prev_note)

    # Candidate moves: stay, +/-1 step, maybe +/-2 with smaller probability
    candidates = []
    weights = []

    for offset, base_weight in [(-2, 0.5), (-1, 1.5), (0, 1.0), (1, 1.5), (2, 0.5)]:
        new_idx = idx + offset
        if 0 <= new_idx < len(MELODY_SCALE):
            candidates.append(MELODY_SCALE[new_idx])
            weights.append(base_weight)

    # Normalize weights and pick
    total = sum(weights)
    probs = [w / total for w in weights]
    choice = random.choices(candidates, probs, k=1)[0]
    return choice


def generate_peaceful_melody(
    sample_rate: int = SAMPLE_RATE,
    note_duration: float = NOTE_DURATION,
    num_notes: int = NUM_NOTES,
) -> np.ndarray:
    
    random.seed(42)  # make it reproducible

    melody_audio = []
    prev_note = None

    for i in range(num_notes):
        # Pick next melodic note
        note_name = choose_next_note(prev_note)
        prev_note = note_name

        freq = NOTE_FREQ[note_name]
        note_wave = synth_note(freq, note_duration, sample_rate)

        # Optionally add a soft low "drone" or root note for warmth
        # e.g., every 2 notes, add a low C3 under the melody
        if i % 2 == 0:
            low_freq = NOTE_FREQ["C3"]
            low_note = synth_note(low_freq, note_duration, sample_rate)
            note_wave = note_wave + 0.4 * low_note

        melody_audio.append(note_wave)

    # Concatenate all notes
    full_audio = np.concatenate(melody_audio)

    # Normalize to avoid clipping
    max_val = np.max(np.abs(full_audio))
    if max_val > 0:
        full_audio = full_audio / max_val * 0.9  # leave a bit of headroom

    return full_audio.astype(np.float32)


def save_to_wav(waveform: np.ndarray, sample_rate: int, path: str) -> None:
    """
    Save the waveform (float32 in [-1, 1]) to a 16-bit PCM WAV file.
    """
    # Convert to int16
    waveform_int16 = np.int16(waveform * 32767)
    write(path, sample_rate, waveform_int16)


def main():
    # Make sure output directory exists
    os.makedirs("output", exist_ok=True)
    output_path = os.path.join("output", "peaceful_piano_30s.wav")

    print("Generating peaceful piano audio...")
    audio = generate_peaceful_melody()
    print(f"Saving to {output_path} ...")
    save_to_wav(audio, SAMPLE_RATE, output_path)
    print("Done! 🎹")


if __name__ == "__main__":
    main()

