"""YourMT3+ runner. Executed by the YourMT3 environment's Python, never imported by Stemforge.

    <yourmt3-python> yourmt3_run.py <code_dir> <audio.wav> <out.json>

``code_dir`` is a checkout of https://huggingface.co/spaces/mimbres/YourMT3 (GPL-3.0 code) with the
checkpoint from https://huggingface.co/mimbres/YourMT3 (repository card: Apache-2.0) under
``amt/logs/2024/...``. Inference follows the authors' ``model_helper.transcribe`` with the
"YPTF.MoE+Multi (noPS)" configuration from their ``app.py``, in float32 for CPU, and writes the raw
note list (seconds, MIDI program, drum flag, velocity) instead of a MIDI file.

The authors' loader uses ``torch.load(weights_only=False)``, i.e. a full pickle. It runs only here,
in the isolated environment, on the authors' published checkpoint.
"""

import json
import os
import sys
from collections import Counter

CHECKPOINT = "mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_b36_nops@last.ckpt"
ARGS = [CHECKPOINT, "-p", "2024", "-tk", "mc13_full_plus_256", "-dec", "multi-t5", "-nl", "26",
        "-enc", "perceiver-tf", "-sqr", "1", "-ff", "moe", "-wf", "4", "-nmoe", "8", "-kmoe", "2",
        "-act", "silu", "-epe", "rope", "-rp", "1", "-ac", "spec", "-hop", "300", "-atc", "1", "-pr", "32"]


def main() -> None:
    code_dir, wav, out = sys.argv[1:4]
    os.chdir(code_dir)
    sys.path.insert(0, os.path.join(code_dir, "amt", "src"))
    sys.path.insert(0, code_dir)
    import torch
    import torchaudio
    from model_helper import load_model_checkpoint
    from utils.audio import slice_padded_array
    from utils.event2note import merge_zipped_note_events_and_ties_to_notes
    from utils.note2event import mix_notes

    torch.set_num_threads(max(1, os.cpu_count() or 1))
    model = load_model_checkpoint(args=ARGS, device="cpu")
    import soundfile as sf

    # soundfile instead of torchaudio.load, which needs TorchCodec in current torchaudio.
    data, sr = sf.read(wav, dtype="float32", always_2d=True)
    audio = torch.from_numpy(data.mean(axis=1)).unsqueeze(0)
    audio = torchaudio.functional.resample(audio, sr, model.audio_cfg["sample_rate"])
    frames = model.audio_cfg["input_frames"]
    segments = slice_padded_array(audio, frames, frames)
    segments = torch.from_numpy(segments.astype("float32")).unsqueeze(1)
    with torch.inference_mode():
        tokens, _ = model.inference_file(bsz=4, audio_segments=segments)
    starts = [frames * i / model.audio_cfg["sample_rate"] for i in range(segments.shape[0])]
    per_channel = []
    errors: Counter = Counter()
    for ch in range(model.task_manager.num_decoding_channels):
        zipped, _, _ = model.task_manager.detokenize_list_batches([a[:, ch, :] for a in tokens], starts,
                                                                    return_events=True)
        notes_ch, err = merge_zipped_note_events_and_ties_to_notes(zipped)
        per_channel.append(notes_ch)
        errors += err
    notes = mix_notes(per_channel)
    json.dump({"notes": [{"pitch": int(n.pitch), "start": float(n.onset), "end": float(n.offset),
                          "program": int(n.program), "is_drum": bool(n.is_drum), "velocity": int(n.velocity)}
                         for n in notes], "decode_errors": dict(errors)}, open(out, "w"))


if __name__ == "__main__":
    main()
