#!/usr/bin/env python3
"""
The code reads a raw PCM file (s16, 16kHz by default) as stream and processes it using the RVC model.
The processed audio is written to a new raw PCM file (s16, 16kHz by default) as stream.

Shell command example:
python realtime_inference.py --input input.pcm --output output.pcm --pth_path model.pth --index_path model.index \
    --pitch 0 --formant 0.0 --f0method fcpe --block_time 0.25 --crossfade_time 0.05 --extra_time 2.5 \
    --I_noise_reduce --O_noise_reduce --use_pv --rms_mix_rate 0.0 --index_rate 0.0 --n_cpu 4 --sr_type sr_model
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio.transforms as tat
import librosa
import re
import shutil
import multiprocessing
from multiprocessing import Queue, cpu_count

# Environment settings
os.environ["OMP_NUM_THREADS"] = "4"
if sys.platform == "darwin":
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from dotenv import load_dotenv
load_dotenv()

# Import custom modules (ensure these are in your PYTHONPATH)
from infer.lib import rtrvc as rvc_for_realtime
from configs.config import Config
from tools.torchgate import TorchGate

# =============================================================================
# Utility: Phase Vocoder (used for SOLA blending if desired)
# =============================================================================
def phase_vocoder(a, b, fade_out, fade_in):
    window = torch.sqrt(fade_out * fade_in)
    fa = torch.fft.rfft(a * window)
    fb = torch.fft.rfft(b * window)
    absab = torch.abs(fa) + torch.abs(fb)
    n = a.shape[0]
    if n % 2 == 0:
        absab[1:-1] *= 2
    else:
        absab[1:] *= 2
    phia = torch.angle(fa)
    phib = torch.angle(fb)
    deltaphase = phib - phia
    deltaphase = deltaphase - 2 * np.pi * torch.floor(deltaphase / (2 * np.pi) + 0.5)
    w = 2 * np.pi * torch.arange(n // 2 + 1, device=a.device) + deltaphase
    t = torch.arange(n, device=a.device).unsqueeze(-1) / n
    result = (
        a * (fade_out ** 2)
        + b * (fade_in ** 2)
        + torch.sum(absab * torch.cos(w * t + phia), -1) * window / n
    )
    return result

# =============================================================================
# Multiprocessing class for f0 extraction using pyworld.harvest
# =============================================================================
class Harvest(multiprocessing.Process):
    def __init__(self, inp_q, opt_q):
        super().__init__()
        self.inp_q = inp_q
        self.opt_q = opt_q

    def run(self):
        import numpy as np
        import pyworld
        while True:
            idx, x, res_f0, n_cpu, ts = self.inp_q.get()
            f0, t = pyworld.harvest(
                x.astype(np.double),
                fs=16000,
                f0_ceil=1100,
                f0_floor=50,
                frame_period=10,
            )
            res_f0[idx] = f0
            if len(res_f0.keys()) >= n_cpu:
                self.opt_q.put(ts)

# =============================================================================
# Main processing: Read from PCM file, process, and write to PCM file.
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="RVC-based PCM converter (raw PCM: s16, 16kHz by default)."
    )
    # File I/O and model parameters
    parser.add_argument("--input", required=True, help="Path to input PCM file (s16, 16kHz by default)")
    parser.add_argument("--output", required=True, help="Path to output PCM file (s16, 16kHz by default)")
    parser.add_argument("--pth_path", required=True, help="Path to the .pth model file")
    parser.add_argument("--index_path", required=True, help="Path to the .index file")
    parser.add_argument("--pitch", type=int, default=0, help="Pitch adjustment (default: 0)")
    parser.add_argument("--formant", type=float, default=0.0, help="Formant shift (default: 0.0)")
    parser.add_argument(
        "--f0method",
        choices=["pm", "harvest", "crepe", "rmvpe", "fcpe"],
        default="fcpe",
        help="f0 extraction method (default: fcpe)",
    )
    # Audio processing parameters
    parser.add_argument("--sr_type", choices=["sr_model", "sr_device"], default="sr_model",
                        help="Sample rate type (default: sr_model)")
    parser.add_argument("--block_time", type=float, default=0.25, help="Block time in seconds (default: 0.25)")
    parser.add_argument("--threhold", type=int, default=-60, help="Response threshold in dB (default: -60)")
    parser.add_argument("--crossfade_time", type=float, default=0.05, help="Crossfade time in seconds (default: 0.05)")
    parser.add_argument("--extra_time", type=float, default=2.5, help="Extra inference time in seconds (default: 2.5)")
    parser.add_argument("--I_noise_reduce", action="store_true", help="Enable input noise reduction")
    parser.add_argument("--O_noise_reduce", action="store_true", help="Enable output noise reduction")
    parser.add_argument("--use_pv", action="store_true", help="Enable phase vocoder for blending")
    parser.add_argument("--rms_mix_rate", type=float, default=0.0, help="RMS mixing rate (default: 0.0)")
    parser.add_argument("--index_rate", type=float, default=0.0, help="Index rate (default: 0.0)")
    parser.add_argument("--n_cpu", type=int, default=min(cpu_count(), 4), help="Number of CPUs for inference (default: min(cpu_count(), 4))")
    # Optional: allow changing the sample rate if needed
    parser.add_argument("--samplerate", type=int, default=16000, help="Input sample rate (default: 16000)")
    args = parser.parse_args()

    # Setup multiprocessing for f0 extraction using the provided CPU count
    inp_q = Queue()
    opt_q = Queue()
    for _ in range(args.n_cpu):
        p = Harvest(inp_q, opt_q)
        p.daemon = True
        p.start()

    # Load configuration and initialize the RVC model
    config = Config()
    rvc = rvc_for_realtime.RVC(
        args.pitch,
        args.formant,
        args.pth_path,
        args.index_path,
        args.index_rate,
        args.n_cpu,
        inp_q,
        opt_q,
        config,
        None,
    )

    # -------------------------------------------------------------------------
    # PCM file stream processing parameters
    # -------------------------------------------------------------------------
    # Use the provided sample rate (default 16000)
    samplerate = args.samplerate
    channels = 1  # assuming mono input
    zc = samplerate // 100  # e.g., 160 for 16000 Hz

    block_time = args.block_time       # seconds per block
    crossfade_time = args.crossfade_time # seconds for SOLA blending
    extra_time = args.extra_time        # extra inference time (seconds)

    block_frame = int(round(block_time * samplerate / zc)) * zc
    block_frame_16k = block_frame  # For 16kHz input, these are the same
    crossfade_frame = int(round(crossfade_time * samplerate / zc)) * zc
    sola_buffer_frame = min(crossfade_frame, 4 * zc)
    sola_search_frame = zc
    extra_frame = int(round(extra_time * samplerate / zc)) * zc
    skip_head = extra_frame // zc
    return_length = (block_frame + sola_buffer_frame + sola_search_frame) // zc

    device = config.device

    # Allocate ring buffers for processing
    input_wav = torch.zeros(
        extra_frame + crossfade_frame + sola_search_frame + block_frame,
        device=device,
        dtype=torch.float32,
    )
    input_wav_res = torch.zeros(
        block_frame,
        device=device,
        dtype=torch.float32,
    )

    # Fade windows for SOLA crossfade
    fade_in_window = (
        torch.sin(
            0.5 * np.pi * torch.linspace(0.0, 1.0, steps=sola_buffer_frame, device=device, dtype=torch.float32)
        ) ** 2
    )
    fade_out_window = 1 - fade_in_window

    # Buffer for SOLA blending and output noise reduction
    sola_buffer = torch.zeros(sola_buffer_frame, device=device, dtype=torch.float32)
    output_buffer = input_wav.clone()

    # If input noise reduction is enabled, prepare additional buffers
    if args.I_noise_reduce:
        input_wav_denoise = input_wav.clone()
        nr_buffer = torch.zeros(sola_buffer_frame, device=device, dtype=torch.float32)

    # TorchGate for noise reduction
    tg = TorchGate(sr=samplerate, n_fft=4 * zc, prop_decrease=0.9).to(device)

    # -------------------------------------------------------------------------
    # Open the input and output PCM files
    # -------------------------------------------------------------------------
    fin = open(args.input, "rb")
    fout = open(args.output, "wb")
    bytes_per_sample = 2  # For s16 PCM (2 bytes per sample)
    block_size_bytes = block_frame * channels * bytes_per_sample

    print("Starting PCM processing...")
    start_time = time.perf_counter()

    while True:
        raw_data = fin.read(block_size_bytes)
        if not raw_data or len(raw_data) < block_size_bytes:
            break  # End-of-file

        # Convert raw int16 PCM data to float32 in range [-1.0, 1.0]
        audio_block = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0

        # ---------------------------------------------------------------------
        # Apply threshold-based noise gating (if enabled)
        # This splits the block into frames of length 'zc' and mutes frames below the threshold.
        # ---------------------------------------------------------------------
        if args.threhold > -60:
            num_frames = len(audio_block) // zc
            for i in range(num_frames):
                start_idx = i * zc
                end_idx = (i + 1) * zc
                frame = audio_block[start_idx:end_idx]
                rms = np.sqrt(np.mean(frame ** 2))
                db = 20 * np.log10(rms) if rms > 0 else -100
                if db < args.threhold:
                    audio_block[start_idx:end_idx] = 0

        # Update the ring buffer with new audio data
        input_wav[:-block_frame] = input_wav[block_frame:].clone()
        input_wav[-block_frame:] = torch.from_numpy(audio_block).to(device)

        # Update input_wav_res with or without noise reduction
        if args.I_noise_reduce:
            input_wav_denoise[:-block_frame] = input_wav_denoise[block_frame:].clone()
            # Select a segment for denoising: last (sola_buffer_frame + block_frame) samples
            segment = input_wav[-(sola_buffer_frame + block_frame):]
            denoised_segment = tg(segment.unsqueeze(0), input_wav.unsqueeze(0)).squeeze(0)
            # Blend the beginning of the segment with previous noise reduction buffer
            denoised_segment[:sola_buffer_frame] *= fade_in_window
            denoised_segment[:sola_buffer_frame] += nr_buffer * fade_out_window
            input_wav_denoise[-block_frame:] = denoised_segment[:block_frame]
            nr_buffer[:] = denoised_segment[block_frame:]
            input_wav_res[:-block_frame_16k] = input_wav_res[block_frame_16k:].clone()
            input_wav_res[-block_frame_16k:] = input_wav_denoise[-block_frame:]
        else:
            input_wav_res[:-block_frame_16k] = input_wav_res[block_frame_16k:].clone()
            input_wav_res[-block_frame_16k:] = input_wav[-block_frame:]

        # ---------------------------------------------------------------------
        # Inference using the RVC model
        # ---------------------------------------------------------------------
        infer_wav = rvc.infer(
            input_wav_res, block_frame_16k, skip_head, return_length, args.f0method
        )
        # If the target sample rate differs, perform resampling (typically not needed for 16kHz)
        if rvc.tgt_sr != samplerate:
            resampler2 = tat.Resample(orig_freq=rvc.tgt_sr, new_freq=samplerate, dtype=torch.float32).to(device)
            infer_wav = resampler2(infer_wav)
        else:
            resampler2 = None

        # Volume envelope mixing if rms_mix_rate < 1
        if args.rms_mix_rate < 1:
            if args.I_noise_reduce:
                base_wav = input_wav_denoise[extra_frame:]
            else:
                base_wav = input_wav[extra_frame:]
            rms1 = librosa.feature.rms(
                y=base_wav[:infer_wav.shape[0]].cpu().numpy(),
                frame_length=4 * zc,
                hop_length=zc
            )
            rms1 = torch.from_numpy(rms1).to(device)
            rms1 = F.interpolate(rms1.unsqueeze(0), size=infer_wav.shape[0] + 1, mode="linear", align_corners=True)[0, 0, :-1]
            rms2 = librosa.feature.rms(
                y=infer_wav.cpu().numpy(),
                frame_length=4 * zc,
                hop_length=zc
            )
            rms2 = torch.from_numpy(rms2).to(device)
            rms2 = F.interpolate(rms2.unsqueeze(0), size=infer_wav.shape[0] + 1, mode="linear", align_corners=True)[0, 0, :-1]
            rms2 = torch.max(rms2, torch.ones_like(rms2) * 1e-3)
            infer_wav *= torch.pow(rms1 / rms2, torch.tensor(1 - args.rms_mix_rate, device=device))

        # Output noise reduction if enabled
        if args.O_noise_reduce:
            output_buffer[:-block_frame] = output_buffer[block_frame:].clone()
            output_buffer[-block_frame:] = infer_wav[-block_frame:]
            infer_wav = tg(infer_wav.unsqueeze(0), output_buffer.unsqueeze(0)).squeeze(0)

        # ---------------------------------------------------------------------
        # SOLA algorithm for smoothing block transitions
        # ---------------------------------------------------------------------
        conv_input = infer_wav[None, None, :sola_buffer_frame + sola_search_frame]
        cor_nom = F.conv1d(conv_input, sola_buffer[None, None, :])
        cor_den = torch.sqrt(
            F.conv1d(
                conv_input ** 2,
                torch.ones(1, 1, sola_buffer_frame, device=device),
            ) + 1e-8
        )
        ratio = cor_nom[0, 0] / cor_den[0, 0]
        sola_offset = torch.argmax(ratio).item()
        print(f"sola_offset = {sola_offset}")

        infer_wav = infer_wav[sola_offset:]
        # Use phase vocoder blending if enabled, otherwise use simple crossfade
        if args.use_pv:
            infer_wav[:sola_buffer_frame] = phase_vocoder(sola_buffer, infer_wav[:sola_buffer_frame], fade_out_window, fade_in_window)
        else:
            infer_wav[:sola_buffer_frame] = infer_wav[:sola_buffer_frame] * fade_in_window + sola_buffer * fade_out_window
        # Update SOLA buffer for next iteration
        sola_buffer = infer_wav[block_frame:block_frame + sola_buffer_frame].clone()

        # The final output block is the first block_frame samples of the processed waveform.
        out_block = infer_wav[:block_frame]

        # Convert the float32 block back to int16 and write to the output file
        out_np = (out_block.cpu().numpy() * 32768.0).clip(-32768, 32767).astype(np.int16)
        fout.write(out_np.tobytes())

    fin.close()
    fout.close()
    elapsed = time.perf_counter() - start_time
    print(f"Processing complete. Elapsed time: {elapsed:.2f} seconds.")

if __name__ == "__main__":
    main()
