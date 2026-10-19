import numpy as np, wave, sys

SR = 48000
BPM = 150
BEAT = 60 / BPM
DUR = float(sys.argv[2]) if len(sys.argv) > 2 else 14.0
N = int(SR * DUR)
L = np.zeros(N); R = np.zeros(N)
rng = np.random.default_rng(7)

def note(m): return 440 * 2 ** ((m - 69) / 12)

def add(buf, start, sig):
    i = int(start * SR)
    if i >= N: return
    sig = sig[: N - i]
    buf[i:i + len(sig)] += sig

def env(n, a=0.005, d=0.1, s=0.6, r=0.05):
    t = np.arange(n) / SR
    e = np.ones(n) * s
    na, nd, nr = int(a * SR), int(d * SR), int(r * SR)
    e[:na] = np.linspace(0, 1, na)
    e[na:na + nd] = np.linspace(1, s, len(e[na:na + nd]))
    if nr: e[-nr:] *= np.linspace(1, 0, nr)
    return e

def saw(f, n, det=(0,)):
    t = np.arange(n) / SR
    return sum(2 * ((t * f * 2 ** (c / 1200) + rng.random()) % 1) - 1 for c in det) / len(det)

def lowpass(x, a):
    y = np.empty_like(x); acc = 0.0
    for i, v in enumerate(x):
        acc += a * (v - acc); y[i] = acc
    return y

def kick():
    n = int(0.35 * SR); t = np.arange(n) / SR
    f = 50 + 120 * np.exp(-t * 35)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 9) * 0.9

def snare():
    n = int(0.22 * SR); t = np.arange(n) / SR
    return (rng.standard_normal(n) * 0.5 + np.sin(2 * np.pi * 190 * t) * 0.5) * np.exp(-t * 18) * 0.5

def hat(open_=False):
    n = int((0.12 if open_ else 0.04) * SR)
    x = np.diff(rng.standard_normal(n + 1))
    return x * np.exp(-np.arange(n) / SR * (25 if open_ else 90)) * 0.12

# Am - F - C - G, one bar each, looped
prog = [(57, [69, 72, 76]), (53, [65, 69, 72]), (48, [64, 67, 72]), (55, [62, 67, 71])]
bar = 4 * BEAT
nbars = int(np.ceil(DUR / bar))
side = np.ones(N)  # sidechain envelope from kicks

K, S_, H, HO = kick(), snare(), hat(), hat(True)
for b in range(nbars):
    t0 = b * bar
    root, chord = prog[b % 4]
    for q in range(4):
        tk = t0 + q * BEAT
        add(L, tk, K); add(R, tk, K)
        i = int(tk * SR); n = int(0.18 * SR)
        if i < N:
            seg = side[i:i + n]; seg *= np.linspace(0.25, 1, n)[: len(seg)] ** 0.5
        if q in (1, 3):
            add(L, tk, S_ * 0.9); add(R, tk, S_)
        for e in range(2):
            h = HO if e == 1 else H
            add(L, tk + e * BEAT / 2, h * 0.8); add(R, tk + e * BEAT / 2, h)
    # fill on last bar of each 4
    if b % 4 == 3:
        for k in range(4):
            add(L, t0 + 3 * BEAT + k * BEAT / 4, S_ * 0.5 * (k + 1) / 4)
            add(R, t0 + 3 * BEAT + k * BEAT / 4, S_ * 0.5 * (k + 1) / 4)

bass = np.zeros(N); pad = np.zeros(N); lead = np.zeros(N)
for b in range(nbars):
    t0 = b * bar
    root, chord = prog[b % 4]
    for e in range(8):  # offbeat-driven 8th bass
        n = int(BEAT / 2 * SR)
        sig = saw(note(root - 12), n) * env(n, 0.003, 0.08, 0.5, 0.02)
        add(bass, t0 + e * BEAT / 2, sig * 0.35)
    n = int(bar * SR)
    for m in chord:
        add(pad, t0, saw(note(m), n, (-12, 0, 12)) * env(n, 0.02, 0.3, 0.7, 0.05) * 0.09)
    arp = chord + [chord[1] + 12, chord[2] + 12, chord[1] + 12, chord[0] + 12, chord[2]]
    for s in range(16):  # 16th arpeggio
        n = int(BEAT / 4 * SR)
        m = arp[s % len(arp)] + 12
        t = np.arange(n) / SR
        sig = (np.sign(np.sin(2 * np.pi * note(m) * t)) * 0.5 + np.sin(2 * np.pi * note(m) * t) * 0.5)
        add(lead, t0 + s * BEAT / 4, sig * env(n, 0.002, 0.05, 0.3, 0.01) * 0.07)

bass = lowpass(bass, 0.08); pad = lowpass(pad, 0.12); lead = lowpass(lead, 0.35)
bass *= side; pad *= side
L += bass + pad * 1.1 + lead * 0.7
R += bass + pad * 0.9 + lead * 1.0
# simple echo on lead for width
d = int(BEAT * 0.75 * SR)
R[d:] += lead[:-d] * 0.35; L[d:] += lead[:-d] * 0.2

out = np.stack([L, R], 1)
fo = int(1.2 * SR); out[-fo:] *= np.linspace(1, 0, fo)[:, None]
out = np.tanh(out * 1.3) / np.tanh(1.3)
out /= np.abs(out).max() / 0.89
with wave.open(sys.argv[1], "wb") as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
    w.writeframes((out * 32767).astype("<i2").tobytes())
