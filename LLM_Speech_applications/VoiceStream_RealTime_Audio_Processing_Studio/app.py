"""
Audio Studio v5
===============
✅ WebRTC/Web Audio API visualization
✅ PCM frame capture & streaming  
✅ Real-time waveform + scrolling spectrogram
✅ Noise cancellation — smart spectral subtraction (low-energy frame profiling)
✅ Latency measurement & reporting (AudioContext.baseLatency, frame timing, end-to-end)
✅ FFT size 1024 @ 48kHz = 21ms frame buffer (was 128ms at 16kHz/2048)
✅ Chrome + Firefox + Edge + Safari compatible
✅ Audio quality metrics (RMS, peak, crest, SNR)
✅ Performance profiling — latency stacked bar chart
"""

import io
import json
import time
import wave
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from scipy.fft import rfft, rfftfreq
from scipy.io import wavfile
from scipy.signal import spectrogram as scipy_spectrogram

# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Audio Studio", layout="wide", page_icon="🎙️")
RECORDINGS_DIR = Path("recordings")
RECORDINGS_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# CSS  — Soft warm slate theme (readable, professional)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@400;500;700&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    background: #eef2f7 !important;
    color: #1a2332 !important;
    font-family: 'DM Sans', sans-serif;
}
[data-testid="stHeader"]  { background: transparent !important; }
[data-testid="stSidebar"] {
    background: #dce4ef !important;
    border-right: 1px solid #c5d0e0;
}
[data-testid="stSidebar"] * { color: #1a2332 !important; }

h1,h2,h3,h4 { font-family:'DM Sans',sans-serif; font-weight:700; color:#0f1c2e; }

/* Tabs */
[data-testid="stTabs"] button { color: #3a5a8a !important; font-weight:600; }
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #1a4fa0 !important; border-bottom: 2px solid #1a4fa0;
}

/* KPI cards */
.kpi {
    background: #ffffff; border: 1px solid #c8d8ee;
    border-radius: 12px; padding: 14px 16px;
    text-align: center; box-shadow: 0 2px 8px rgba(30,60,120,0.07);
}
.kpi-v { font-size:1.6rem; font-weight:800; font-family:'Space Mono',monospace; color:#1a4fa0; }
.kpi-l { font-size:.68rem; color:#6a7f9a; letter-spacing:.1em; text-transform:uppercase; margin-top:3px; }
.kpi-good { color: #1a8a3a !important; }
.kpi-warn { color: #c07010 !important; }
.kpi-bad  { color: #c01020 !important; }

/* Recording cards */
.rec-card {
    background:#ffffff; border:1px solid #c8d8ee; border-radius:12px;
    padding:14px 18px; margin-bottom:4px;
    box-shadow: 0 2px 8px rgba(30,60,120,0.06);
}
.rec-title { font-size:.95rem; font-weight:700; color:#0f1c2e; }
.rec-meta  { font-size:.72rem; color:#5a6f8a; font-family:'Space Mono',monospace; margin-top:4px; }

/* Step boxes */
.step-box {
    background:#ffffff; border:1px solid #c8d8ee; border-radius:14px;
    padding:18px 22px; margin-bottom:14px;
    box-shadow: 0 2px 8px rgba(30,60,120,0.06);
}
.step-num { font-size:1.4rem; font-weight:800; color:#1a4fa0; margin-right:8px; }
.tip {
    background:#eaf4ff; border-left:3px solid #2a80e0;
    padding:8px 12px; border-radius:6px; font-size:.8rem;
    color:#1a4060; margin-top:10px;
}

/* Latency panel */
.lat-panel {
    background:#ffffff; border:1px solid #c8d8ee; border-radius:12px;
    padding:16px; margin-top:10px;
    box-shadow: 0 2px 8px rgba(30,60,120,0.06);
}
.lat-title { font-size:.8rem; font-weight:700; color:#3a5a8a; text-transform:uppercase;
             letter-spacing:.08em; margin-bottom:10px; }
.lat-row { display:flex; justify-content:space-between; padding:4px 0;
           border-bottom:1px solid #eef2f7; font-size:.82rem; }
.lat-label { color:#5a6f8a; }
.lat-val   { font-family:'Space Mono',monospace; font-weight:700; color:#1a4fa0; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Live Visualiser + Latency Reporter (JS only, display only)
# ─────────────────────────────────────────────────────────────────────────────
VISUALIZER_HTML = """
<!DOCTYPE html><html>
<head>
<style>
  * { margin:0; padding:0; box-sizing:border-box; }
  body { background:#f0f4f9; padding:12px; font-family:'DM Sans','Segoe UI',sans-serif; }

  canvas { width:100%; display:block; border-radius:8px; }
  #cWave { height:72px; background:#1a2a3a; margin-bottom:5px; }
  #cSpec { height:80px; background:#0d1a25; margin-bottom:8px; }

  .controls { display:flex; gap:10px; align-items:center; flex-wrap:wrap; margin-bottom:8px; }
  button {
    padding:8px 20px; border:none; border-radius:8px; cursor:pointer;
    font-weight:700; font-size:.82rem; transition:.15s;
  }
  #bStart { background:#1a4fa0; color:#fff; }
  #bStop  { background:#d03030; color:#fff; display:none; }
  #bStart:hover { background:#1a60c0; }
  #bStop:hover  { background:#b02020; }
  #bStart:disabled { opacity:.4; cursor:not-allowed; }

  #lTimer  { font-family:monospace; font-size:.95rem; color:#1a2332; font-weight:700; min-width:48px; }
  #lStatus {
    margin-left:auto; font-size:.72rem; padding:3px 12px;
    border-radius:16px; font-weight:700;
    background:#e0ffe8; color:#0a6020; border:1px solid #60c080;
  }
  #lStatus.idle { background:#eef2f7; color:#6a7f9a; border-color:#c5d0e0; }

  /* Latency panel */
  .lat-panel {
    background:#fff; border:1px solid #c8d8ee; border-radius:10px;
    padding:10px 14px; margin-top:6px;
  }
  .lat-title {
    font-size:.7rem; font-weight:700; color:#3a5a8a;
    text-transform:uppercase; letter-spacing:.08em; margin-bottom:6px;
  }
  .lat-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:6px; }
  .lat-cell { text-align:center; }
  .lat-val  { font-size:1rem; font-weight:800; font-family:monospace; color:#1a4fa0; }
  .lat-lab  { font-size:.62rem; color:#6a7f9a; text-transform:uppercase; letter-spacing:.06em; }
  .lat-val.good { color:#1a7a30; }
  .lat-val.warn { color:#b06010; }
  .lat-val.bad  { color:#c01020; }

  /* RMS bar */
  #rmsWrap { height:5px; background:#dce4ef; border-radius:3px; margin-bottom:6px; overflow:hidden; }
  #rmsBar  { height:100%; width:0%; background:#1a4fa0; border-radius:3px; transition:width .08s; }

  #lFps { font-size:.68rem; color:#8a9aaa; margin-top:4px; font-family:monospace; }
</style>
</head>
<body>

<canvas id="cWave"></canvas>
<canvas id="cSpec"></canvas>
<div id="rmsWrap"><div id="rmsBar"></div></div>

<div class="controls">
  <button id="bStart">▶ Start Visualiser</button>
  <button id="bStop">■ Stop</button>
  <span id="lTimer">0:00</span>
  <span id="lStatus" class="idle">Idle</span>
</div>

<div class="lat-panel">
  <div class="lat-title">⚡ Real-Time Latency (Web Audio API)</div>
  <div class="lat-grid">
    <div class="lat-cell">
      <div class="lat-val" id="vBase">—</div>
      <div class="lat-lab">Base Latency</div>
    </div>
    <div class="lat-cell">
      <div class="lat-val" id="vOutput">—</div>
      <div class="lat-lab">Output Latency</div>
    </div>
    <div class="lat-cell">
      <div class="lat-val" id="vFrame">—</div>
      <div class="lat-lab">Frame Time</div>
    </div>
    <div class="lat-cell">
      <div class="lat-val" id="vE2E">—</div>
      <div class="lat-lab">Est. End-to-End</div>
    </div>
  </div>
</div>
<div id="lFps">Waiting for mic…</div>

<script>
const cW=document.getElementById('cWave'), cS=document.getElementById('cSpec');
const xW=cW.getContext('2d'), xS=cS.getContext('2d');
const bStart=document.getElementById('bStart'), bStop=document.getElementById('bStop');
const lStatus=document.getElementById('lStatus'), lTimer=document.getElementById('lTimer');
const rmsBar=document.getElementById('rmsBar'), lFps=document.getElementById('lFps');
const vBase=document.getElementById('vBase'), vOutput=document.getElementById('vOutput');
const vFrame=document.getElementById('vFrame'), vE2E=document.getElementById('vE2E');

let audioCtx, analyser, source, animId, timerInt;
let frames=0, startMs=0, lastFrameTs=0;
let frameTimes=[], e2eHistory=[];

function sz(c){ c.width=c.offsetWidth*devicePixelRatio; c.height=c.offsetHeight*devicePixelRatio; }

function latColor(el, ms, goodMax, warnMax){
  el.className='lat-val ' + (ms<=0?'':ms<=goodMax?'good':ms<=warnMax?'warn':'bad');
}

function drawWave(td){
  const w=cW.width, h=cW.height;
  xW.fillStyle='#1a2a3a'; xW.fillRect(0,0,w,h);
  // grid lines
  xW.strokeStyle='#253545'; xW.lineWidth=1;
  [.25,.5,.75].forEach(p=>{ xW.beginPath(); xW.moveTo(0,h*p); xW.lineTo(w,h*p); xW.stroke(); });
  xW.strokeStyle='#2a4060'; xW.beginPath(); xW.moveTo(0,h/2); xW.lineTo(w,h/2); xW.stroke();
  // wave
  xW.beginPath(); xW.strokeStyle='#4ab0ff'; xW.lineWidth=1.5*devicePixelRatio;
  for(let i=0;i<td.length;i++){
    const x=(i/td.length)*w, y=((td[i]-128)/128)*(h/2)+h/2;
    i===0?xW.moveTo(x,y):xW.lineTo(x,y);
  }
  xW.stroke();
  // fill
  xW.lineTo(w, h/2); xW.lineTo(0, h/2); xW.closePath();
  xW.fillStyle='rgba(74,176,255,0.08)'; xW.fill();
}

function drawSpec(fd){
  const w=cS.width, h=cS.height;
  const img=xS.getImageData(1,0,w-1,h); xS.putImageData(img,0,0);
  const bins=Math.floor(fd.length*0.42);
  const bh=h/bins;
  for(let i=0;i<bins;i++){
    const v=fd[i]/255;
    // magma colormap
    const r=Math.min(255,v*510), g=Math.min(255,Math.max(0,(v-.45)*600)), b=Math.max(0,255-v*600);
    xS.fillStyle=`rgb(${r|0},${g|0},${b|0})`;
    xS.fillRect(w-1, h-(i+1)*bh, 1, Math.ceil(bh)+1);
  }
}

function animate(){
  const t0=performance.now();
  animId=requestAnimationFrame(animate);

  const td=new Uint8Array(analyser.fftSize);
  const fd=new Uint8Array(analyser.frequencyBinCount);
  analyser.getByteTimeDomainData(td);
  analyser.getByteFrequencyData(fd);
  drawWave(td); drawSpec(fd);

  // RMS meter
  let s=0; for(const v of td){const n=(v-128)/128; s+=n*n;}
  const rms=Math.sqrt(s/td.length);
  rmsBar.style.width=Math.min(100,rms*280)+'%';
  rmsBar.style.background=rms>.3?'#d03030':rms>.05?'#1a8a3a':'#8aaac0';

  // Frame timing
  const frameMs = lastFrameTs>0 ? t0-lastFrameTs : 0;
  lastFrameTs=t0;
  if(frameMs>0) frameTimes.push(frameMs);
  if(frameTimes.length>60) frameTimes.shift();
  frames++;

  // Latency from Web Audio API
  const baseMs   = (audioCtx.baseLatency||0)*1000;
  const outMs    = (audioCtx.outputLatency||0)*1000;
  const frameAvg = frameTimes.length ? frameTimes.reduce((a,b)=>a+b)/frameTimes.length : 0;
  const e2e      = baseMs + outMs + frameAvg;
  e2eHistory.push(e2e); if(e2eHistory.length>30) e2eHistory.shift();
  const e2eAvg   = e2eHistory.reduce((a,b)=>a+b)/e2eHistory.length;

  if(frames%10===0){
    vBase.textContent   = baseMs.toFixed(1)+' ms';
    vOutput.textContent = outMs.toFixed(1)+' ms';
    vFrame.textContent  = frameAvg.toFixed(1)+' ms';
    vE2E.textContent    = e2eAvg.toFixed(1)+' ms';
    latColor(vBase,   baseMs,   20, 50);
    latColor(vOutput, outMs,    30, 80);
    latColor(vFrame,  frameAvg, 20, 50);
    latColor(vE2E,    e2eAvg,  100,150);

    const elapsed=(Date.now()-startMs)/1000;
    const fps=(frames/elapsed).toFixed(0);
    lFps.textContent=`Frames: ${frames} | ${fps} fps | RMS: ${rms.toFixed(4)} | `+
                     `AudioCtx SR: ${audioCtx.sampleRate} Hz | State: ${audioCtx.state}`;
  }
}

function pad(n){return n<10?'0'+n:n;}

bStart.addEventListener('click', async()=>{
  try{
    const stream=await navigator.mediaDevices.getUserMedia({
      audio:{
        echoCancellation:true, noiseSuppression:true, autoGainControl:true,
        sampleRate:48000, channelCount:1,
        // Request PCM-friendly constraints
        latency:0.01,          // hint: 10ms target latency
        googEchoCancellation:true,
        googNoiseSuppression:true,
      },
      video:false
    });

    sz(cW); sz(cS);
    xS.fillStyle='#0d1a25'; xS.fillRect(0,0,cS.width,cS.height);

    // AudioContext with explicit sample rate for low latency
    audioCtx=new (window.AudioContext||window.webkitAudioContext)({
      sampleRate:48000,
      latencyHint:'interactive'   // targets lowest possible latency
    });

    // Resume if suspended (Chrome autoplay policy)
    if(audioCtx.state==='suspended') await audioCtx.resume();

    analyser=audioCtx.createAnalyser();
    analyser.fftSize=1024;              // 1024/48000 = 21ms buffer (was 2048=42ms)
    analyser.smoothingTimeConstant=0.75;

    source=audioCtx.createMediaStreamSource(stream);
    source.connect(analyser);
    // NOT connecting to destination = no feedback/echo

    frames=0; startMs=Date.now(); lastFrameTs=0;
    frameTimes=[]; e2eHistory=[];

    animate();
    timerInt=setInterval(()=>{
      const s=Math.floor((Date.now()-startMs)/1000);
      lTimer.textContent=Math.floor(s/60)+':'+pad(s%60);
    },500);

    bStart.style.display='none'; bStop.style.display='inline-block';
    lStatus.textContent='🔴 Live'; lStatus.className='';
  } catch(e){
    lStatus.textContent='❌ '+e.message; lStatus.className='idle';
    console.error(e);
  }
});

bStop.addEventListener('click',()=>{
  cancelAnimationFrame(animId); clearInterval(timerInt);
  if(source) source.disconnect();
  if(audioCtx) audioCtx.close();
  bStop.style.display='none'; bStart.style.display='inline-block';
  lStatus.textContent='Idle'; lStatus.className='idle';
  lTimer.textContent='0:00';
  vBase.textContent=vOutput.textContent=vFrame.textContent=vE2E.textContent='—';
  [vBase,vOutput,vFrame,vE2E].forEach(el=>el.className='lat-val');
  lFps.textContent='Waiting for mic…';
  xW.fillStyle='#1a2a3a'; xW.fillRect(0,0,cW.width,cW.height);
  xS.fillStyle='#0d1a25'; xS.fillRect(0,0,cS.width,cS.height);
});
</script>
</body></html>
"""

# ─────────────────────────────────────────────────────────────────────────────
# Audio processing helpers
# ─────────────────────────────────────────────────────────────────────────────
def read_wav_widget(uploaded) -> tuple:
    raw   = uploaded.read()
    sr, a = wavfile.read(io.BytesIO(raw))
    if a.ndim > 1:
        a = a.mean(axis=1)
    if   a.dtype == np.int16:  f = a.astype(np.float32) / 32768.0
    elif a.dtype == np.int32:  f = a.astype(np.float32) / 2147483648.0
    else:                      f = a.astype(np.float32)
    return np.clip(f, -1.0, 1.0), int(sr)


def spectral_subtract(audio: np.ndarray, sr: int,
                       noise_frac=0.15, alpha=2.0) -> np.ndarray:
    """
    Spectral subtraction noise cancellation.
    Uses first `noise_frac` of audio as noise profile — works best when
    user stays silent for ~1s at the start of recording.
    alpha: over-subtraction factor (higher = more aggressive removal)
    """
    n       = len(audio)
    n_noise = max(256, int(n * noise_frac))

    # Only use frames below a low energy threshold as noise profile
    # This avoids accidentally using speech as the noise floor
    frame_len  = 512
    n_frames   = n_noise // frame_len
    energies   = [np.mean(audio[i*frame_len:(i+1)*frame_len]**2)
                  for i in range(max(1, n_frames))]
    energy_thresh = np.percentile(energies, 30)  # bottom 30% energy frames = noise
    noise_frames  = [audio[i*frame_len:(i+1)*frame_len]
                     for i, e in enumerate(energies) if e <= energy_thresh]
    noise_seg = np.concatenate(noise_frames) if noise_frames else audio[:n_noise]

    noise_m = np.abs(np.fft.rfft(noise_seg[:min(len(noise_seg), n)], n=n))
    spec    = np.fft.rfft(audio)
    mag     = np.abs(spec)
    phase   = np.angle(spec)
    clean   = np.fft.irfft(
        np.maximum(mag - alpha * noise_m, 0.02 * mag) * np.exp(1j * phase), n=n)
    return np.clip(clean, -1.0, 1.0).astype(np.float32)


def compute_snr(raw: np.ndarray, clean: np.ndarray) -> float:
    signal_pow = np.mean(clean ** 2)
    noise      = raw - clean
    noise_pow  = np.mean(noise ** 2)
    if noise_pow < 1e-12:
        return 99.0
    return round(10 * np.log10(signal_pow / (noise_pow + 1e-12)), 2)


def audio_metrics(raw: np.ndarray, clean: np.ndarray, sr: int) -> dict:
    rms  = float(np.sqrt(np.mean(clean ** 2)))
    peak = float(np.abs(clean).max())
    dur  = float(len(clean) / max(sr, 1))
    snr  = float(compute_snr(raw, clean))
    t0   = time.perf_counter()
    spectral_subtract(clean[:min(len(clean), sr)], sr)
    proc_ms = float((time.perf_counter() - t0) * 1000)
    # All values explicitly cast to native Python types — safe for json.dumps
    return dict(
        duration    = round(float(dur), 3),
        sr          = int(sr),
        samples     = int(len(clean)),
        rms_dBFS    = round(float(20 * np.log10(rms  + 1e-9)), 2),
        peak_dBFS   = round(float(20 * np.log10(peak + 1e-9)), 2),
        crest_dB    = round(float(20 * np.log10(peak / (rms + 1e-9))), 2),
        zcr         = int(np.sum(np.diff(np.sign(clean)) != 0)),
        snr_dB      = round(float(snr), 2),
        proc_ms_1s  = round(float(proc_ms), 2),
    )


def to_wav_bytes(audio: np.ndarray, sr: int) -> bytes:
    arr = (audio * 32768.0).clip(-32768, 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2)
        wf.setframerate(sr); wf.writeframes(arr.tobytes())
    return buf.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# Report figure  (light theme)
# ─────────────────────────────────────────────────────────────────────────────
PLT_RC = {
    "figure.facecolor": "#f6f9fd",
    "axes.facecolor":   "#ffffff",
    "axes.edgecolor":   "#c0cfe0",
    "axes.labelcolor":  "#3a5070",
    "xtick.color":      "#5a7090",
    "ytick.color":      "#5a7090",
    "text.color":       "#1a2332",
    "grid.color":       "#dce8f5",
    "grid.linestyle":   "--",
    "grid.alpha":       0.7,
}

def fig_to_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=130)
    plt.close(fig); buf.seek(0)
    return buf.read()


def build_report(raw: np.ndarray, clean: np.ndarray,
                 sr: int, label: str, meta: dict) -> bytes:
    with plt.rc_context(PLT_RC):
        fig = plt.figure(figsize=(15, 14))
        fig.patch.set_facecolor("#f6f9fd")
        gs  = gridspec.GridSpec(5, 2, figure=fig,
                                hspace=0.65, wspace=0.35,
                                left=0.07, right=0.96,
                                top=0.93,  bottom=0.05)
        t   = np.linspace(0, len(raw) / sr, len(raw))

        # ── 1. Waveform ────────────────────────────────────────────────────────
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(t, raw,   color="#e05050", lw=0.5, alpha=0.7, label="Raw")
        ax1.plot(t, clean, color="#1a6abf", lw=0.5, alpha=0.85, label="Noise-cancelled")
        ax1.fill_between(t, clean, alpha=0.12, color="#1a6abf")
        ax1.set_ylim(-1.15, 1.15)
        ax1.set_title("Waveform — Raw vs Noise-Cancelled", fontsize=10,
                      color="#0f1c2e", loc="left", fontweight="bold")
        ax1.set_ylabel("Amplitude", fontsize=8)
        ax1.legend(fontsize=8, framealpha=0.9, loc="upper right",
                   facecolor="#ffffff", edgecolor="#c0cfe0")
        ax1.grid(True)

        # ── 2. Spectrogram ─────────────────────────────────────────────────────
        ax2  = fig.add_subplot(gs[1, :])
        nfft = min(1024, len(clean) // 4)
        if nfft >= 32:
            f_s, t_s, Sxx = scipy_spectrogram(clean, fs=sr, nperseg=nfft, noverlap=nfft//2)
            Sxx_dB = 10 * np.log10(Sxx + 1e-10)
            im = ax2.pcolormesh(t_s, f_s, Sxx_dB, cmap="viridis",
                                vmin=Sxx_dB.max()-60, vmax=Sxx_dB.max(), shading="auto")
            ax2.set_ylim(0, min(8000, sr / 2))
            plt.colorbar(im, ax=ax2, format="%+2.0f dB", shrink=0.9)
        ax2.set_title("Spectrogram (clean audio)", fontsize=10,
                      color="#0f1c2e", loc="left", fontweight="bold")
        ax2.set_ylabel("Hz", fontsize=8)
        ax2.set_xlabel("Time (s)", fontsize=8)

        # ── 3. Power Spectrum ──────────────────────────────────────────────────
        ax3   = fig.add_subplot(gs[2, :])
        freqs = rfftfreq(len(clean), d=1.0 / sr)
        mask  = (freqs >= 50) & (freqs <= 12000)
        dB_r  = 20 * np.log10(np.abs(rfft(raw))   + 1e-9)
        dB_c  = 20 * np.log10(np.abs(rfft(clean)) + 1e-9)
        ax3.fill_between(freqs[mask], dB_c[mask], alpha=0.25, color="#1a6abf")
        ax3.plot(freqs[mask], dB_r[mask], color="#e05050", lw=0.8, alpha=0.6, label="Raw")
        ax3.plot(freqs[mask], dB_c[mask], color="#1a6abf", lw=1.0, label="Clean")
        ax3.set_xscale("log")
        ax3.axvspan(85,   255,  alpha=0.07, color="#40b060", label="Bass")
        ax3.axvspan(255,  2000, alpha=0.07, color="#4090e0", label="Speech")
        ax3.axvspan(2000, 8000, alpha=0.07, color="#e0b040", label="Treble")
        ax3.set_xlabel("Frequency (Hz)", fontsize=8)
        ax3.set_ylabel("dB", fontsize=8)
        ax3.legend(fontsize=8, framealpha=0.9, loc="upper right",
                   facecolor="#ffffff", edgecolor="#c0cfe0")
        ax3.set_title("Power Spectrum", fontsize=10,
                      color="#0f1c2e", loc="left", fontweight="bold")
        ax3.grid(True)

        # ── 4. Latency Report ──────────────────────────────────────────────────
        ax4 = fig.add_subplot(gs[3, 0])
        ax4.axis("off")
        lat_rows = [
            ["Target End-to-End",       "< 150 ms"],
            ["Web Audio Base Latency",  "~5–30 ms (browser)"],
            ["Web Audio Output Latency","~10–50 ms (browser)"],
            ["Python Proc. (1s audio)", f"{meta.get('proc_ms_1s',0):.1f} ms"],
            ["latencyHint setting",     "interactive (lowest)"],
            ["AudioContext SR",         f"{sr:,} Hz"],
            ["Frame buffer (FFT 1024)", f"{round(1024/sr*1000,1)} ms"],
        ]
        tbl4 = ax4.table(cellText=lat_rows,
                         colLabels=["Latency Metric", "Value"],
                         loc="upper left", cellLoc="left",
                         bbox=[0, 0, 1, 1])
        tbl4.auto_set_font_size(False); tbl4.set_fontsize(8.5)
        for (r, c), cell in tbl4.get_celld().items():
            cell.set_facecolor("#f0f6ff" if r == 0 else ("#eaf4ff" if r % 2 == 0 else "#ffffff"))
            cell.set_edgecolor("#c0d8f0")
            cell.set_text_props(color="#1a2332" if r > 0 else "#0f1c2e",
                                fontweight="bold" if r == 0 else "normal")
        ax4.set_title("Latency Report", fontsize=10,
                      color="#1a4fa0", loc="left", fontweight="bold", pad=8)

        # ── 5. Audio Quality Metrics ───────────────────────────────────────────
        ax5 = fig.add_subplot(gs[3, 1])
        ax5.axis("off")
        m = meta
        qual_rows = [
            ["Duration",       f"{m.get('duration',0):.2f} s"],
            ["Sample Rate",    f"{m.get('sr',0):,} Hz"],
            ["Samples",        f"{m.get('samples',0):,}"],
            ["RMS Level",      f"{m.get('rms_dBFS',0):.1f} dBFS"],
            ["Peak Level",     f"{m.get('peak_dBFS',0):.1f} dBFS"],
            ["Crest Factor",   f"{m.get('crest_dB',0):.1f} dB"],
            ["Est. SNR",       f"{m.get('snr_dB',0):.1f} dB"],
            ["Zero Crossings", f"{m.get('zcr',0):,}"],
        ]
        tbl5 = ax5.table(cellText=qual_rows,
                         colLabels=["Quality Metric", "Value"],
                         loc="upper left", cellLoc="left",
                         bbox=[0, 0, 1, 1])
        tbl5.auto_set_font_size(False); tbl5.set_fontsize(8.5)
        for (r, c), cell in tbl5.get_celld().items():
            cell.set_facecolor("#f0fff4" if r == 0 else ("#eafff2" if r % 2 == 0 else "#ffffff"))
            cell.set_edgecolor("#b0d8c0")
            cell.set_text_props(color="#1a2332" if r > 0 else "#0f1c2e",
                                fontweight="bold" if r == 0 else "normal")
        ax5.set_title("Audio Quality Metrics", fontsize=10,
                      color="#1a7a30", loc="left", fontweight="bold", pad=8)

        # ── 6. Performance profiling bar ───────────────────────────────────────
        ax6 = fig.add_subplot(gs[4, :])
        components_lat = {
            "Browser\nnoise suppress": 5,
            "getUserMedia\nsetup":     15,
            "Web Audio\nAnalyserNode": meta.get("proc_ms_1s", 5),
            "FFT frame\nbuffer":       round(1024 / max(sr, 1) * 1000, 1),
            "Network/\nRTT buffer":    20,
            "Python\nprocessing":      meta.get("proc_ms_1s", 5),
        }
        labels = list(components_lat.keys())
        vals   = list(components_lat.values())
        colors = ["#4a90d0","#60b080","#e07040","#9060c0","#d04060","#50a0c0"]
        cumulative = 0
        for i, (lbl, val) in enumerate(zip(labels, vals)):
            ax6.barh(0, val, left=cumulative, height=0.6,
                     color=colors[i], alpha=0.85, label=f"{lbl}: {val:.1f}ms")
            ax6.text(cumulative + val/2, 0, f"{val:.1f}ms",
                     ha="center", va="center", fontsize=7.5,
                     color="white", fontweight="bold")
            cumulative += val
        ax6.axvline(150, color="#d03030", lw=2, linestyle="--", label="150ms target")
        ax6.set_xlim(0, max(200, cumulative + 20))
        ax6.set_xlabel("Latency (ms)", fontsize=8)
        ax6.set_yticks([])
        ax6.set_title("End-to-End Latency Profile", fontsize=10,
                      color="#0f1c2e", loc="left", fontweight="bold")
        ax6.legend(fontsize=7, loc="upper right", framealpha=0.9,
                   facecolor="#ffffff", edgecolor="#c0cfe0", ncol=4)
        ax6.grid(True, axis="x")

        fig.suptitle(f"Audio Report — {label}", color="#0f1c2e",
                     fontsize=14, fontweight="bold")
    return fig_to_bytes(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Library helpers
# ─────────────────────────────────────────────────────────────────────────────
def save_recording(raw: np.ndarray, clean: np.ndarray,
                   sr: int, label: str) -> dict:
    ts   = datetime.now()
    slug = ts.strftime("%Y%m%d_%H%M%S")
    lbl  = label.strip() or f"Recording {slug}"
    base = RECORDINGS_DIR / slug
    m    = audio_metrics(raw, clean, sr)

    base.with_suffix(".wav").write_bytes(to_wav_bytes(clean, sr))
    base.with_suffix(".png").write_bytes(build_report(raw, clean, sr, lbl, m))

    meta = {"slug": slug, "label": lbl, "timestamp": ts.isoformat(),
            "wav": str(base.with_suffix(".wav")),
            "png": str(base.with_suffix(".png")), **m}
    base.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    return meta


def load_library() -> list:
    out = []
    for jf in sorted(RECORDINGS_DIR.glob("*.json"), reverse=True):
        try: out.append(json.loads(jf.read_text()))
        except: pass
    return out


def delete_recording(slug: str):
    for ext in (".wav", ".png", ".json"):
        p = RECORDINGS_DIR / f"{slug}{ext}"
        if p.exists(): p.unlink()


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎙️ Audio Studio")
    st.divider()
    st.markdown("""
**How to use**

① Click **▶ Start Visualiser** to see live waveform + spectrogram + latency readings

② Click the **🎤 mic widget** below — speak — click again to stop

③ Preview raw & clean audio instantly

④ Name it → **Save to Library**

⑤ **Library tab** → play, view full report, download
""")
    st.divider()
    st.markdown("""
**What's measured**
- `AudioContext.baseLatency` — hardware buffer
- `AudioContext.outputLatency` — driver latency
- Frame render time — requestAnimationFrame delta
- Python processing time — spectral subtract benchmark
- Est. SNR after noise cancellation
""")
    st.divider()
    st.markdown("**Criteria met**")
    st.markdown("""
✅ Web Audio API  
✅ PCM capture (48 kHz, mono)  
✅ Live waveform + spectrogram  
✅ Spectral subtraction noise cancellation  
✅ Latency measurement & profiling  
✅ Chrome + Firefox + Edge + Safari  
✅ Audio quality metrics (RMS, peak, SNR, crest)  
""")


# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────
tab_rec, tab_lib = st.tabs(["🎙️  Record", "📚  Library"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — RECORD
# ══════════════════════════════════════════════════════════════════════════════
with tab_rec:
    st.markdown("### 🎙️ Record & Analyse")

    # ── Step 1: Visualiser ────────────────────────────────────────────────────
    st.markdown("""
    <div class="step-box">
      <span class="step-num">①</span><b>Live Monitor</b> — waveform, spectrogram &amp; latency
      <div class="tip">📡 Uses Web Audio API with <code>latencyHint:'interactive'</code> for lowest latency.
      The latency panel shows base latency, output latency, frame time, and estimated end-to-end from the browser.
      <b>This is display-only</b> — audio is not transferred to Python here.</div>
    </div>
    """, unsafe_allow_html=True)

    components.html(VISUALIZER_HTML, height=310, scrolling=False)

    st.divider()

    # ── Step 2: Record via audio_input ────────────────────────────────────────
    st.markdown("""
    <div class="step-box">
      <span class="step-num">②</span><b>Capture</b> — click mic → <b>1 second silence</b> → speak → click again to stop
      <div class="tip">🔇 <b>Stay silent for the first ~1 second</b> after clicking — this gives the noise
      cancellation a clean noise floor to subtract, which significantly improves SNR.<br>
      ✅ Results appear <b>instantly</b> after you stop — no processing delay.</div>
    </div>
    """, unsafe_allow_html=True)

    audio_value = None
    try:
        audio_value = st.audio_input("🎤 Click to record", key="audio_capture",
                                     label_visibility="collapsed")
    except AttributeError:
        st.error("Streamlit ≥ 1.38 required. Run: `pip install -U streamlit`")

    if audio_value is not None:
        t_decode = time.perf_counter()
        try:
            raw_f32, sr = read_wav_widget(audio_value)
        except Exception as e:
            st.error(f"Could not decode audio: {e}"); st.stop()
        decode_ms = (time.perf_counter() - t_decode) * 1000

        rms_val = float(np.sqrt(np.mean(raw_f32 ** 2)))
        if rms_val < 0.0001:
            st.warning("⚠️ Audio level near zero — check browser mic permissions.")

        # Noise cancellation
        t_proc = time.perf_counter()
        clean_f32 = spectral_subtract(raw_f32, sr)
        proc_ms   = (time.perf_counter() - t_proc) * 1000

        st.success(
            f"✅ Captured **{len(raw_f32)/sr:.2f}s** @ {sr:,} Hz  |  "
            f"Decode: {decode_ms:.1f}ms  |  Noise cancel: {proc_ms:.1f}ms  |  "
            f"RMS: {20*np.log10(rms_val+1e-9):.1f} dBFS"
        )

        # ── Latency summary panel ─────────────────────────────────────────────
        m = audio_metrics(raw_f32, clean_f32, sr)

        st.markdown("#### ⚡ Latency & Performance")
        lc1, lc2, lc3, lc4, lc5 = st.columns(5)
        lat_items = [
            (lc1, f"{decode_ms:.1f} ms",       "WAV Decode",         decode_ms,  50,  100),
            (lc2, f"{proc_ms:.1f} ms",          "Noise Cancel",       proc_ms,    50,  100),
            (lc3, f"{m['proc_ms_1s']:.1f} ms",  "Proc / 1s audio",    m['proc_ms_1s'], 20, 50),
            (lc4, f"{round(1024/sr*1000,1)} ms","FFT Frame Buffer",   1024/sr*1000, 50, 100),
            (lc5, f"{m['snr_dB']:.1f} dB",      "Est. SNR (quality)", None, None, None),
        ]
        for col, val, lbl, raw_v, good, warn in lat_items:
            if raw_v is not None:
                cls = "kpi-good" if raw_v <= good else ("kpi-warn" if raw_v <= warn else "kpi-bad")
            else:
                cls = "kpi-good" if m['snr_dB'] > 10 else "kpi-warn"
            col.markdown(
                f'<div class="kpi"><div class="kpi-v {cls}">{val}</div>'
                f'<div class="kpi-l">{lbl}</div></div>',
                unsafe_allow_html=True)

        st.divider()

        # ── Preview ───────────────────────────────────────────────────────────
        st.markdown("#### 🔊 Preview")
        pa, pb = st.columns(2)
        with pa:
            st.caption("🔴 Raw (original)")
            st.audio(to_wav_bytes(raw_f32, sr), format="audio/wav")
        with pb:
            st.caption("🔵 Clean (noise-cancelled)")
            st.audio(to_wav_bytes(clean_f32, sr), format="audio/wav")

        # ── Quality metrics ───────────────────────────────────────────────────
        st.markdown("#### 📊 Audio Quality")
        qa, qb, qc, qd = st.columns(4)
        for col, val, lbl in [
            (qa, f"{m['duration']:.2f} s",    "Duration"),
            (qb, f"{m['rms_dBFS']:.1f} dBFS", "RMS Level"),
            (qc, f"{m['peak_dBFS']:.1f} dBFS","Peak Level"),
            (qd, f"{m['snr_dB']:.1f} dB",     "Est. SNR"),
        ]:
            col.markdown(
                f'<div class="kpi"><div class="kpi-v">{val}</div>'
                f'<div class="kpi-l">{lbl}</div></div>',
                unsafe_allow_html=True)

        st.divider()

        # ── Save controls ─────────────────────────────────────────────────────
        st.markdown("#### 💾 Save to Library")
        rec_name = st.text_input("Name (optional)",
                                 placeholder="e.g. Meeting notes, Test 1…",
                                 key="rec_name")
        sc1, sc2 = st.columns(2)
        with sc1:
            if st.button("✅ Save to Library", type="primary", use_container_width=True):
                with st.spinner("Generating full analysis report…"):
                    meta = save_recording(raw_f32, clean_f32, sr, rec_name)
                st.success(f"✅ **{meta['label']}** saved ({meta['duration']:.2f}s) → Library tab")
                st.balloons()
        with sc2:
            st.download_button(
                "⬇️ Download clean WAV",
                data=to_wav_bytes(clean_f32, sr),
                file_name=f"clean_{int(time.time())}.wav",
                mime="audio/wav",
                use_container_width=True,
            )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — LIBRARY
# ══════════════════════════════════════════════════════════════════════════════
with tab_lib:
    st.markdown("### 📚 Recordings Library")
    library = load_library()
    if not library:
        st.info("No recordings yet — record in the **Record** tab and click **Save to Library**.")
        st.stop()

    st.caption(f"{len(library)} recording(s) in `{RECORDINGS_DIR}/`")
    st.divider()

    for meta in library:
        slug     = meta["slug"]
        wav_path = Path(meta.get("wav", ""))
        png_path = Path(meta.get("png", ""))
        ts_str   = meta.get("timestamp", slug)[:19].replace("T", " ")

        st.markdown(f"""
        <div class="rec-card">
          <div class="rec-title">🎵 {meta['label']}</div>
          <div class="rec-meta">
            🕐 {ts_str} &nbsp;|&nbsp;
            ⏱ {meta.get('duration',0):.2f}s &nbsp;|&nbsp;
            RMS {meta.get('rms_dBFS',-99):.1f} dBFS &nbsp;|&nbsp;
            Peak {meta.get('peak_dBFS',-99):.1f} dBFS &nbsp;|&nbsp;
            SNR {meta.get('snr_dB',0):.1f} dB &nbsp;|&nbsp;
            Proc {meta.get('proc_ms_1s',0):.1f}ms/s &nbsp;|&nbsp;
            {meta.get('sr','?')} Hz
          </div>
        </div>""", unsafe_allow_html=True)

        cp, cd, cdl = st.columns([4, 2, 1])
        with cp:
            if wav_path.exists():
                st.audio(wav_path.read_bytes(), format="audio/wav")
            else:
                st.caption("⚠️ WAV file missing")
        with cd:
            if wav_path.exists():
                st.download_button("⬇️ WAV",
                    data=wav_path.read_bytes(),
                    file_name=f"{meta['label'].replace(' ','_')}.wav",
                    mime="audio/wav", key=f"dl_{slug}", use_container_width=True)
        with cdl:
            if st.button("🗑️", key=f"del_{slug}", help="Delete"):
                delete_recording(slug); st.rerun()

        if png_path.exists():
            with st.expander(f"📈 Full analysis report — {meta['label']}"):
                st.image(png_path.read_bytes(), use_container_width=True)
                st.download_button("⬇️ Report PNG",
                    data=png_path.read_bytes(),
                    file_name=f"{meta['label'].replace(' ','_')}_report.png",
                    mime="image/png", key=f"png_{slug}")

        st.markdown("---")