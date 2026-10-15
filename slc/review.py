"""Review web app: 7-10 checkers each take a part, listen, fix text and names.

    python -m slc review WORKDIR [--library LIB] [--host 0.0.0.0] [--port 8000]

Every change is applied to transcript.json (atomic write) and appended to
edits.jsonl (who, when, before -> after), so nothing a reviewer does is lost
and `learn` / `export` can run afterwards.
"""
import io
import json
import os
import threading
import time

import soundfile as sf

from slc.audio import duration_s, read_segment

AUDIO_PAD_S = 0.4
MAX_CLIP_S = 180.0
EDITABLE = {"text": str, "speaker": str, "reviewed": bool}


class Store:
    def __init__(self, workdir, library=None):
        self.workdir = workdir
        self.path = os.path.join(workdir, "transcript.json")
        self.edits = os.path.join(workdir, "edits.jsonl")
        self.state_path = os.path.join(workdir, "review_state.json")
        self.lock = threading.Lock()
        with open(self.path) as f:
            self.t = json.load(f)
        self.by_id = {s["id"]: s for s in self.t["segments"]}
        self.state = {"claims": {}}
        if os.path.exists(self.state_path):
            with open(self.state_path) as f:
                self.state = json.load(f)
        self.library_names = []
        if library:
            from slc.voicelib import VoiceLibrary
            self.library_names = VoiceLibrary(library).names()
        self.channels = sorted(int(f[3:-4]) for f in os.listdir(os.path.join(workdir, "aligned"))
                               if f.startswith("rec") and f.endswith(".wav"))
        self.duration = duration_s(self.wav("mix"))

    def wav(self, ch):
        name = "mix.wav" if ch == "mix" else f"rec{int(ch)}.wav"
        return os.path.join(self.workdir, "aligned", name)

    def _save_json(self, path, obj):
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(obj, f, indent=1)
        os.replace(tmp, path)

    def _log(self, rec):
        rec["time"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        with open(self.edits, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def speakers(self):
        return sorted(set(self.library_names) | {s["speaker"] for s in self.t["segments"]})

    def parts(self):
        out = {}
        for s in self.t["segments"]:
            p = out.setdefault(s["part"], {"part": s["part"], "start": s["start"], "lines": 0,
                                           "flagged": 0, "reviewed": 0})
            p["start"] = min(p["start"], s["start"])
            p["lines"] += 1
            p["flagged"] += bool(s["flags"])
            p["reviewed"] += bool(s.get("reviewed"))
        for p in out.values():
            p["claimed_by"] = self.state["claims"].get(str(p["part"]), "")
        return [out[k] for k in sorted(out)]

    def part_segments(self, part):
        return sorted((s for s in self.t["segments"] if s["part"] == part), key=lambda s: (s["start"], s["id"]))

    def update(self, seg_id, changes, reviewer):
        seg = self.by_id.get(seg_id)
        if seg is None:
            raise KeyError(seg_id)
        clean = {}
        for k, v in changes.items():
            if k not in EDITABLE or not isinstance(v, EDITABLE[k]):
                raise ValueError(f"bad field {k!r}")
            if k == "speaker":
                v = v.strip()
                if not v:
                    raise ValueError("empty speaker name")
            clean[k] = v
        with self.lock:
            before = {k: seg.get(k) for k in clean}
            seg.update(clean)
            if clean.get("reviewed"):
                seg["reviewed_by"] = reviewer
            self._save_json(self.path, self.t)
            self._log({"reviewer": reviewer, "segment": seg_id, "before": before, "after": clean})
        return seg

    def rename_cluster(self, cluster, name, reviewer):
        """Name every not-yet-reviewed line of a diarization cluster at once."""
        name = name.strip()
        if not name:
            raise ValueError("empty speaker name")
        with self.lock:
            ids = [s["id"] for s in self.t["segments"] if s["cluster"] == cluster and not s.get("reviewed")]
            for i in ids:
                self.by_id[i]["speaker"] = name
            self._save_json(self.path, self.t)
            self._log({"reviewer": reviewer, "cluster": cluster, "after": {"speaker": name}, "segments": ids})
        return ids

    def claim(self, part, reviewer):
        with self.lock:
            self.state["claims"][str(part)] = reviewer
            self._save_json(self.state_path, self.state)

    def clip(self, seg_id, ch):
        seg = self.by_id[seg_id]
        a = max(0.0, seg["start"] - AUDIO_PAD_S)
        b = min(self.duration, seg["end"] + AUDIO_PAD_S, a + MAX_CLIP_S)
        buf = io.BytesIO()
        sf.write(buf, read_segment(self.wav(ch), a, b), 16000, format="WAV", subtype="PCM_16")
        return buf.getvalue()


def create_app(workdir, library=None):
    from flask import Flask, Response, abort, jsonify, request

    store = Store(workdir, library)
    app = Flask(__name__)
    app.config["store"] = store

    def reviewer():
        r = (request.get_json(silent=True) or {}).get("reviewer", "").strip()
        if not r:
            abort(400, "reviewer name required")
        return r[:60]

    @app.get("/")
    def index():
        return Response(PAGE, mimetype="text/html")

    @app.get("/api/meta")
    def meta():
        return jsonify(meeting=store.t.get("meeting", ""), parts=store.parts(),
                       speakers=store.speakers(), channels=store.channels)

    @app.get("/api/part/<int:part>")
    def part(part):
        segs = store.part_segments(part)
        if not segs:
            abort(404)
        return jsonify(segments=segs)

    @app.post("/api/part/<int:part>/claim")
    def claim(part):
        store.claim(part, reviewer())
        return jsonify(ok=True)

    @app.post("/api/segment/<int:seg_id>")
    def update(seg_id):
        body = request.get_json(silent=True) or {}
        try:
            seg = store.update(seg_id, body.get("changes", {}), reviewer())
        except KeyError:
            abort(404)
        except ValueError as e:
            abort(400, str(e))
        return jsonify(segment=seg)

    @app.post("/api/cluster/<cluster>")
    def cluster(cluster):
        body = request.get_json(silent=True) or {}
        try:
            ids = store.rename_cluster(cluster, body.get("name", ""), reviewer())
        except ValueError as e:
            abort(400, str(e))
        return jsonify(updated=ids)

    @app.get("/audio/<int:seg_id>")
    def audio(seg_id):
        ch = request.args.get("ch", "mix")
        if seg_id not in store.by_id or not (ch == "mix" or (ch.isdigit() and int(ch) in store.channels)):
            abort(404)
        return Response(store.clip(seg_id, ch), mimetype="audio/wav")

    return app


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Transcript Review</title>
<style>
:root{--bg:#fafaf8;--fg:#1d1d1b;--mute:#6b6b66;--line:#e2e1dc;--card:#fff;--flag:#fff3e0;--flagb:#e8a33d;--ok:#e8f5e9;--okb:#4f9a59;--acc:#2f5fb3}
@media (prefers-color-scheme:dark){:root{--bg:#191917;--fg:#ecebe6;--mute:#9b9a94;--line:#33322f;--card:#22221f;--flag:#3a2c15;--flagb:#c98a2b;--ok:#1e3121;--okb:#5fae69;--acc:#7fa6ee}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.45 system-ui,sans-serif}
header{position:sticky;top:0;background:var(--bg);border-bottom:1px solid var(--line);padding:10px 16px;display:flex;gap:12px;flex-wrap:wrap;align-items:center;z-index:2}
header h1{font-size:16px;margin:0 8px 0 0}input,select,button,textarea{font:inherit;color:inherit;background:var(--card);border:1px solid var(--line);border-radius:6px;padding:5px 8px}
button{cursor:pointer}button.primary{background:var(--acc);color:#fff;border-color:var(--acc)}
main{max-width:980px;margin:0 auto;padding:16px}table{border-collapse:collapse;width:100%}td,th{padding:6px 8px;border-bottom:1px solid var(--line);text-align:left}
.seg{background:var(--card);border:1px solid var(--line);border-left:4px solid var(--line);border-radius:8px;padding:10px 12px;margin:10px 0}
.seg.flagged{border-left-color:var(--flagb);background:var(--flag)}.seg.reviewed{border-left-color:var(--okb);background:var(--ok)}
.seg.current{outline:2px solid var(--acc)}.row{display:flex;gap:8px;flex-wrap:wrap;align-items:center}
.time{color:var(--mute);font-variant-numeric:tabular-nums}.badge{font-size:12px;border:1px solid var(--flagb);border-radius:10px;padding:0 7px}
textarea{width:100%;margin-top:8px;min-height:3em;resize:vertical}.mute{color:var(--mute);font-size:13px}.hidden{display:none}
</style></head><body>
<header><h1 id="title">Review</h1>
<label>Your name <input id="who" size="12" placeholder="required"></label>
<span id="nav"></span></header>
<main id="main"></main>
<script>
const $=s=>document.querySelector(s), el=(t,a={},...k)=>{const e=document.createElement(t);Object.assign(e,a);e.append(...k);return e};
let META=null, SEGS=[], CUR=0, PART=0, FF=false, AUDIO=new Audio();
try{$('#who').value=localStorage.getItem('who')||''}catch(e){}
$('#who').onchange=()=>{try{localStorage.setItem('who',$('#who').value.trim())}catch(e){}};
const who=()=>{const w=$('#who').value.trim(); if(!w){alert('Enter your name first');$('#who').focus();throw 'no name'} return w};
const hms=t=>{t=Math.floor(t);return [t/3600|0,(t%3600)/60|0,t%60].map(x=>String(x).padStart(2,'0')).join(':')};
async function api(url,body){const r=await fetch(url,body?{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}:{});
  if(!r.ok) throw new Error((await r.text()).replace(/<[^>]+>/g,' ').trim()); return r.json()}
async function loadMeta(){META=await api('/api/meta'); $('#title').textContent=META.meeting||'Review'}
async function home(){await loadMeta(); $('#nav').replaceChildren();
  const tb=el('tbody'); for(const p of META.parts){
    const go=el('button',{textContent:'Open',onclick:()=>openPart(p.part)});
    const cl=el('button',{textContent:p.claimed_by?'Take over':'Claim',onclick:async()=>{await api(`/api/part/${p.part}/claim`,{reviewer:who()});home()}});
    tb.append(el('tr',{},el('td',{textContent:`Part ${p.part+1}`}),el('td',{className:'time',textContent:hms(p.start)}),
      el('td',{textContent:`${p.reviewed}/${p.lines}`}),el('td',{textContent:p.flagged}),el('td',{textContent:p.claimed_by||'—'}),el('td',{},cl,' ',go)))}
  $('#main').replaceChildren(el('p',{className:'mute',textContent:'Claim a part, open it, listen to each line, fix the words and the name, then confirm. Flagged lines (orange) need the most attention.'}),
    el('table',{},el('thead',{},el('tr',{},...['Part','From','Reviewed','Flagged','Checker',''].map(h=>el('th',{textContent:h})))),tb))}
function speakerSelect(cur){const s=el('select'); for(const n of META.speakers) s.append(el('option',{value:n,textContent:n,selected:n===cur}));
  s.append(el('option',{value:'__new',textContent:'+ new name…'})); return s}
async function openPart(part){await loadMeta(); const d=await api(`/api/part/${part}`); SEGS=d.segments; CUR=0;
  const flaggedFirst=el('input',{type:'checkbox',onchange:()=>render(part,flaggedFirst.checked)});
  const chan=el('select',{id:'chan'},el('option',{value:'auto',textContent:'Best mic'}),el('option',{value:'mix',textContent:'Mix'}),...META.channels.map(c=>el('option',{value:c,textContent:`Recorder ${c+1}`})));
  $('#nav').replaceChildren(el('button',{textContent:'← Parts',onclick:home}),` Part ${part+1} `,el('label',{},flaggedFirst,' flagged first'),' Audio ',chan,
    el('span',{className:'mute',textContent:'  Ctrl+Enter confirm & next · Alt+P play'}));
  render(part,false)}
function render(part,flaggedFirst){PART=part; FF=flaggedFirst; const list=[...SEGS]; if(flaggedFirst) list.sort((a,b)=>(!!b.flags.length-!!a.flags.length)||(a.start-b.start));
  const byId=Object.fromEntries(SEGS.map(s=>[s.id,s]));
  $('#main').replaceChildren(...list.map((s,i)=>card(s,i,byId))); CUR=0; focusCard(0)}
function cls(s){return 'seg'+(s.reviewed?' reviewed':s.flags.length?' flagged':'')}
function card(s,i,byId){
  const sel=speakerSelect(s.speaker), txt=el('textarea',{value:s.text}), status=el('span',{className:'mute'});
  const box=el('div',{className:cls(s)}); box.dataset.i=i;
  const play=el('button',{textContent:'▶ Play',onclick:()=>playSeg(s)});
  async function save(reviewed){let sp=sel.value; if(sp==='__new'){sp=(prompt('Speaker name')||'').trim(); if(!sp) return}
    const ch={text:txt.value.trim(),speaker:sp}; if(reviewed!==undefined) ch.reviewed=reviewed;
    try{const r=await api(`/api/segment/${s.id}`,{reviewer:who(),changes:ch}); Object.assign(s,r.segment); box.className=cls(s);
      if(!META.speakers.includes(sp)) META.speakers.push(sp); status.textContent='saved '+new Date().toLocaleTimeString()}catch(e){status.textContent='error: '+e.message}}
  const allCl=el('button',{textContent:"Name this voice everywhere…",title:'Set this name on every unreviewed line of this voice',onclick:async()=>{
    const n=(prompt(`Name for every unreviewed line of voice ${s.cluster}`,s.speaker)||'').trim(); if(!n) return;
    const r=await api(`/api/cluster/${encodeURIComponent(s.cluster)}`,{reviewer:who(),name:n}); status.textContent=`renamed ${r.updated.length} lines`;
    if(!META.speakers.includes(n)) META.speakers.push(n); for(const x of SEGS) if(r.updated.includes(x.id)) x.speaker=n;
    render(PART,FF) }});
  const badges=s.flags.map(f=>el('span',{className:'badge',textContent:f.replace(/_/g,' ')}));
  const others=(s.overlaps_with||[]).map(o=>byId[o]).filter(Boolean).map(o=>`${o.speaker}: “${o.text.slice(0,80)}”`);
  box.append(el('div',{className:'row'},el('span',{className:'time',textContent:`${hms(s.start)}–${hms(s.end)}`}),play,sel,...badges,
      el('span',{className:'mute',textContent:`conf ${s.speaker_confidence}`})),
    txt, others.length?el('div',{className:'mute',textContent:'Talking over: '+others.join(' | ')}):'',
    el('div',{className:'row'},el('button',{className:'primary',textContent:'✓ Confirm',onclick:()=>save(true).then(()=>focusCard(i+1))}),
      el('button',{textContent:'Save',onclick:()=>save()}),s.reviewed?el('button',{textContent:'Un-confirm',onclick:()=>save(false)}):'',allCl,status));
  box.addEventListener('focusin',()=>{CUR=i;document.querySelectorAll('.seg.current').forEach(c=>c.classList.remove('current'));box.classList.add('current')});
  box._save=save; box._seg=s; return box}
function focusCard(i){const c=document.querySelectorAll('.seg')[i]; if(!c) return; c.querySelector('textarea').focus({preventScroll:true}); c.scrollIntoView({block:'center',behavior:'smooth'})}
function playSeg(s){const v=$('#chan')?$('#chan').value:'auto', ch=v==='auto'?s.channel:v;
  if(AUDIO.dataset.k===`${s.id}/${ch}`&&!AUDIO.paused){AUDIO.pause();return}
  AUDIO.src=`/audio/${s.id}?ch=${ch}`; AUDIO.dataset.k=`${s.id}/${ch}`; AUDIO.play()}
document.addEventListener('keydown',e=>{const c=document.querySelectorAll('.seg')[CUR]; if(!c) return;
  if(e.ctrlKey&&e.key==='Enter'){e.preventDefault(); c._save(true).then(()=>focusCard(CUR+1))}
  if(e.altKey&&(e.key==='p'||e.key==='P')){e.preventDefault(); playSeg(c._seg)}});
home();
</script></body></html>
"""
