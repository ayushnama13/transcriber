#!/usr/bin/env bash
# Build bin/slc-diarize from the cpp-annote sources vendored in the moonshine repo.
# Usage: tools/build_diarizer.sh [MOONSHINE_REPO]   (default: ../moonshine)
#
# Links against the onnxruntime-gpu wheel in .venv when present (CUDA EP,
# enabled at runtime with SLC_ORT_CUDA=1), else the vendored CPU-only ORT.
# The vendored cpp-annote.cpp is never modified: a patched copy is compiled.
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
ms="$(cd "${1:-$here/../moonshine}" && pwd)"
core="$ms/core"; ca="$core/cpp-annote/src"; tp="$core/third-party"

gpu_capi="$(ls -d "$here"/.venv/lib/python3*/site-packages/onnxruntime/capi 2>/dev/null | head -1 || true)"
mkdir -p "$here/bin/lib" "$here/bin/obj"
if [ -n "$gpu_capi" ] && [ -f "$gpu_capi/libonnxruntime_providers_cuda.so" ]; then
  cp -f "$gpu_capi"/libonnxruntime.so.1.* "$here/bin/lib/libonnxruntime.so.1"
  cp -f "$gpu_capi"/libonnxruntime_providers_{shared,cuda}.so "$here/bin/lib/"
  echo "using onnxruntime-gpu from $gpu_capi"
else
  ort="$tp/onnxruntime/lib/linux/$(uname -m)"
  [ -f "$ort/libonnxruntime.so.1" ] || { echo "no onnxruntime lib at $ort" >&2; exit 1; }
  cp -f "$ort/libonnxruntime.so.1" "$here/bin/lib/"
  echo "using vendored CPU onnxruntime"
fi
ln -sf libonnxruntime.so.1 "$here/bin/lib/libonnxruntime.so"

# Patched copy: opt-in CUDA EP on the shared SessionOptions (segmentation
# session is created first, so the embedding session inherits it too).
patched="$here/bin/obj/cpp-annote.patched.cpp"
if [ ! "$patched" -nt "$ca/cpp-annote.cpp" ]; then
  python3 - "$ca/cpp-annote.cpp" "$patched" <<'PY'
import sys
src = open(sys.argv[1]).read()
anchor = '  require_model(model, "segmentation");\n'
assert src.count(anchor) == 1, "cpp-annote.cpp changed; update the CUDA patch"
hook = ('  if (const char* e = std::getenv("SLC_ORT_CUDA"); e && e[0] == \'1\') {\n'
        '    OrtCUDAProviderOptions cuda{};\n'
        '    opts.AppendExecutionProvider_CUDA(cuda);\n'
        '  }\n')
open(sys.argv[2], "w").write(src.replace(anchor, anchor + hook))
PY
fi

srcs=("$patched"
      "$ca"/{cpp-annote-streaming,community1_cpp_annote_embedded,compute_fbank,embedding_ort_infer,filter_train,clustering_vbx,plda_vbx,scipy_linkage,parity_log}.cpp
      "$tp"/kissfft/{kiss_fft,kiss_fftr}.c
      "$tp"/kaldi-native-fbank/csrc/{feature-fbank,feature-functions,feature-mfcc,feature-raw-audio-samples,feature-window,istft,kaldi-math,mel-computations,online-feature,rfft,stft,whisper-feature}.cc)
objs=(); pids=()
for s in "${srcs[@]}"; do
  o="$here/bin/obj/$(basename "$s").o"; objs+=("$o")
  [ "$o" -nt "$s" ] && continue
  case "$s" in *.c) cc=(gcc -O2 -w) ;; *) cc=(g++ -std=c++20 -O2 -w) ;; esac
  "${cc[@]}" -DEIGEN_MPL2_ONLY=1 -I"$ca" -I"$tp" -I"$tp/kissfft" -I"$tp/onnxruntime/include" -c "$s" -o "$o" &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
g++ -std=c++20 -O2 -I"$ca" "$here/tools/cpp/slc_diarize.cpp" "${objs[@]}" \
    -L"$here/bin/lib" -lonnxruntime -Wl,-rpath,'$ORIGIN/lib' -lpthread -o "$here/bin/slc-diarize"
echo "built $here/bin/slc-diarize"
