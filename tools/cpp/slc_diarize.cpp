// Offline pyannote community-1 diarization of one block of audio, via the
// cpp-annote port vendored in the moonshine repo.
// Usage: slc-diarize SEGMENTATION.ort EMBEDDING.ort AUDIO.f32
//   AUDIO.f32: raw little-endian float32 mono samples at 16 kHz.
// Prints JSON: {"turns": [{"start": s, "end": s, "speaker": k}, ...]}
#include <cstdio>
#include <fstream>
#include <iostream>
#include <vector>

#include "cpp-annote.h"

int main(int argc, char** argv) {
  if (argc != 4) {
    std::cerr << "usage: slc-diarize SEGMENTATION.ort EMBEDDING.ort AUDIO.f32\n";
    return 2;
  }
  std::ifstream f(argv[3], std::ios::binary | std::ios::ate);
  if (!f) {
    std::cerr << "cannot open " << argv[3] << "\n";
    return 1;
  }
  const std::streamsize bytes = f.tellg();
  f.seekg(0);
  std::vector<float> audio(static_cast<size_t>(bytes) / sizeof(float));
  f.read(reinterpret_cast<char*>(audio.data()), audio.size() * sizeof(float));
  try {
    cppannote::CppAnnote engine(argv[1], argv[2]);
    cppannote::DiarizationResults r = engine.diarize(audio.data(), audio.size(), 16000);
    std::cout << "{\"turns\": [";
    for (size_t i = 0; i < r.turns.size(); ++i) {
      const auto& t = r.turns[i];
      std::printf("%s{\"start\": %.3f, \"end\": %.3f, \"speaker\": %d}",
                  i ? ", " : "", t.start, t.end, t.speaker);
      std::fflush(stdout);
    }
    std::cout << "]}\n";
  } catch (const std::exception& e) {
    std::cerr << "diarization failed: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
