// Host reference for the common examples: TensorFlow Lite Micro itself, built with the host compiler from the same
// sources and flags as the target library (tools/tflm_host.py).  Runs a model on CSV inputs, writes CSV outputs.
// usage: runner MODEL.tflite int8|float INPUT.csv OUTPUT.csv ARENA_KB      (ops: ops.h of the model, -include)
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"

static std::vector<char> slurp(const char *p)
{
  FILE *f = std::fopen(p, "rb");
  if (!f) { std::perror(p); std::exit(2); }
  std::vector<char> v;
  char b[65536];
  size_t n;
  while ((n = std::fread(b, 1, sizeof b, f)) > 0) v.insert(v.end(), b, b + n);
  std::fclose(f);
  return v;
}

int main(int argc, char **argv)
{
  if (argc != 6) { std::fprintf(stderr, "usage: %s MODEL int8|float IN.csv OUT.csv ARENA_KB\n", argv[0]); return 2; }
  std::vector<char> model = slurp(argv[1]);
  alignas(16) static unsigned char *arena = nullptr;
  size_t arena_size = std::atoi(argv[5]) * 1024;
  arena = static_cast<unsigned char *>(std::aligned_alloc(16, arena_size));
  bool int8 = std::strcmp(argv[2], "int8") == 0;
  std::vector<char> in = slurp(argv[3]);
  in.push_back(0);
  std::vector<double> vals;
  for (char *s = std::strtok(in.data(), ",\n"); s; s = std::strtok(nullptr, ",\n")) vals.push_back(std::atof(s));

  tflite::MicroMutableOpResolver<MODEL_NUM_OPS> r;
#define ADD_OP(name) r.Add##name();
  MODEL_OPS(ADD_OP)
  tflite::MicroInterpreter it(tflite::GetModel(model.data()), r, arena, arena_size);
  if (it.AllocateTensors() != kTfLiteOk) { std::fprintf(stderr, "AllocateTensors failed\n"); return 1; }
  FILE *out = std::fopen(argv[4], "w");
  bool first = true;
  for (size_t t = 0; t < vals.size() / MODEL_INPUT_SIZE; t++) {
    for (int i = 0; i < MODEL_INPUT_SIZE; i++) {
      if (int8) it.input(0)->data.int8[i] = static_cast<int8_t>(vals[t * MODEL_INPUT_SIZE + i]);
      else it.input(0)->data.f[i] = static_cast<float>(vals[t * MODEL_INPUT_SIZE + i]);
    }
    if (it.Invoke() != kTfLiteOk) { std::fprintf(stderr, "Invoke failed\n"); return 1; }
    for (int o = 0; o < MODEL_OUTPUT_SIZE; o++) {
      if (int8) std::fprintf(out, "%s%d", first ? "" : ",", it.output(0)->data.int8[o]);
      else std::fprintf(out, "%s%.9g", first ? "" : ",", it.output(0)->data.f[o]);
      first = false;
    }
  }
  std::fprintf(out, "\n");
  std::fclose(out);
  return 0;
}
