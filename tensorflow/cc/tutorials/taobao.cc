#undef NDEBUG

#include <dlfcn.h>
#include <stdio.h>
#include <unistd.h>

#include <cstdio>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/compiler/jit/flags.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/gpu/gpu_event_mgr.h"
#include "tensorflow/core/common_runtime/threadpool_device.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"

using tensorflow::int32;
using tensorflow::string;

#define BATCH_SIZE 64   // default batch size
#define INFER_NUM 1000  // default infer iterations for each stream
#define NUM_STREAMS \
  2  // default total infer iterations num will be NUM_STREAMS * INFER_NUM
#define MAX_NUM_STREAMS 1024

const std::string four_times_bz_names[] = {
    "CTR_SubModel/Attention_Network/pvfeature_seq/cross/multihead_attention/"
    "Tile/user_tile/Tile_0_cuda_graph_input",
    "CTR_SubModel/Attention_Network/his_clk_seq/cross/multihead_attention/Tile/"
    "user_tile/Tile_0_cuda_graph_input",
    "CTR_SubModel/Attention_Network/his_cart_seq/cross/multihead_attention/"
    "Tile/user_tile/Tile_0_cuda_graph_input",
    "CTR_SubModel/Attention_Network/his_ord_seq/cross/multihead_attention/Tile/"
    "user_tile/Tile_0_cuda_graph_input",
    "CTR_SubModel/Attention_Network/query_clk_seq/cross/multihead_attention/"
    "Tile/user_tile/Tile_0_cuda_graph_input",
    "CTR_SubModel/Attention_Network/user_trigger_seq/cross/multihead_attention/"
    "Tile/user_tile/Tile_0_cuda_graph_input"};

const std::string first_dim_equal_one_names[] = {
    "ph2",
};

static bool is_four_times_bz(const std::string& name) {
  for (int i = 0; i < 6; i++) {
    if (name == four_times_bz_names[i]) return true;
  }
  return false;
}

static bool is_first_dim_equals_one(const std::string& name) {
  for (int i = 0; i < 1; i++) {
    if (name == first_dim_equal_one_names[i]) return true;
  }
  return false;
}

namespace tensorflow {

// cpu allocator
static Allocator* host_allocator = nullptr;
std::string model_name;
std::string tensor_file;

namespace example {

typedef std::vector<std::pair<std::string, Tensor>> InputsMap;

// thread function for multiple thread TF session run
void TFRun(Session* sess, int infer_num,
           std::vector<std::pair<std::string, Tensor>>* inputs,
           std::vector<std::string>* output_names,
           std::vector<Tensor>* output_tensors) {
  for (int i = 0; i < infer_num; i++) {
    TF_CHECK_OK(sess->Run(*inputs, *output_names, {}, output_tensors));
  }
}

void TFCallableRun(Session* sess, Session::CallableHandle* callable_handler,
                   int infer_num, std::vector<Tensor>* inputs,
                   std::vector<Tensor>* output_tensors) {
  for (int i = 0; i < infer_num; i++) {
    TF_CHECK_OK(
        sess->RunCallable(*callable_handler, *inputs, output_tensors, nullptr));
  }
}

TensorShape getNodeShape(const GraphDef& graph_def, const std::string name,
                         int batch_size) {
  for (int i = 0; i < graph_def.node_size(); i++) {
    auto n = graph_def.node(i);
    if (n.name() == name) {
      auto shape = n.attr().at("shape").shape();
      int dims = shape.dim_size();
      TensorShape tensorShape;

      for (int d = 0; d < dims; d++) {
        int dim_size = shape.dim(d).size();

        if (d == 0 && dim_size == -1) {
          int new_size = batch_size;

          if (is_four_times_bz(name)) {
            new_size *= 4;
          }

          if (is_first_dim_equals_one(name)) {
            new_size = 1;
          }

          // assume the first dimension is batch size, note that it may not be
          // true for some models. LOG(INFO) << "change batch size from: " <<
          // dim_size << " to " << new_size << std::endl;
          dim_size = new_size;
        }
        tensorShape.AddDim(dim_size);
      }

      return tensorShape;
    }
  }
  LOG(ERROR) << "Cannot find the node " << name << std::endl;
  exit(1);
}

DataType getNodeType(const GraphDef& graph_def, const std::string name) {
  for (int i = 0; i < graph_def.node_size(); i++) {
    auto n = graph_def.node(i);
    if (n.name() == name) {
      auto dtype = n.attr().at("dtype").type();
      return dtype;
    }
  }
  LOG(ERROR) << "Cannot find the node" << name << std::endl;
  exit(1);
}

void RandomInitialize(Tensor& t) {
  int num_elements = t.NumElements();
  if (t.dtype() == DT_HALF) {
    auto* data = t.flat<Eigen::half>().data();
    for (int i = 0; i < num_elements; i++) {
      float value = static_cast<float>(rand() % 101 - 50) / 100.0f;
      data[i] = static_cast<Eigen::half>(value);
    }
  } else if (t.dtype() == DT_FLOAT) {
    float* data = t.flat<float>().data();
    for (int i = 0; i < num_elements; i++) {
      float value = static_cast<float>(rand() % 101 - 50) / 100.0f;
      data[i] = value;
    }
  } else if (t.dtype() == DT_INT32) {
    int* data = t.flat<int>().data();
    for (int i = 0; i < num_elements; i++) {
      // int value = static_cast<int>(rand() % 10);
      int value = 1;
      data[i] = value;
    }
  } else if (t.dtype() == DT_BOOL) {
    bool* data = t.flat<bool>().data();
    for (int i = 0; i < num_elements; i++) {
      bool value = static_cast<bool>(rand() % 2);
      data[i] = value;
    }
  } else if (t.dtype() == DT_INT64) {
    int64* data = t.flat<int64>().data();
    for (int i = 0; i < num_elements; i++) {
      // int64 value = static_cast<int64>(rand() % 10);
      int64 value = 1;
      data[i] = value;
    }
  } else {
    std::cout << t.dtype() << std::endl;
    std::cout << "Random init: unsupported data type." << std::endl;
  }
}

void PrintTensorData(Tensor& t) {
  void* data;
  if (t.dtype() == DT_HALF) {
    data = static_cast<void*>(t.flat<Eigen::half>().data());
  } else if (t.dtype() == DT_FLOAT) {
    data = static_cast<void*>(t.flat<float>().data());
  } else if (t.dtype() == DT_BOOL) {
    data = static_cast<void*>(t.flat<bool>().data());
  } else if (t.dtype() == DT_INT32) {
    data = static_cast<void*>(t.flat<int>().data());
  } else {
    std::cout << "Print Tensor: Unsupported data type!" << std::endl;
    return;
  }

  int dims = t.dims();
  std::cout << "shape: " << std::endl;
  for (int i = 0; i < dims; i++) {
    std::cout << t.dim_size(i) << ", ";
  }
  std::cout << std::endl;

  int size = t.NumElements();
  size = size > 32 ? 32 : size;

  for (int i = 0; i < size; i++) {
    float value;
    if (t.dtype() == DT_HALF) {
      value = static_cast<float>(static_cast<Eigen::half*>(data)[i]);
    } else if (t.dtype() == DT_INT32) {
      value = static_cast<int*>(data)[i];
    } else if (t.dtype() == DT_BOOL) {
      value = static_cast<bool*>(data)[i];
    } else {
      value = static_cast<float*>(data)[i];
    }
    std::cout << value << ", ";
  }
  std::cout << std::endl;
}

void PrintTensorFile(Tensor& t, std::string file_name) {
  void* data;
  if (t.dtype() == DT_HALF) {
    data = static_cast<void*>(t.flat<Eigen::half>().data());
  } else if (t.dtype() == DT_FLOAT) {
    data = static_cast<void*>(t.flat<float>().data());
  } else if (t.dtype() == DT_BOOL) {
    data = static_cast<void*>(t.flat<bool>().data());
  } else if (t.dtype() == DT_INT32) {
    data = static_cast<void*>(t.flat<int>().data());
  } else {
    std::cout << "Print Tensor: Unsupported data type!" << std::endl;
    return;
  }

  int dims = t.dims();
  std::cout << "shape: " << std::endl;
  for (int i = 0; i < dims; i++) {
    std::cout << t.dim_size(i) << ", ";
  }
  std::cout << std::endl;

  int size = t.NumElements();

  FILE* fp = fopen((char*)file_name.data(), "wb");
  for (int i = 0; i < size; i++) {
    float value;
    if (t.dtype() == DT_HALF) {
      value = static_cast<float>(static_cast<Eigen::half*>(data)[i]);
    } else if (t.dtype() == DT_INT32) {
      value = static_cast<int*>(data)[i];
    } else if (t.dtype() == DT_BOOL) {
      value = static_cast<bool*>(data)[i];
    } else {
      value = static_cast<float*>(data)[i];
    }
    fprintf(fp, "%f\n", value);
  }
}

void GenerateInputs(GraphDef& graph_def, const std::vector<string>& input_names,
                    std::vector<Tensor>& input_tensors, int batch_size) {
  input_tensors.clear();
  for (int i = 0; i < input_names.size(); i++) {
    auto tensorshape = getNodeShape(graph_def, input_names[i], batch_size);
    auto tensortype = getNodeType(graph_def, input_names[i]);

    Tensor t;
    if (host_allocator) {
      t = Tensor(host_allocator, tensortype, tensorshape);
    } else {
      t = Tensor(tensortype, tensorshape);
    }
    RandomInitialize(t);

    input_tensors.push_back(t);
  }
}

void FillInputsMap(InputsMap& inputs_map, std::vector<std::string>& input_names,
                   std::vector<Tensor>& input_tensors) {
  assert(input_names.size() == input_tensors.size());

  for (size_t i = 0; i < input_tensors.size(); i++) {
    inputs_map.push_back(
        std::pair<std::string, Tensor>(input_names[i], input_tensors[i]));
  }
}

void CopyTensorContents(Tensor& dst_tensor, Tensor& src_tensor) {
  // assert(dst_tensor.AllocatedBytes() == src_tensor.AllocatedBytes());
  int dst_eles = dst_tensor.NumElements();
  int src_eles = src_tensor.NumElements();
  assert(dst_eles == src_eles);

  assert(dst_tensor.dtype() == src_tensor.dtype());

  char *dst, *src;
  int ele_size;
  if (dst_tensor.dtype() == DT_HALF) {
    dst = reinterpret_cast<char*>(dst_tensor.flat<Eigen::half>().data());
    src = reinterpret_cast<char*>(src_tensor.flat<Eigen::half>().data());
    ele_size = 2;
  } else if (dst_tensor.dtype() == DT_FLOAT) {
    dst = reinterpret_cast<char*>(dst_tensor.flat<float>().data());
    src = reinterpret_cast<char*>(src_tensor.flat<float>().data());
    ele_size = 4;
  } else if (dst_tensor.dtype() == DT_INT32) {
    dst = reinterpret_cast<char*>(dst_tensor.flat<int>().data());
    src = reinterpret_cast<char*>(src_tensor.flat<int>().data());
    ele_size = 4;
  } else if (dst_tensor.dtype() == DT_BOOL) {
    dst = reinterpret_cast<char*>(dst_tensor.flat<bool>().data());
    src = reinterpret_cast<char*>(src_tensor.flat<bool>().data());
    ele_size = 1;
  } else if (dst_tensor.dtype() == DT_INT64) {
    dst = reinterpret_cast<char*>(dst_tensor.flat<int64>().data());
    src = reinterpret_cast<char*>(src_tensor.flat<int64>().data());
    ele_size = 8;
  } else {
    std::cout << "Copy Tensor: Unsupported data type!" << std::endl;
    return;
  }

  for (int i = 0; i < src_eles * ele_size; i++) {
    dst[i] = src[i];
  }
}

inline void SetDevice(const string& device, GraphDef* graph_def) {
  for (int i = 0; i < graph_def->node_size(); ++i) {
    auto node = graph_def->mutable_node(i);
    // if (node->device().empty()) {
    node->set_device(device);
    //}
  }
}

CallableOptions MakeCallableOptions(std::vector<string> feeds,
                                    std::vector<string> fetches,
                                    std::vector<string> targets) {
  CallableOptions ret;
  for (const string& feed : feeds) {
    ret.add_feed(feed);
  }
  for (const string& fetch : fetches) {
    ret.add_fetch(fetch);
  }
  for (const string& target : targets) {
    ret.add_target(target);
  }
  return ret;
}

void SetSessionOptions(SessionOptions& options, string& config_proto) {
  ConfigProto* config = &options.config;
  if (!config_proto.empty()) {
    Status s = ReadTextProto(Env::Default(), config_proto.c_str(), config);
    if (!s.ok()) {
      s = ReadBinaryProto(Env::Default(), config_proto.c_str(), config);
      if (!s.ok()) {
        LOG(ERROR) << "Read config proto from file " << config_proto
                   << " failed: " << s.ToString()
                   << ". Use default ConfigProto.";
        return;
      }
    }
    VLOG(1) << "Read config proto: " << config->DebugString();

    if (config->graph_options().optimizer_options().global_jit_level() ==
        tensorflow::OptimizerOptions::ON_1) {
      // Use XLA.
      auto* flags = tensorflow::GetMarkForCompilationPassFlags();
      flags->tf_xla_cpu_global_jit = true;
      flags->tf_xla_min_cluster_size = 1;
      tensorflow::SetXlaAutoJitFlagFromFlagString("single-gpu(2)");
    }
  }
}

Status Test(GraphDef& graph_def, std::vector<std::string>& input_names,
            std::vector<std::string>& output_names, int batch_size,
            int num_infers_per_stream, int num_streams, string config_proto) {
  assert(num_streams <= MAX_NUM_STREAMS);

  // Creates a session.
  SessionOptions options;
  SetSessionOptions(options, config_proto);

  std::unique_ptr<Session> session(NewSession(options));

  // if (options.target.empty()) {
  //     graph::SetDefaultDevice("/device:GPU:0", &graph_def);
  // }
  SetDevice("/device:GPU:0", &graph_def);

  TF_CHECK_OK(session->Create(graph_def));

  const DeviceMgr* device_manager;
  TF_CHECK_OK(session->LocalDeviceManager(&device_manager));
  std::vector<Device*> devices = device_manager->ListDevices();
  for (auto* d : devices) {
    if (d->name().find("CPU") != std::string::npos) {
      std::cout << "CPU device:" << d->name() << std::endl;
      host_allocator = dynamic_cast<ThreadPoolDevice*>(d)->GetAllocator(
          AllocatorAttributes());
      break;
    }
  }

  std::vector<Tensor> input_tensors_tf;  // input tensors for Normal TF runs
  GenerateInputs(graph_def, input_names, input_tensors_tf, batch_size);

  // first normal TF session run
  InputsMap inputs_tf;  // input map for Normal TF run
  FillInputsMap(inputs_tf, input_names, input_tensors_tf);

  // TF Multiple threads runs
  // Run session.run in multiple threads
  // The number of threads are same with num_streams
  const int num_infers_per_thread = num_infers_per_stream;
  const int num_threads = num_streams;
  std::vector<Tensor> output_tensors_tf[MAX_NUM_STREAMS];
  std::vector<std::thread> threads;

  // CallableOptions opts = MakeCallableOptions(input_names, output_names,
  // std::vector<string>()); Session::CallableHandle callable_handler;
  // TF_CHECK_OK(session->MakeCallable(opts, &callable_handler));

  // init run
  for (int i = 0; i < num_threads; i++) {
    threads.push_back(
        std::thread(TFRun, session.get(),
                    num_infers_per_thread < 200 ? num_infers_per_thread : 200,
                    &inputs_tf, &output_names, &output_tensors_tf[i]));
  }
  for (auto& thread : threads) {
    thread.join();
  }
  threads.clear();
  sleep(1);

  // the second session run can be used to compare single thread runing
  auto start = std::chrono::system_clock::now();
  TFRun(session.get(), num_infers_per_thread, &inputs_tf, &output_names,
        &output_tensors_tf[0]);
  auto end = std::chrono::system_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  double duration_seconds = duration.count() * 1.0 / 1000;
  LOG(INFO) << "[TF + Single Thread] Duration = " << duration_seconds
            << " seconds." << std::endl;
  double qps = num_infers_per_thread * 1.0 / duration_seconds;
  LOG(INFO) << "[TF + Single Thread] QPS = " << qps << std::endl;
  sleep(1);

  std::ofstream fout("tmp_single.csv", std::ios::app);
  fout << model_name << "," << qps << "," << duration_seconds << std::endl;
  fout.close();

  start = std::chrono::system_clock::now();
  for (int i = 0; i < num_threads; i++) {
    threads.push_back(std::thread(TFRun, session.get(), num_infers_per_thread,
                                  &inputs_tf, &output_names,
                                  &output_tensors_tf[i]));
    // threads.push_back(std::thread(TFCallableRun, session.get(),
    // &callable_handler, num_infers_per_thread,  &input_tensors_tf,
    // &output_tensors_tf[i]));
  }
  for (auto& thread : threads) {
    thread.join();
  }
  end = std::chrono::system_clock::now();

  for (int i = 0; i < num_threads; i++) {
    LOG(INFO) << "TF results: ";
    PrintTensorData(output_tensors_tf[i][0]);  // print first output tensor
    if (!tensor_file.empty()) {
      PrintTensorFile(output_tensors_tf[i][0],
                      "tensor_" + tensor_file + "_" + std::to_string(i) +
                          ".txt");  // print first output tensor to file
    }
    output_tensors_tf[i].clear();
  }

  duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  duration_seconds = duration.count() * 1.0 / 1000;
  LOG(INFO) << "[TF + Multiple Threads] Duration = " << duration_seconds
            << " seconds." << std::endl;
  qps = num_infers_per_thread * num_threads * 1.0 / duration_seconds;
  LOG(INFO) << "[TF + Multiple Threads] Reference Average QPS = " << qps
            << std::endl;

  fout.open("tmp_multi.csv", std::ios::app);
  fout << model_name << "," << qps << "," << duration_seconds << std::endl;
  fout.close();

  TF_CHECK_OK(session->Close());
  return OkStatus();
}

}  // end namespace example

}  // end namespace tensorflow

using namespace tensorflow;

int main(int argc, char* argv[]) {
  // Example: ./application model_path in_num input_name [input_names]
  //                        out_num output_name [output_names]
  //                        batch_size infer_num_per_stream num_streams
  //                        model_name use_xla number_of_threadpool
  //                        number_of_threads_in_one_threadpool
  // read command line arguments
  int arg_idx = 1;
  std::string model_path = argv[arg_idx++];
  int input_num = std::stoi(argv[arg_idx++]);
  assert(input_num >= 1);

  std::vector<std::string> input_names;
  std::cout << input_num << " inputs: ";
  for (int i = 0; i < input_num; i++) {
    input_names.push_back(argv[arg_idx++]);
    std::cout << argv[arg_idx - 1] << ",";
  }
  std::cout << std::endl;

  int output_num = std::stoi(argv[arg_idx++]);
  assert(output_num >= 1);
  std::vector<std::string> output_names;
  std::cout << output_num << " outputs: ";
  for (int i = 0; i < output_num; i++) {
    output_names.push_back(argv[arg_idx++]);
    std::cout << argv[arg_idx - 1] << ",";
  }
  std::cout << std::endl;

  // std::string mode = argv[arg_idx++];
  // std::cout << "mode = " << mode << std::endl;

  int batch_size = BATCH_SIZE;
  if (argc > arg_idx) {
    batch_size = std::stoi(argv[arg_idx++]);
    assert(batch_size >= 1);
  }
  std::cout << "batch size = " << batch_size << std::endl;

  int num_infers_per_stream = INFER_NUM;
  if (argc > arg_idx) {
    num_infers_per_stream = std::stoi(argv[arg_idx++]);
    assert(num_infers_per_stream >= 1);
  }
  std::cout << "num_infers_per_stream = " << num_infers_per_stream << std::endl;

  int num_streams = NUM_STREAMS;
  if (argc > arg_idx) {
    num_streams = std::stoi(argv[arg_idx++]);
    assert(num_streams >= 1);
    assert(num_streams <= MAX_NUM_STREAMS);
  }
  std::cout << "num_streams = " << num_streams << std::endl;

  // if(argc > arg_idx){
  //     const char * custom_op_lib = argv[arg_idx++];
  //     dlopen(custom_op_lib, RTLD_LAZY);
  //     std::cout << "with custom op lib: " << custom_op_lib << std::endl;
  // }

  if (argc > arg_idx) {
    model_name = argv[arg_idx++];
  }

  std::string config_proto;
  if (argc > arg_idx) {
    config_proto = argv[arg_idx++];
  }
  std::cout << "config_proto = " << config_proto << std::endl;

  if (argc > arg_idx) {
    tensor_file = argv[arg_idx++];
  }
  // command line arguments reading done.

  GraphDef graph_def;
  Status status;

  if (model_path.find(".pbtxt") == std::string::npos) {
    status = ReadBinaryProto(Env::Default(), model_path, &graph_def);
  } else {
    status = ReadTextProto(Env::Default(), model_path, &graph_def);
  }

  if (!status.ok()) {
    std::cout << status.ToString() << "\n";
    return 1;
  }

  TF_CHECK_OK(example::Test(graph_def, input_names, output_names, batch_size,
                            num_infers_per_stream, num_streams, config_proto));
  return 0;
}
