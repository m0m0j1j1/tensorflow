/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#if (defined(GOOGLE_CUDA) && GOOGLE_CUDA) || \
    (defined(TENSORFLOW_USE_ROCM) && TENSORFLOW_USE_ROCM)

#define EIGEN_USE_GPU

#include "tensorflow/core/common_runtime/gpu/gpu_device.h"
#include "tensorflow/core/common_runtime/gpu/gpu_process_state.h"
#include "tensorflow/core/common_runtime/threadpool_device.h"
#include "tensorflow/core/platform/numa.h"
#include "tensorflow/tsl/framework/device_id.h"

namespace tensorflow {

class GPUDevice : public BaseGPUDevice {
 public:
  GPUDevice(const SessionOptions& options, const string& name,
            Bytes memory_limit, const DeviceLocality& locality,
            tsl::TfDeviceId tf_device_id, const string& physical_device_desc,
            Allocator* gpu_allocator, Allocator* cpu_allocator)
      : BaseGPUDevice(options, name, memory_limit, locality, tf_device_id,
                      physical_device_desc, gpu_allocator, cpu_allocator,
                      false /* sync every op */),
        gpu_options_(options.config.gpu_options()) {
    if (options.config.has_gpu_options()) {
      force_gpu_compatible_ =
          options.config.gpu_options().force_gpu_compatible();
      string host_allocator_mode = options.config.gpu_options()
                                       .multi_stream_options()
                                       .host_allocator_mode();
      if (!host_allocator_mode.empty()) {
        if (host_allocator_mode == "private") {
          host_allocator_mode_ = host_allocator_mode;
        } else if (host_allocator_mode != "shared") {
          string error_message = strings::StrCat(
              "Invalid host_allocator_mode: ", host_allocator_mode);
          LOG(WARNING) << error_message;
        }
      }
    }
  }

  Allocator* GetAllocator(AllocatorAttributes attr) override {
    CHECK(cpu_allocator_) << "bad place 1";
    if (attr.on_host()) {
      if (attr.gpu_compatible() || force_gpu_compatible_) {
        GPUProcessState* ps = GPUProcessState::singleton();
        return ps->GetGpuHostAllocator(
            gpu_options_, 0,
            host_allocator_mode_ == "private" ? stream_id_ : 0);
      } else {
        return cpu_allocator_;
      }
    } else {
      return gpu_allocator_;
    }
  }

 private:
  GPUOptions gpu_options_;
  bool force_gpu_compatible_ = false;
  string host_allocator_mode_ = "shared";
};

//-------------------------------------------------------------------------------
// A StreamDevice that manages multiple stream groups in GPU. TF will create as
// many StreamDevice objects as there are stream groups. The objects will be
// created only when there are more than one stream groups.
// ------------------------------------------------------------------------------
class StreamDevice : public GPUDevice {
 public:
  StreamDevice(const SessionOptions& options, const string& name,
               Bytes memory_limit, const DeviceLocality& locality,
               tsl::TfDeviceId tf_device_id, const string& physical_device_desc,
               Allocator* gpu_allocator, Allocator* cpu_allocator)
      : GPUDevice(options, name, memory_limit, locality, tf_device_id,
                  physical_device_desc, gpu_allocator, cpu_allocator) {}

  void SetRealDevice(Device* device) override { real_device_ = device; }

  const Device* GetRealDevice() const override { return real_device_; }

  ResourceMgr* resource_manager() override {
    return real_device_->resource_manager();
  }

 private:
  Device* real_device_;
};

class GPUDeviceFactory : public BaseGPUDeviceFactory {
 private:
  std::unique_ptr<BaseGPUDevice> CreateGPUDevice(
      const SessionOptions& options, const string& name, Bytes memory_limit,
      const DeviceLocality& locality, tsl::TfDeviceId tf_device_id,
      const string& physical_device_desc, Allocator* gpu_allocator,
      Allocator* cpu_allocator) override {
    return absl::make_unique<GPUDevice>(options, name, memory_limit, locality,
                                        tf_device_id, physical_device_desc,
                                        gpu_allocator, cpu_allocator);
  }
};

REGISTER_LOCAL_DEVICE_FACTORY("GPU", GPUDeviceFactory, 210);

class StreamDeviceFactory : public BaseGPUDeviceFactory {
 public:
  StreamDeviceFactory() { is_stream_factory_ = true; }

  Status ListPhysicalDevices(std::vector<string>* devices) override {
    // We don't know how many Stream Devices to create until we specify it
    // explicitly through the SessionConfig. Even if we did, a Stream Device
    // is not considered "physical". Therefore, we do nothing here.
    return OkStatus();
  }

 private:
  std::unique_ptr<BaseGPUDevice> CreateGPUDevice(
      const SessionOptions& options, const string& name, Bytes memory_limit,
      const DeviceLocality& locality, tsl::TfDeviceId tf_device_id,
      const string& physical_device_desc, Allocator* gpu_allocator,
      Allocator* cpu_allocator) override {
    return absl::make_unique<StreamDevice>(
        options, name, memory_limit, locality, tf_device_id,
        physical_device_desc, gpu_allocator, cpu_allocator);
  }
};

REGISTER_LOCAL_DEVICE_FACTORY("STREAM_GPU", StreamDeviceFactory);

//------------------------------------------------------------------------------
// A CPUDevice that optimizes for interaction with GPUs in the
// process.
// -----------------------------------------------------------------------------
class GPUCompatibleCPUDevice : public ThreadPoolDevice {
 public:
  GPUCompatibleCPUDevice(const SessionOptions& options, const string& name,
                         Bytes memory_limit, const DeviceLocality& locality,
                         Allocator* allocator)
      : ThreadPoolDevice(options, name, memory_limit, locality, allocator),
        numa_node_(locality.numa_node()),
        gpu_options_(options.config.gpu_options()) {
    if (options.config.has_gpu_options()) {
      force_gpu_compatible_ =
          options.config.gpu_options().force_gpu_compatible();
      string host_allocator_mode = options.config.gpu_options()
                                       .multi_stream_options()
                                       .host_allocator_mode();
      if (!host_allocator_mode.empty()) {
        if (host_allocator_mode == "private") {
          host_allocator_mode_ = host_allocator_mode;
        } else if (host_allocator_mode != "shared") {
          string error_message = strings::StrCat(
              "Invalid host_allocator_mode: ", host_allocator_mode);
          LOG(WARNING) << error_message;
        }
      }
    }
  }
  ~GPUCompatibleCPUDevice() override {}

  Allocator* GetAllocator(AllocatorAttributes attr) override {
    GPUProcessState* ps = GPUProcessState::singleton();
    if (attr.gpu_compatible() || force_gpu_compatible_) {
      return ps->GetGpuHostAllocator(
          gpu_options_, numa_node_,
          host_allocator_mode_ == "private" ? stream_id_ : 0);
    } else {
      // Call the parent's implementation.
      return ThreadPoolDevice::GetAllocator(attr);
    }
  }

 private:
  bool force_gpu_compatible_ = false;
  int numa_node_;
  GPUOptions gpu_options_;
  string host_allocator_mode_ = "shared";
};

//------------------------------------------------------------------------------
// A StreamCPUDevice that manages GPU host memory allocation when multiple
// stream groups in GPU is enabled. TF will create as many StreamCPUDevice
// objects as there are stream groups. The objects will be created only when
// there are more than one stream groups and host_allocator_mode is "private".
// -----------------------------------------------------------------------------
class StreamCompatibleCPUDevice : public GPUCompatibleCPUDevice {
 public:
  StreamCompatibleCPUDevice(const SessionOptions& options, const string& name,
                            Bytes memory_limit, const DeviceLocality& locality,
                            Allocator* allocator)
      : GPUCompatibleCPUDevice(options, name, memory_limit, locality,
                               allocator) {}
  ~StreamCompatibleCPUDevice() override {}

  void SetRealDevice(Device* device) override { real_device_ = device; }

  const Device* GetRealDevice() const override { return real_device_; }

  ResourceMgr* resource_manager() override {
    return real_device_->resource_manager();
  }

 private:
  Device* real_device_;
};

// The associated factory.
class GPUCompatibleCPUDeviceFactory : public DeviceFactory {
 public:
  virtual Status ListPhysicalDevices(std::vector<string>* devices) override {
    devices->push_back("/physical_device:CPU:0");

    return OkStatus();
  }

  Status CreateDevices(const SessionOptions& options, const string& name_prefix,
                       std::vector<std::unique_ptr<Device>>* devices) override {
    int n = 1;
    auto iter = options.config.device_count().find("CPU");
    if (iter != options.config.device_count().end()) {
      n = iter->second;
    }
    int num_numa_nodes = options.config.experimental().use_numa_affinity()
                             ? port::NUMANumNodes()
                             : 1;
    int gpu_stream_group_count = 1;
    if (is_stream_factory_) {
      if (options.config.gpu_options()
              .multi_stream_options()
              .host_allocator_mode() != "private") {
        return OkStatus();
      }
      gpu_stream_group_count = std::max(1, options.config.gpu_options()
                                               .multi_stream_options()
                                               .stream_group_count());
    }
    for (int i = 0; i < n; i++) {
      string name = strings::StrCat(name_prefix, "/device:CPU:", i);
      int numa_node = i % num_numa_nodes;
      DeviceLocality locality;
      locality.set_numa_node(numa_node);
      for (int j = 0; j < gpu_stream_group_count; ++j) {
        if (is_stream_factory_) {
          name = strings::StrCat(name_prefix, "/device:STREAM_CPU_", i, ":", j);
          devices->push_back(absl::make_unique<StreamCompatibleCPUDevice>(
              options, name, Bytes(256 << 20), DeviceLocality(),
              ProcessState::singleton()->GetCPUAllocator(numa_node)));
          devices->back()->SetStreamId(j);
        } else {
          devices->push_back(absl::make_unique<GPUCompatibleCPUDevice>(
              options, name, Bytes(256 << 20), DeviceLocality(),
              ProcessState::singleton()->GetCPUAllocator(numa_node)));
        }
      }
    }

    return OkStatus();
  }
};
REGISTER_LOCAL_DEVICE_FACTORY("CPU", GPUCompatibleCPUDeviceFactory, 70);

class StreamCompatibleCPUDeviceFactory : public GPUCompatibleCPUDeviceFactory {
 public:
  StreamCompatibleCPUDeviceFactory() { is_stream_factory_ = true; }

  Status ListPhysicalDevices(std::vector<string>* devices) override {
    // We don't know how many StreamCPU Devices to create until we specify it
    // explicitly through the SessionConfig. Even if we did, a Stream Device is
    // not considered "physical". Therefore, we do nothing here.
    return OkStatus();
  }
};
REGISTER_LOCAL_DEVICE_FACTORY("STREAM_CPU", StreamCompatibleCPUDeviceFactory);

}  // namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
