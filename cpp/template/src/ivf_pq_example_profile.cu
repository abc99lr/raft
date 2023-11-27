/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cstdint>
#include <optional>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/resource/thrust_policy.hpp>
#include <raft/neighbors/ivf_pq.cuh>
#include <raft/neighbors/refine.cuh>
// #include <raft_runtime/neighbors/ivf_pq.hpp>
// #include <raft_runtime/neighbors/refine.hpp>
#include <raft/util/cudart_utils.hpp>

#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/iterator/counting_iterator.h>

#include "common.cuh"

void ivf_pq_build_search_simple(raft::device_resources const& dev_resources,
                                  raft::device_matrix_view<const float, int64_t> dataset,
                                  raft::device_matrix_view<const float, int64_t> queries)
{
  using namespace raft::neighbors;

  raft::neighbors::ivf_pq::index_params index_params;
  index_params.n_lists                  = 5000;
  index_params.kmeans_n_iters           = 1; 
  index_params.kmeans_trainset_fraction = 1;  
  index_params.pq_bits                  = 5; 
  index_params.pq_dim                   = 96;
  index_params.metric                   = raft::distance::DistanceType::L2Expanded;

  std::cout << "Building IVF-PQ index" << std::endl;
  auto index = ivf_pq::build(dev_resources, index_params, dataset);

  std::cout << "Number of clusters " << index.n_lists() << ", number of vectors added to index "
            << index.size() << std::endl;

  // The call to ivf_pq::search is asynchronous. Before accessing the data, sync by calling
  raft::resource::sync_stream(dev_resources);

  // print_results(dev_resources, neighbors.view(), distances.view());
}

int main()
{
  raft::device_resources dev_resources;

  // Set pool memory resource with 1 GiB initial pool size. All allocations use the same pool.
  rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource> pool_mr(
    rmm::mr::get_current_device_resource(), 1024 * 1024 * 1024ull);
  rmm::mr::set_current_device_resource(&pool_mr);

  // Alternatively, one could define a pool allocator for temporary arrays (used within RAFT
  // algorithms). In that case only the internal arrays would use the pool, any other allocation
  // uses the default RMM memory resource. Here is how to change the workspace memory resource to
  // a pool with 2 GiB upper limit.
  // raft::resource::set_workspace_to_pool_resource(dev_resources, 2 * 1024 * 1024 * 1024ull);

  // Create input arrays.
  int64_t n_samples = 10'000'000;
  int64_t n_dim     = 96;
  int64_t n_queries = 10;
  auto dataset      = raft::make_device_matrix<float, int64_t>(dev_resources, n_samples, n_dim);
  auto queries      = raft::make_device_matrix<float, int64_t>(dev_resources, n_queries, n_dim);
  std::cout <<"generating dataset..."; 
  generate_dataset(dev_resources, dataset.view(), queries.view());
  std::cout <<". Finished\n"; 

  // Simple build and search example.
  ivf_pq_build_search_simple(dev_resources,
                               raft::make_const_mdspan(dataset.view()),
                               raft::make_const_mdspan(queries.view()));
}
