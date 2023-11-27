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
#include <raft_runtime/neighbors/ivf_pq.hpp>
#include <raft_runtime/neighbors/refine.hpp>
#include <raft/util/cudart_utils.hpp>

#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/iterator/counting_iterator.h>

#include "common.cuh"

void ivf_pq_build_search_simple_shmoo(raft::device_resources const& dev_resources,
                                  raft::device_matrix_view<const float, int64_t> dataset,
                                  raft::device_matrix_view<const float, int64_t> queries)
{
  using namespace raft::neighbors;

  for (int shmoo_pq_bit = 4; shmoo_pq_bit <= 8; shmoo_pq_bit++) {
    for (int shmoo_pq_len = 1; shmoo_pq_len <= 32; shmoo_pq_len *= 2) {
      int shmoo_pq_dim = 96 / shmoo_pq_len; 
      if (shmoo_pq_bit * shmoo_pq_dim % 8 != 0) {
        continue; 
      }
      std::cout << shmoo_pq_bit << "," << shmoo_pq_dim << ",";
      ivf_pq::index_params index_params;
      index_params.n_lists                  = 50000;
      index_params.kmeans_n_iters           = 25; 
      index_params.kmeans_trainset_fraction = 0.1;  
      index_params.pq_bits                  = shmoo_pq_bit; 
      index_params.pq_dim                   = 96 / shmoo_pq_len;
      index_params.metric                   = raft::distance::DistanceType::L2Expanded;

      auto start = std::chrono::system_clock::now(); 
      auto index = ivf_pq::build(dev_resources, index_params, dataset);
      auto end = std::chrono::system_clock::now(); 
      auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
      std::cout << elapsed_ms.count() << ",";
      std::cout << index.n_lists() << "," 
                << index.size() << std::endl;

    }
  }
  
  // // Create output arrays.
  // int64_t topk      = 10;
  // int64_t n_queries = queries.extent(0);
  // auto neighbors    = raft::make_device_matrix<int64_t>(dev_resources, n_queries, topk);
  // auto distances    = raft::make_device_matrix<float>(dev_resources, n_queries, topk);

  // // Set search parameters.
  // ivf_pq::search_params search_params;
  // search_params.n_probes = 50;
  // search_params.lut_dtype = CUDA_R_16F; 
  // search_params.internal_distance_dtype = CUDA_R_16F; 

  // Search K nearest neighbors for each of the queries.
  // TODO: figure out why we have to use runtime API. 
  // raft::runtime::neighbors::ivf_pq::search(
  //   dev_resources, search_params, index, queries, neighbors.view(), distances.view());

  // The call to ivf_pq::search is asynchronous. Before accessing the data, sync by calling
  raft::resource::sync_stream(dev_resources);

  // print_results(dev_resources, neighbors.view(), distances.view());
}

// void ivf_pq_build_search_simple(raft::device_resources const& dev_resources,
//                                 raft::device_matrix_view<const float, int64_t> dataset,
//                                 raft::device_matrix_view<const float, int64_t> queries)
// {
//   using namespace raft::neighbors;

//   raft::neighbors::ivf_pq::index_params index_params;
//   index_params.n_lists                  = 50000;
//   index_params.kmeans_n_iters           = 1; // reduced 
//   index_params.kmeans_trainset_fraction = 0.1;  
//   index_params.pq_bits                  = 5; 
//   index_params.pq_dim                   = 96;
//   index_params.metric                   = raft::distance::DistanceType::L2Expanded;

//   std::cout << "Building IVF-PQ index" << std::endl;
//   auto index = ivf_pq::build(dev_resources, index_params, dataset);

//   std::cout << "Number of clusters " << index.n_lists() << ", number of vectors added to index "
//             << index.size() << std::endl;

//   // Create output arrays.
//   int64_t topk      = 10;
//   int64_t n_queries = queries.extent(0);
//   auto neighbors    = raft::make_device_matrix<int64_t>(dev_resources, n_queries, topk);
//   auto distances    = raft::make_device_matrix<float>(dev_resources, n_queries, topk);

//   // Set search parameters.
//   ivf_pq::search_params search_params;
//   search_params.n_probes = 50;
//   search_params.lut_dtype = CUDA_R_16F; 
//   search_params.internal_distance_dtype = CUDA_R_16F; 

//   // Create output arrays for refine. 
//   int64_t refine_ratio   = 2;
//   int64_t topk_refine    = refine_ratio * topk;  
//   auto neighbors_tmp     = raft::make_device_matrix<int64_t>(dev_resources, n_queries, topk_refine);
//   auto distances_tmp     = raft::make_device_matrix<float>(dev_resources, n_queries, topk_refine);

//   // Search K nearest neighbors for each of the queries.
//   raft::runtime::neighbors::ivf_pq::search(
//     dev_resources, search_params, index, queries, neighbors_tmp.view(), distances_tmp.view());

//   // Refine the search results 
//   raft::runtime::neighbors::refine(
//     dev_resources, dataset, queries, neighbors_tmp.view(), neighbors.view(), distances.view(), index_params.metric);

//   // The call to ivf_pq::search is asynchronous. Before accessing the data, sync by calling
//   raft::resource::sync_stream(dev_resources);

//   print_results(dev_resources, neighbors.view(), distances.view());
// }

// void ivf_pq_build_search_refine(raft::device_resources const& dev_resources,
//                                 raft::device_matrix_view<const float, int64_t> dataset,
//                                 raft::device_matrix_view<const float, int64_t> queries)
// {
//   using namespace raft::neighbors;

//   raft::neighbors::ivf_pq::index_params index_params;
//   index_params.n_lists                  = 50000;
//   index_params.kmeans_n_iters           = 25; 
//   index_params.kmeans_trainset_fraction = 0.1;  
//   index_params.pq_bits                  = 5; 
//   index_params.pq_dim                   = 96;
//   index_params.metric                   = raft::distance::DistanceType::L2Expanded;

//   std::cout << "Building IVF-PQ index" << std::endl;
//   auto index = ivf_pq::build(dev_resources, index_params, dataset);

//   std::cout << "Number of clusters " << index.n_lists() << ", number of vectors added to index "
//             << index.size() << std::endl;

//   // Create output arrays.
//   int64_t topk      = 10;
//   int64_t n_queries = queries.extent(0);
//   auto neighbors    = raft::make_device_matrix<int64_t>(dev_resources, n_queries, topk);
//   auto distances    = raft::make_device_matrix<float>(dev_resources, n_queries, topk);

//   // Set search parameters.
//   ivf_pq::search_params search_params;
//   search_params.n_probes = 50;
//   search_params.lut_dtype = CUDA_R_16F; 
//   search_params.internal_distance_dtype = CUDA_R_16F; 

//   // Create output arrays for refine. 
//   int64_t refine_ratio   = 2;
//   int64_t topk_refine    = refine_ratio * topk;  
//   auto neighbors_tmp     = raft::make_device_matrix<int64_t>(dev_resources, n_queries, topk_refine);
//   auto distances_tmp     = raft::make_device_matrix<float>(dev_resources, n_queries, topk_refine);

//   // Search K nearest neighbors for each of the queries.
//   raft::runtime::neighbors::ivf_pq::search(
//     dev_resources, search_params, index, queries, neighbors_tmp.view(), distances_tmp.view());

//   // Refine the search results 
//   raft::runtime::neighbors::refine(
//     dev_resources, dataset, queries, neighbors_tmp.view(), neighbors.view(), distances.view(), index_params.metric);

//   // The call to ivf_pq::search is asynchronous. Before accessing the data, sync by calling
//   raft::resource::sync_stream(dev_resources);

//   print_results(dev_resources, neighbors.view(), distances.view());
// }

// void ivf_pq_build_extend_search(raft::device_resources const& dev_resources,
//                                   raft::device_matrix_view<const float, int64_t> dataset,
//                                   raft::device_matrix_view<const float, int64_t> queries)
// {
//   using namespace raft::neighbors;

//   // Define dataset indices.
//   auto data_indices = raft::make_device_vector<int64_t, int64_t>(dev_resources, dataset.extent(0));
//   thrust::counting_iterator<int64_t> first(0);
//   thrust::device_ptr<int64_t> ptr(data_indices.data_handle());
//   thrust::copy(
//     raft::resource::get_thrust_policy(dev_resources), first, first + dataset.extent(0), ptr);

//   // Sub-sample the dataset to create a training set.
//   auto trainset =
//     subsample(dev_resources, dataset, raft::make_const_mdspan(data_indices.view()), 0.1);

//   ivf_pq::index_params index_params;
//   index_params.n_lists           = 100;
//   index_params.metric            = raft::distance::DistanceType::L2Expanded;
//   index_params.add_data_on_build = false;

//   std::cout << "\nRun k-means clustering using the training set" << std::endl;
//   auto index =
//     ivf_pq::build(dev_resources, index_params, raft::make_const_mdspan(trainset.view()));

//   std::cout << "Number of clusters " << index.n_lists() << ", number of vectors added to index "
//             << index.size() << std::endl;

//   std::cout << "Filling index with the dataset vectors" << std::endl;
//   index = ivf_pq::extend(dev_resources,
//                            dataset,
//                            std::make_optional(raft::make_const_mdspan(data_indices.view())),
//                            index);

//   std::cout << "Index size after addin dataset vectors " << index.size() << std::endl;

//   // Set search parameters.
//   ivf_pq::search_params search_params;
//   search_params.n_probes = 10;

//   // Create output arrays.
//   int64_t topk      = 10;
//   int64_t n_queries = queries.extent(0);
//   auto neighbors    = raft::make_device_matrix<int64_t, int64_t>(dev_resources, n_queries, topk);
//   auto distances    = raft::make_device_matrix<float, int64_t>(dev_resources, n_queries, topk);

//   // Search K nearest neighbors for each queries.
//   ivf_pq::search(
//     dev_resources, search_params, index, queries, neighbors.view(), distances.view());

//   // The call to ivf_pq::search is asynchronous. Before accessing the data, sync using:
//   // raft::resource::sync_stream(dev_resources);

//   print_results(dev_resources, neighbors.view(), distances.view());
// }

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
  int64_t n_samples = 100'000'000;
  int64_t n_dim     = 96;
  int64_t n_queries = 10;
  auto dataset      = raft::make_device_matrix<float, int64_t>(dev_resources, n_samples, n_dim);
  auto queries      = raft::make_device_matrix<float, int64_t>(dev_resources, n_queries, n_dim);
  std::cout <<"generating dataset..."; 
  generate_dataset(dev_resources, dataset.view(), queries.view());
  std::cout <<". Finished\n"; 

  // Simple build and search example.
  ivf_pq_build_search_simple_shmoo(dev_resources,
                               raft::make_const_mdspan(dataset.view()),
                               raft::make_const_mdspan(queries.view()));

  // Build, search, and refine the search results.
  // ivf_pq_build_search_refine(dev_resources,
  //                              raft::make_const_mdspan(dataset.view()),
  //                              raft::make_const_mdspan(queries.view()));
//   // Build and extend example.
//   ivf_pq_build_extend_search(dev_resources,
//                                raft::make_const_mdspan(dataset.view()),
//                                raft::make_const_mdspan(queries.view()));
}

