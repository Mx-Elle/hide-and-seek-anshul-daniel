#include "apsp.h"

#include <cmath>
#include <limits>

#include "../core/vec2_simd.h"

void AllPairsShortestPath::Build(const NavMesh &mesh) {
  num_nodes_ = mesh.NumTriangles();
  size_t n = num_nodes_;

  if (n > 3000) {
    next_hop_.clear();
    return;
  }

  dist_matrix_.assign(n * n, std::numeric_limits<float>::max());
  next_hop_.assign(n * n, kNoNeighbor);

  for (uint16_t i = 0; i < n; ++i) {
    dist_matrix_[i * n + i] = 0;
    next_hop_[i * n + i] = i;

    const auto &t = mesh.GetTriangle(i);
    Vec2 centroid_i = t.Centroid();

    std::vector<Vec2> neighbor_centroids;
    std::vector<uint16_t> neighbor_ids;

    for (int k = 0; k < 3; ++k) {
      uint16_t nbr = t.neighbors[k];
      if (nbr != kNoNeighbor) {
        neighbor_centroids.push_back(mesh.GetTriangle(nbr).Centroid());
        neighbor_ids.push_back(nbr);
      }
    }

    size_t neighbor_count = neighbor_centroids.size();
    if (neighbor_count > 0) {
      std::vector<Vec2> diffs(neighbor_count);
      std::vector<Vec2> centroid_copies(neighbor_count, centroid_i);

      vec2_simd::BatchSub(neighbor_centroids.data(), centroid_copies.data(),
                          diffs.data(), neighbor_count);

      std::vector<float> length_sq(neighbor_count);
      vec2_simd::BatchLengthSq(diffs.data(), length_sq.data(), neighbor_count);

      for (size_t j = 0; j < neighbor_count; ++j) {
        float d = std::sqrt(length_sq[j]);
        dist_matrix_[i * n + neighbor_ids[j]] = d;
        next_hop_[i * n + neighbor_ids[j]] = neighbor_ids[j];
      }
    }
  }

  for (size_t k = 0; k < n; ++k) {
    for (size_t i = 0; i < n; ++i) {
      float dik = dist_matrix_[i * n + k];
      if (dik == std::numeric_limits<float>::max())
        continue;

      uint16_t next_ik = next_hop_[i * n + k];
      size_t row_i_idx = i * n;
      size_t row_k_idx = k * n;

      for (size_t j = 0; j < n; ++j) {
        float dkj = dist_matrix_[row_k_idx + j];
        if (dkj == std::numeric_limits<float>::max())
          continue;

        if (dik + dkj < dist_matrix_[row_i_idx + j]) {
          dist_matrix_[row_i_idx + j] = dik + dkj;
          next_hop_[row_i_idx + j] = next_ik;
        }
      }
    }
  }
}

std::vector<uint16_t> AllPairsShortestPath::FindPath(uint16_t start,
                                                     uint16_t end) const {
  if (start == end)
    return {start};
  if (next_hop_.empty())
    return {};

  if (next_hop_[start * num_nodes_ + end] == kNoNeighbor)
    return {};

  std::vector<uint16_t> path;
  path.reserve(64);
  path.push_back(start);

  uint16_t curr = start;
  int safety = 0;
  while (curr != end && safety++ < 10000) {
    curr = next_hop_[curr * num_nodes_ + end];
    if (curr == kNoNeighbor)
      return {};
    path.push_back(curr);
  }

  return path;
}
