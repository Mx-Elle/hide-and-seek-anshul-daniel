#ifndef NAVIGATOR_SRC_ALGORITHM_APSP_H_
#define NAVIGATOR_SRC_ALGORITHM_APSP_H_

#include "../core/navmesh.h"
#include <cstdint>
#include <vector>

class AllPairsShortestPath {
public:
  void Build(const NavMesh &mesh);
  std::vector<uint16_t> FindPath(uint16_t start, uint16_t end) const;

  const std::vector<float> &GetDistanceMatrix() const { return dist_matrix_; }
  bool IsValid() const { return !next_hop_.empty(); }

private:
  std::vector<uint16_t> next_hop_;
  std::vector<float> dist_matrix_;
  size_t num_nodes_ = 0;
};

#endif // NAVIGATOR_SRC_ALGORITHM_APSP_H_
