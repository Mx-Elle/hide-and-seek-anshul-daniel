#ifndef NAVIGATOR_SRC_ALGORITHM_BVH_H_
#define NAVIGATOR_SRC_ALGORITHM_BVH_H_

#include "../core/navmesh.h"
#include <vector>

struct BvhNode {
  AABB bounds;
  uint32_t left_or_tri;
  uint32_t right;
  bool IsLeaf() const { return right == 0; }
};

struct alignas(16) QBvhNode {
  float min_x[4], min_y[4], max_x[4], max_y[4];
  int32_t children[4];
  int32_t count;
};

class TriangleBvh {
public:
  void Build(const NavMesh &mesh);
  uint16_t FindTriangle(Vec2 p) const;

private:
  uint32_t BuildRecursive(std::vector<uint16_t> &ids, int start, int end,
                          std::vector<BvhNode> &nodes);
  void CollapseToQuad(uint32_t bin_idx, std::vector<QBvhNode> &q_nodes,
                      const std::vector<BvhNode> &bin_nodes);

  std::vector<QBvhNode> nodes_;
  const NavMesh *mesh_ = nullptr;
};

#endif // NAVIGATOR_SRC_ALGORITHM_BVH_H_
