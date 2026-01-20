#include "navigator.h"
#include "algorithm/funnel.h"

void Navigator::LoadMesh(const float *verts, size_t num_verts,
                         const int32_t *tris, size_t num_tris,
                         const int32_t *nbrs) {
  mesh_.Load(verts, num_verts, tris, num_tris, nbrs);
  bvh_.Build(mesh_);
  apsp_.Build(mesh_);
}

std::vector<Vec2> Navigator::FindPath(float sx, float sy, float gx, float gy) {
  Vec2 start{sx, sy};
  Vec2 goal{gx, gy};

  uint16_t start_tri = bvh_.FindTriangle(start);
  uint16_t goal_tri = bvh_.FindTriangle(goal);

  if (start_tri == kNoNeighbor || goal_tri == kNoNeighbor)
    return {};
  if (start_tri == goal_tri)
    return {start, goal};

  std::vector<uint16_t> corridor = apsp_.FindPath(start_tri, goal_tri);
  if (corridor.empty())
    return {};

  return Funnel::StringPull(start, goal, corridor, mesh_);
}
