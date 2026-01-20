#ifndef NAVIGATOR_SRC_NAVIGATOR_H_
#define NAVIGATOR_SRC_NAVIGATOR_H_

#include "algorithm/apsp.h"
#include "algorithm/bvh.h"
#include "algorithm/funnel.h"
#include "core/navmesh.h"
#include <array>
#include <vector>

class Navigator {
public:
  void LoadMesh(const float *verts, size_t num_verts, const int32_t *tris,
                size_t num_tris, const int32_t *nbrs);

  std::vector<Vec2> FindPath(float sx, float sy, float gx, float gy);

  std::vector<uint16_t> FindCorridor(float sx, float sy, float gx, float gy);

  std::array<Vec2, 3> GetTriangleVertices(uint16_t id) const {
    const auto &t = mesh_.GetTriangle(id);
    return {t.v[0], t.v[1], t.v[2]};
  }

  int32_t FindTriangle(float x, float y) const {
    uint16_t id = bvh_.FindTriangle({x, y});
    return (id == kNoNeighbor) ? -1 : static_cast<int32_t>(id);
  }

  size_t NumTriangles() const { return mesh_.NumTriangles(); }
  bool IsReady() const { return mesh_.NumTriangles() > 0; }

private:
  NavMesh mesh_;
  TriangleBvh bvh_;
  AllPairsShortestPath apsp_;
};

#endif // NAVIGATOR_SRC_NAVIGATOR_H_
