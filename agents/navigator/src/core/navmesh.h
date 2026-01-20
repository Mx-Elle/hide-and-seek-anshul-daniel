#ifndef NAVIGATOR_SRC_CORE_NAVMESH_H_
#define NAVIGATOR_SRC_CORE_NAVMESH_H_

#include "triangle.h"
#include <cstdint>
#include <vector>

class NavMesh {
public:
  void Load(const float *verts, size_t num_verts, const int32_t *tris,
            size_t num_tris, const int32_t *neighbors);

  const Triangle &GetTriangle(uint16_t id) const { return triangles_[id]; }
  size_t NumTriangles() const { return triangles_.size(); }

private:
  std::vector<Triangle> triangles_;
};

#endif // NAVIGATOR_SRC_CORE_NAVMESH_H_
