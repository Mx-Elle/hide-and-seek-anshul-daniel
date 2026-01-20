#include "navmesh.h"

void NavMesh::Load(const float *verts, size_t num_verts, const int32_t *tris,
                   size_t num_tris, const int32_t *neighbors) {
  triangles_.clear();
  triangles_.reserve(num_tris);

  for (size_t i = 0; i < num_tris; ++i) {
    Triangle t;
    t.id = static_cast<uint16_t>(i);

    for (int j = 0; j < 3; ++j) {
      int32_t vi = tris[i * 3 + j];
      t.v[j] = {verts[vi * 2], verts[vi * 2 + 1]};

      int32_t ni = neighbors[i * 3 + j];
      t.neighbors[j] = (ni == -1) ? kNoNeighbor : static_cast<uint16_t>(ni);
    }

    if (TriArea2(t.v[0], t.v[1], t.v[2]) < 0) {
      std::swap(t.v[1], t.v[2]);
      uint16_t n0 = t.neighbors[0], n1 = t.neighbors[1], n2 = t.neighbors[2];
      t.neighbors[0] = n2;
      t.neighbors[1] = n1;
      t.neighbors[2] = n0;
    }
    triangles_.push_back(t);
  }
}
