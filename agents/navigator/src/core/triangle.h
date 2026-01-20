#ifndef NAVIGATOR_SRC_CORE_TRIANGLE_H_
#define NAVIGATOR_SRC_CORE_TRIANGLE_H_

#include "aabb.h"
#include "vec2.h"
#include <algorithm>
#include <cstdint>

constexpr uint16_t kNoNeighbor = 0xFFFF;

struct alignas(32) Triangle {
  Vec2 v[3];
  uint16_t id;
  uint16_t neighbors[3];

  Vec2 Centroid() const {
    return {(v[0].x + v[1].x + v[2].x) / 3.0f,
            (v[0].y + v[1].y + v[2].y) / 3.0f};
  }

  bool Contains(Vec2 p) const {
    return TriArea2(v[0], v[1], p) >= -1e-6f &&
           TriArea2(v[1], v[2], p) >= -1e-6f &&
           TriArea2(v[2], v[0], p) >= -1e-6f;
  }

  AABB Bounds() const {
    return {
        std::min({v[0].x, v[1].x, v[2].x}), std::min({v[0].y, v[1].y, v[2].y}),
        std::max({v[0].x, v[1].x, v[2].x}), std::max({v[0].y, v[1].y, v[2].y})};
  }
};

#endif // NAVIGATOR_SRC_CORE_TRIANGLE_H_
