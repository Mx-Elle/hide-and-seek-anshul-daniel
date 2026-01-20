#ifndef NAVIGATOR_SRC_CORE_AABB_H_
#define NAVIGATOR_SRC_CORE_AABB_H_

#include "vec2.h"
#include <algorithm>

struct AABB {
  float min_x, min_y, max_x, max_y;

  static AABB Merge(const AABB &a, const AABB &b) {
    return {std::min(a.min_x, b.min_x), std::min(a.min_y, b.min_y),
            std::max(a.max_x, b.max_x), std::max(a.max_y, b.max_y)};
  }
};

#endif // NAVIGATOR_SRC_CORE_AABB_H_
