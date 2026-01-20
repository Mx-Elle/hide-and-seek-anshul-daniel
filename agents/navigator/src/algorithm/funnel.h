#ifndef NAVIGATOR_SRC_ALGORITHM_FUNNEL_H_
#define NAVIGATOR_SRC_ALGORITHM_FUNNEL_H_

#include <vector>

#include "../core/navmesh.h"

struct FunnelDebugStep {
  Vec2 apex;
  Vec2 left_leg;
  Vec2 right_leg;
  Vec2 current_p_left;
  Vec2 current_p_right;
  std::vector<Vec2> current_path;
};

class Funnel {
public:
  static std::vector<Vec2> StringPull(Vec2 start, Vec2 end,
                                      const std::vector<uint16_t> &corridor,
                                      const NavMesh &mesh);
};

#endif
