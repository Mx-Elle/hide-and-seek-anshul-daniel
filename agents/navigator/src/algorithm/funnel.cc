#include "funnel.h"

#include <vector>

namespace {

int FindSharedEdgeIndex(const Triangle &current_triangle,
                        uint16_t next_triangle_id) {
  for (int j = 0; j < 3; ++j) {
    if (current_triangle.neighbors[j] == next_triangle_id) {
      return j;
    }
  }
  return -1;
}

void DeterminePortalVertices(const Vec2 &flow, const Vec2 &current_centroid,
                             const Vec2 &v1, const Vec2 &v2,
                             std::vector<Vec2> *portals_left,
                             std::vector<Vec2> *portals_right) {
  Vec2 to_v1 = {v1.x - current_centroid.x, v1.y - current_centroid.y};
  float cross = flow.x * to_v1.y - flow.y * to_v1.x;

  if (cross < 0.0f) {
    portals_left->push_back(v1);
    portals_right->push_back(v2);
  } else {
    portals_left->push_back(v2);
    portals_right->push_back(v1);
  }
}

void ProcessCorridorSegment(const Triangle &current_triangle,
                            const Triangle &next_triangle,
                            uint16_t next_triangle_id,
                            std::vector<Vec2> *portals_left,
                            std::vector<Vec2> *portals_right) {
  int edge_idx = FindSharedEdgeIndex(current_triangle, next_triangle_id);
  if (edge_idx == -1) {
    return;
  }

  Vec2 v1 = current_triangle.v[edge_idx];
  Vec2 v2 = current_triangle.v[(edge_idx + 1) % 3];

  Vec2 current_centroid = current_triangle.Centroid();
  Vec2 next_centroid = next_triangle.Centroid();
  Vec2 flow = {next_centroid.x - current_centroid.x,
               next_centroid.y - current_centroid.y};

  DeterminePortalVertices(flow, current_centroid, v1, v2, portals_left,
                          portals_right);
}

} // namespace

void BuildPortals(const std::vector<uint16_t> &corridor, const NavMesh &mesh,
                  Vec2 start, Vec2 end, std::vector<Vec2> &portals_left,
                  std::vector<Vec2> &portals_right) {
  portals_left.push_back(start);
  portals_right.push_back(start);

  for (size_t i = 0; i < corridor.size() - 1; ++i) {
    const Triangle &current_triangle = mesh.GetTriangle(corridor[i]);
    const Triangle &next_triangle = mesh.GetTriangle(corridor[i + 1]);

    ProcessCorridorSegment(current_triangle, next_triangle, corridor[i + 1],
                           &portals_left, &portals_right);
  }

  portals_left.push_back(end);
  portals_right.push_back(end);
}

namespace {

inline bool ArePointsEqual(const Vec2 &a, const Vec2 &b) {
  return a.x == b.x && a.y == b.y;
}

struct FunnelState {
  Vec2 apex;
  Vec2 left_leg;
  Vec2 right_leg;
  int left_idx;
  int right_idx;
};

bool HandleRightNarrowing(const Vec2 &new_right, size_t current_idx,
                          FunnelState *state, std::vector<Vec2> *path,
                          size_t *next_idx) {
  float area = TriArea2(state->apex, state->right_leg, new_right);
  if (area > 0.0f) {
    return false;
  }

  bool apex_is_right = ArePointsEqual(state->apex, state->right_leg);
  float cross_area = TriArea2(state->apex, state->left_leg, new_right);

  if (apex_is_right || cross_area >= 0.0f) {
    state->right_leg = new_right;
    state->right_idx = current_idx;
    return false;
  }

  path->push_back(state->left_leg);
  state->apex = state->left_leg;
  state->right_leg = state->apex;
  state->left_leg = state->apex;
  *next_idx = state->left_idx;
  state->right_idx = state->left_idx;
  return true;
}

bool HandleLeftNarrowing(const Vec2 &new_left, size_t current_idx,
                         FunnelState *state, std::vector<Vec2> *path,
                         size_t *next_idx) {
  float area = TriArea2(state->apex, state->left_leg, new_left);
  if (area < 0.0f) {
    return false;
  }

  bool apex_is_left = ArePointsEqual(state->apex, state->left_leg);
  float cross_area = TriArea2(state->apex, state->right_leg, new_left);

  if (apex_is_left || cross_area <= 0.0f) {
    state->left_leg = new_left;
    state->left_idx = current_idx;
    return false;
  }

  path->push_back(state->right_leg);
  state->apex = state->right_leg;
  state->left_leg = state->apex;
  state->right_leg = state->apex;
  *next_idx = state->right_idx;
  state->left_idx = state->right_idx;
  return true;
}

void ProcessPortal(const Vec2 &new_left, const Vec2 &new_right,
                   size_t current_idx, FunnelState *state,
                   std::vector<Vec2> *path, size_t *next_idx) {
  *next_idx = current_idx + 1;

  if (HandleRightNarrowing(new_right, current_idx, state, path, next_idx)) {
    return;
  }

  HandleLeftNarrowing(new_left, current_idx, state, path, next_idx);
}

void EnsurePathEndsAtDestination(const Vec2 &end, std::vector<Vec2> *path) {
  if (path->empty() || !ArePointsEqual(path->back(), end)) {
    path->push_back(end);
  }
}

} // namespace

std::vector<Vec2> Funnel::StringPull(Vec2 start, Vec2 end,
                                     const std::vector<uint16_t> &corridor,
                                     const NavMesh &mesh) {
  if (corridor.empty()) {
    return {};
  }

  if (corridor.size() == 1) {
    return {start, end};
  }

  std::vector<Vec2> portals_left;
  std::vector<Vec2> portals_right;
  BuildPortals(corridor, mesh, start, end, portals_left, portals_right);

  std::vector<Vec2> path;
  path.push_back(start);

  FunnelState state = {
      .apex = start,
      .left_leg = portals_left[0],
      .right_leg = portals_right[0],
      .left_idx = 0,
      .right_idx = 0,
  };

  for (size_t i = 1; i < portals_left.size();) {
    Vec2 new_left = portals_left[i];
    Vec2 new_right = portals_right[i];

    size_t next_idx;
    ProcessPortal(new_left, new_right, i, &state, &path, &next_idx);
    i = next_idx;
  }

  EnsurePathEndsAtDestination(end, &path);
  return path;
}
