#include "bvh.h"
#include <algorithm>
#include <cfloat>
#include <numeric>

#if defined(__aarch64__)
#include <arm_neon.h>
#elif defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace {

#if defined(__aarch64__)
inline uint32x4_t CheckIntersection(float32x4_t px, float32x4_t py,
                                    const QBvhNode &node)
    __attribute__((always_inline));

inline uint32x4_t CheckIntersection(float32x4_t px, float32x4_t py,
                                    const QBvhNode &node) {
  uint32x4_t m = vcgeq_f32(px, vld1q_f32(node.min_x));
  m = vandq_u32(m, vcleq_f32(px, vld1q_f32(node.max_x)));
  m = vandq_u32(m, vcgeq_f32(py, vld1q_f32(node.min_y)));
  m = vandq_u32(m, vcleq_f32(py, vld1q_f32(node.max_y)));
  return m;
}
#elif defined(__x86_64__) || defined(_M_X64)
inline int CheckIntersectionMask(__m128 px, __m128 py, const QBvhNode &node)
    __attribute__((always_inline));

inline int CheckIntersectionMask(__m128 px, __m128 py, const QBvhNode &node) {
  __m128 min_x = _mm_loadu_ps(node.min_x);
  __m128 max_x = _mm_loadu_ps(node.max_x);
  __m128 min_y = _mm_loadu_ps(node.min_y);
  __m128 max_y = _mm_loadu_ps(node.max_y);

  __m128 m = _mm_cmpge_ps(px, min_x);
  m = _mm_and_ps(m, _mm_cmple_ps(px, max_x));
  m = _mm_and_ps(m, _mm_cmpge_ps(py, min_y));
  m = _mm_and_ps(m, _mm_cmple_ps(py, max_y));

  return _mm_movemask_ps(m);
}
#endif

} // namespace

void TriangleBvh::Build(const NavMesh &mesh) {
  mesh_ = &mesh;
  nodes_.clear();
  size_t n = mesh.NumTriangles();
  if (n == 0)
    return;

  std::vector<uint16_t> ids(n);
  std::iota(ids.begin(), ids.end(), 0);

  std::vector<BvhNode> bin_nodes;
  bin_nodes.reserve(n * 2);
  BuildRecursive(ids, 0, n, bin_nodes);

  if (!bin_nodes.empty()) {
    nodes_.reserve(bin_nodes.size());
    CollapseToQuad(0, nodes_, bin_nodes);
  }
}

uint32_t TriangleBvh::BuildRecursive(std::vector<uint16_t> &ids, int start,
                                     int end, std::vector<BvhNode> &nodes) {
  uint32_t idx = nodes.size();
  nodes.emplace_back();

  const auto &t_start = mesh_->GetTriangle(ids[start]);
  AABB bounds = t_start.Bounds();
  for (int i = start + 1; i < end; ++i) {
    bounds = AABB::Merge(bounds, mesh_->GetTriangle(ids[i]).Bounds());
  }
  nodes[idx].bounds = bounds;

  int count = end - start;
  if (count <= 1) {
    nodes[idx].left_or_tri = ids[start];
    nodes[idx].right = 0;
    return idx;
  }

  float dx = bounds.max_x - bounds.min_x;
  float dy = bounds.max_y - bounds.min_y;
  auto mid_it = ids.begin() + start + count / 2;

  auto cmp_x = [&](uint16_t a, uint16_t b) {
    return mesh_->GetTriangle(a).Centroid().x <
           mesh_->GetTriangle(b).Centroid().x;
  };
  auto cmp_y = [&](uint16_t a, uint16_t b) {
    return mesh_->GetTriangle(a).Centroid().y <
           mesh_->GetTriangle(b).Centroid().y;
  };

  if (dx > dy) {
    std::nth_element(ids.begin() + start, mid_it, ids.begin() + end, cmp_x);
  } else {
    std::nth_element(ids.begin() + start, mid_it, ids.begin() + end, cmp_y);
  }

  int mid = start + count / 2;
  uint32_t left = BuildRecursive(ids, start, mid, nodes);
  uint32_t right = BuildRecursive(ids, mid, end, nodes);
  nodes[idx].left_or_tri = left;
  nodes[idx].right = right;
  return idx;
}

void TriangleBvh::CollapseToQuad(uint32_t bin_idx,
                                 std::vector<QBvhNode> &q_nodes,
                                 const std::vector<BvhNode> &bin_nodes) {
  uint32_t my_idx = q_nodes.size();
  q_nodes.emplace_back();
  const auto &root = bin_nodes[bin_idx];

  for (int i = 0; i < 4; ++i) {
    q_nodes[my_idx].min_x[i] = FLT_MAX;
    q_nodes[my_idx].min_y[i] = FLT_MAX;
    q_nodes[my_idx].max_x[i] = -FLT_MAX;
    q_nodes[my_idx].max_y[i] = -FLT_MAX;
  }

  if (root.IsLeaf()) {
    auto &q = q_nodes[my_idx];
    q.count = 1;
    q.min_x[0] = root.bounds.min_x;
    q.min_y[0] = root.bounds.min_y;
    q.max_x[0] = root.bounds.max_x;
    q.max_y[0] = root.bounds.max_y;
    q.children[0] = ~root.left_or_tri;
    return;
  }

  uint32_t children[4];
  int count = 0;

  uint32_t l = root.left_or_tri;
  uint32_t r = root.right;

  if (!bin_nodes[l].IsLeaf()) {
    children[0] = bin_nodes[l].left_or_tri;
    children[1] = bin_nodes[l].right;
    count = 2;
  } else {
    children[0] = l;
    count = 1;
  }

  if (!bin_nodes[r].IsLeaf()) {
    children[count++] = bin_nodes[r].left_or_tri;
    children[count++] = bin_nodes[r].right;
  } else {
    children[count++] = r;
  }

  for (int i = 0; i < count; ++i) {
    const auto &c = bin_nodes[children[i]];
    q_nodes[my_idx].min_x[i] = c.bounds.min_x;
    q_nodes[my_idx].min_y[i] = c.bounds.min_y;
    q_nodes[my_idx].max_x[i] = c.bounds.max_x;
    q_nodes[my_idx].max_y[i] = c.bounds.max_y;
  }
  q_nodes[my_idx].count = count;

  for (int i = 0; i < count; ++i) {
    if (bin_nodes[children[i]].IsLeaf()) {
      q_nodes[my_idx].children[i] = ~bin_nodes[children[i]].left_or_tri;
    } else {
      q_nodes[my_idx].children[i] = q_nodes.size();
      CollapseToQuad(children[i], q_nodes, bin_nodes);
    }
  }
}

uint16_t TriangleBvh::FindTriangle(Vec2 p) const {
  if (nodes_.empty())
    return kNoNeighbor;

  int32_t stack[64];
  int sp = 0;
  stack[sp++] = 0;

#if defined(__aarch64__)
  float32x4_t px = vdupq_n_f32(p.x);
  float32x4_t py = vdupq_n_f32(p.y);
#elif defined(__x86_64__) || defined(_M_X64)
  __m128 px = _mm_set1_ps(p.x);
  __m128 py = _mm_set1_ps(p.y);
#endif

  while (sp > 0) {
    int32_t idx = stack[--sp];
    const auto &node = nodes_[idx];

#if defined(__aarch64__)
    uint32x4_t m = CheckIntersection(px, py, node);

    if (vmaxvq_u32(m) != 0) {
      if (vgetq_lane_u32(m, 0)) {
        int32_t c = node.children[0];
        if (c < 0) {
          if (mesh_->GetTriangle(~c).Contains(p))
            return ~c;
        } else
          stack[sp++] = c;
      }
      if (vgetq_lane_u32(m, 1)) {
        int32_t c = node.children[1];
        if (c < 0) {
          if (mesh_->GetTriangle(~c).Contains(p))
            return ~c;
        } else
          stack[sp++] = c;
      }
      if (vgetq_lane_u32(m, 2)) {
        int32_t c = node.children[2];
        if (c < 0) {
          if (mesh_->GetTriangle(~c).Contains(p))
            return ~c;
        } else
          stack[sp++] = c;
      }
      if (vgetq_lane_u32(m, 3)) {
        int32_t c = node.children[3];
        if (c < 0) {
          if (mesh_->GetTriangle(~c).Contains(p))
            return ~c;
        } else
          stack[sp++] = c;
      }
    }
#elif defined(__x86_64__) || defined(_M_X64)
    int mask = CheckIntersectionMask(px, py, node);
    if (mask != 0) {
      if (mask & 1) {
        int32_t c = node.children[0];
        if (c < 0) {
          if (mesh_->GetTriangle(~c).Contains(p))
            return ~c;
        } else
          stack[sp++] = c;
      }
      if (mask & 2) {
        int32_t c = node.children[1];
        if (c < 0) {
          if (mesh_->GetTriangle(~c).Contains(p))
            return ~c;
        } else
          stack[sp++] = c;
      }
      if (mask & 4) {
        int32_t c = node.children[2];
        if (c < 0) {
          if (mesh_->GetTriangle(~c).Contains(p))
            return ~c;
        } else
          stack[sp++] = c;
      }
      if (mask & 8) {
        int32_t c = node.children[3];
        if (c < 0) {
          if (mesh_->GetTriangle(~c).Contains(p))
            return ~c;
        } else
          stack[sp++] = c;
      }
    }
#else
    // Scalar fallback
    for (int i = 0; i < node.count; ++i) {
      if (p.x >= node.min_x[i] && p.x <= node.max_x[i] &&
          p.y >= node.min_y[i] && p.y <= node.max_y[i]) {
        int32_t c = node.children[i];
        if (c < 0) {
          if (mesh_->GetTriangle(~c).Contains(p))
            return ~c;
        } else {
          stack[sp++] = c;
        }
      }
    }
#endif
  }
  return kNoNeighbor;
}
