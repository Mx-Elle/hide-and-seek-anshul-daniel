#ifndef NAVIGATOR_SRC_CORE_VEC2_SIMD_H_
#define NAVIGATOR_SRC_CORE_VEC2_SIMD_H_

#include "vec2.h"

#include <cstddef>
#include <cstring>

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

#ifdef __SSE__
#include <xmmintrin.h>
#endif

#ifdef __AVX__
#include <immintrin.h>
#endif

namespace vec2_simd {

#ifdef __ARM_NEON

inline void BatchLengthSqNEON(const Vec2 *vectors, float *results,
                              size_t count) {
  size_t i = 0;

  for (; i + 3 < count; i += 4) {
    float32x4_t x = {vectors[i].x, vectors[i + 1].x, vectors[i + 2].x,
                     vectors[i + 3].x};
    float32x4_t y = {vectors[i].y, vectors[i + 1].y, vectors[i + 2].y,
                     vectors[i + 3].y};

    float32x4_t xx = vmulq_f32(x, x);
    float32x4_t yy = vmulq_f32(y, y);
    float32x4_t length_sq = vaddq_f32(xx, yy);

    vst1q_f32(&results[i], length_sq);
  }

  for (; i < count; ++i) {
    results[i] = vectors[i].LengthSq();
  }
}

inline void BatchDotNEON(const Vec2 *a, const Vec2 *b, float *results,
                         size_t count) {
  size_t i = 0;

  for (; i + 3 < count; i += 4) {
    float32x4_t ax = {a[i].x, a[i + 1].x, a[i + 2].x, a[i + 3].x};
    float32x4_t ay = {a[i].y, a[i + 1].y, a[i + 2].y, a[i + 3].y};
    float32x4_t bx = {b[i].x, b[i + 1].x, b[i + 2].x, b[i + 3].x};
    float32x4_t by = {b[i].y, b[i + 1].y, b[i + 2].y, b[i + 3].y};

    float32x4_t prod_x = vmulq_f32(ax, bx);
    float32x4_t prod_y = vmulq_f32(ay, by);
    float32x4_t dot = vaddq_f32(prod_x, prod_y);

    vst1q_f32(&results[i], dot);
  }

  for (; i < count; ++i) {
    results[i] = a[i].Dot(b[i]);
  }
}

inline void BatchCrossNEON(const Vec2 *a, const Vec2 *b, float *results,
                           size_t count) {
  size_t i = 0;

  for (; i + 3 < count; i += 4) {
    float32x4_t ax = {a[i].x, a[i + 1].x, a[i + 2].x, a[i + 3].x};
    float32x4_t ay = {a[i].y, a[i + 1].y, a[i + 2].y, a[i + 3].y};
    float32x4_t bx = {b[i].x, b[i + 1].x, b[i + 2].x, b[i + 3].x};
    float32x4_t by = {b[i].y, b[i + 1].y, b[i + 2].y, b[i + 3].y};

    float32x4_t prod1 = vmulq_f32(ax, by);
    float32x4_t prod2 = vmulq_f32(ay, bx);
    float32x4_t cross = vsubq_f32(prod1, prod2);

    vst1q_f32(&results[i], cross);
  }

  for (; i < count; ++i) {
    results[i] = a[i].Cross(b[i]);
  }
}

inline void BatchTriArea2NEON(const Vec2 *a, const Vec2 *b, const Vec2 *c,
                              float *results, size_t count) {
  size_t i = 0;

  for (; i + 3 < count; i += 4) {
    float32x4_t ax = {a[i].x, a[i + 1].x, a[i + 2].x, a[i + 3].x};
    float32x4_t ay = {a[i].y, a[i + 1].y, a[i + 2].y, a[i + 3].y};
    float32x4_t bx = {b[i].x, b[i + 1].x, b[i + 2].x, b[i + 3].x};
    float32x4_t by = {b[i].y, b[i + 1].y, b[i + 2].y, b[i + 3].y};
    float32x4_t cx = {c[i].x, c[i + 1].x, c[i + 2].x, c[i + 3].x};
    float32x4_t cy = {c[i].y, c[i + 1].y, c[i + 2].y, c[i + 3].y};

    float32x4_t dx = vsubq_f32(bx, ax);
    float32x4_t dy = vsubq_f32(by, ay);
    float32x4_t ex = vsubq_f32(cx, ax);
    float32x4_t ey = vsubq_f32(cy, ay);

    float32x4_t prod1 = vmulq_f32(dx, ey);
    float32x4_t prod2 = vmulq_f32(dy, ex);
    float32x4_t area = vsubq_f32(prod1, prod2);

    vst1q_f32(&results[i], area);
  }

  for (; i < count; ++i) {
    results[i] = TriArea2(a[i], b[i], c[i]);
  }
}

inline void BatchAddNEON(const Vec2 *a, const Vec2 *b, Vec2 *results,
                         size_t count) {
  size_t i = 0;

  for (; i + 3 < count; i += 4) {
    float32x4_t ax = {a[i].x, a[i + 1].x, a[i + 2].x, a[i + 3].x};
    float32x4_t ay = {a[i].y, a[i + 1].y, a[i + 2].y, a[i + 3].y};
    float32x4_t bx = {b[i].x, b[i + 1].x, b[i + 2].x, b[i + 3].x};
    float32x4_t by = {b[i].y, b[i + 1].y, b[i + 2].y, b[i + 3].y};

    float32x4_t rx = vaddq_f32(ax, bx);
    float32x4_t ry = vaddq_f32(ay, by);

    // Store results
    for (size_t j = 0; j < 4; ++j) {
      results[i + j].x = rx[j];
      results[i + j].y = ry[j];
    }
  }

  for (; i < count; ++i) {
    results[i] = a[i] + b[i];
  }
}

inline void BatchSubNEON(const Vec2 *a, const Vec2 *b, Vec2 *results,
                         size_t count) {
  size_t i = 0;

  for (; i + 3 < count; i += 4) {
    float32x4_t ax = {a[i].x, a[i + 1].x, a[i + 2].x, a[i + 3].x};
    float32x4_t ay = {a[i].y, a[i + 1].y, a[i + 2].y, a[i + 3].y};
    float32x4_t bx = {b[i].x, b[i + 1].x, b[i + 2].x, b[i + 3].x};
    float32x4_t by = {b[i].y, b[i + 1].y, b[i + 2].y, b[i + 3].y};

    float32x4_t rx = vsubq_f32(ax, bx);
    float32x4_t ry = vsubq_f32(ay, by);

    for (size_t j = 0; j < 4; ++j) {
      results[i + j].x = rx[j];
      results[i + j].y = ry[j];
    }
  }

  for (; i < count; ++i) {
    results[i] = a[i] - b[i];
  }
}

#endif // __ARM_NEON

#ifdef __SSE__

inline void BatchLengthSqSSE(const Vec2 *vectors, float *results,
                             size_t count) {
  size_t i = 0;

  for (; i + 3 < count; i += 4) {
    __m128 x = _mm_set_ps(vectors[i + 3].x, vectors[i + 2].x, vectors[i + 1].x,
                          vectors[i].x);
    __m128 y = _mm_set_ps(vectors[i + 3].y, vectors[i + 2].y, vectors[i + 1].y,
                          vectors[i].y);

    __m128 xx = _mm_mul_ps(x, x);
    __m128 yy = _mm_mul_ps(y, y);
    __m128 length_sq = _mm_add_ps(xx, yy);

    _mm_storeu_ps(&results[i], length_sq);
  }

  for (; i < count; ++i) {
    results[i] = vectors[i].LengthSq();
  }
}

inline void BatchDotSSE(const Vec2 *a, const Vec2 *b, float *results,
                        size_t count) {
  size_t i = 0;

  for (; i + 3 < count; i += 4) {
    __m128 ax = _mm_set_ps(a[i + 3].x, a[i + 2].x, a[i + 1].x, a[i].x);
    __m128 ay = _mm_set_ps(a[i + 3].y, a[i + 2].y, a[i + 1].y, a[i].y);
    __m128 bx = _mm_set_ps(b[i + 3].x, b[i + 2].x, b[i + 1].x, b[i].x);
    __m128 by = _mm_set_ps(b[i + 3].y, b[i + 2].y, b[i + 1].y, b[i].y);

    __m128 prod_x = _mm_mul_ps(ax, bx);
    __m128 prod_y = _mm_mul_ps(ay, by);
    __m128 dot = _mm_add_ps(prod_x, prod_y);

    _mm_storeu_ps(&results[i], dot);
  }

  for (; i < count; ++i) {
    results[i] = a[i].Dot(b[i]);
  }
}

inline void BatchCrossSSE(const Vec2 *a, const Vec2 *b, float *results,
                          size_t count) {
  size_t i = 0;

  for (; i + 3 < count; i += 4) {
    __m128 ax = _mm_set_ps(a[i + 3].x, a[i + 2].x, a[i + 1].x, a[i].x);
    __m128 ay = _mm_set_ps(a[i + 3].y, a[i + 2].y, a[i + 1].y, a[i].y);
    __m128 bx = _mm_set_ps(b[i + 3].x, b[i + 2].x, b[i + 1].x, b[i].x);
    __m128 by = _mm_set_ps(b[i + 3].y, b[i + 2].y, b[i + 1].y, b[i].y);

    __m128 prod1 = _mm_mul_ps(ax, by);
    __m128 prod2 = _mm_mul_ps(ay, bx);
    __m128 cross = _mm_sub_ps(prod1, prod2);

    _mm_storeu_ps(&results[i], cross);
  }

  for (; i < count; ++i) {
    results[i] = a[i].Cross(b[i]);
  }
}

inline void BatchTriArea2SSE(const Vec2 *a, const Vec2 *b, const Vec2 *c,
                             float *results, size_t count) {
  size_t i = 0;

  for (; i + 3 < count; i += 4) {
    __m128 ax = _mm_set_ps(a[i + 3].x, a[i + 2].x, a[i + 1].x, a[i].x);
    __m128 ay = _mm_set_ps(a[i + 3].y, a[i + 2].y, a[i + 1].y, a[i].y);
    __m128 bx = _mm_set_ps(b[i + 3].x, b[i + 2].x, b[i + 1].x, b[i].x);
    __m128 by = _mm_set_ps(b[i + 3].y, b[i + 2].y, b[i + 1].y, b[i].y);
    __m128 cx = _mm_set_ps(c[i + 3].x, c[i + 2].x, c[i + 1].x, c[i].x);
    __m128 cy = _mm_set_ps(c[i + 3].y, c[i + 2].y, c[i + 1].y, c[i].y);

    __m128 dx = _mm_sub_ps(bx, ax);
    __m128 dy = _mm_sub_ps(by, ay);
    __m128 ex = _mm_sub_ps(cx, ax);
    __m128 ey = _mm_sub_ps(cy, ay);

    __m128 prod1 = _mm_mul_ps(dx, ey);
    __m128 prod2 = _mm_mul_ps(dy, ex);
    __m128 area = _mm_sub_ps(prod1, prod2);

    _mm_storeu_ps(&results[i], area);
  }

  for (; i < count; ++i) {
    results[i] = TriArea2(a[i], b[i], c[i]);
  }
}

inline void BatchAddSSE(const Vec2 *a, const Vec2 *b, Vec2 *results,
                        size_t count) {
  size_t i = 0;

  for (; i + 3 < count; i += 4) {
    __m128 ax = _mm_set_ps(a[i + 3].x, a[i + 2].x, a[i + 1].x, a[i].x);
    __m128 ay = _mm_set_ps(a[i + 3].y, a[i + 2].y, a[i + 1].y, a[i].y);
    __m128 bx = _mm_set_ps(b[i + 3].x, b[i + 2].x, b[i + 1].x, b[i].x);
    __m128 by = _mm_set_ps(b[i + 3].y, b[i + 2].y, b[i + 1].y, b[i].y);

    __m128 rx = _mm_add_ps(ax, bx);
    __m128 ry = _mm_add_ps(ay, by);

    alignas(16) float temp_x[4];
    alignas(16) float temp_y[4];
    _mm_store_ps(temp_x, rx);
    _mm_store_ps(temp_y, ry);

    for (size_t j = 0; j < 4; ++j) {
      results[i + j].x = temp_x[j];
      results[i + j].y = temp_y[j];
    }
  }

  for (; i < count; ++i) {
    results[i] = a[i] + b[i];
  }
}

inline void BatchSubSSE(const Vec2 *a, const Vec2 *b, Vec2 *results,
                        size_t count) {
  size_t i = 0;

  for (; i + 3 < count; i += 4) {
    __m128 ax = _mm_set_ps(a[i + 3].x, a[i + 2].x, a[i + 1].x, a[i].x);
    __m128 ay = _mm_set_ps(a[i + 3].y, a[i + 2].y, a[i + 1].y, a[i].y);
    __m128 bx = _mm_set_ps(b[i + 3].x, b[i + 2].x, b[i + 1].x, b[i].x);
    __m128 by = _mm_set_ps(b[i + 3].y, b[i + 2].y, b[i + 1].y, b[i].y);

    __m128 rx = _mm_sub_ps(ax, bx);
    __m128 ry = _mm_sub_ps(ay, by);

    alignas(16) float temp_x[4];
    alignas(16) float temp_y[4];
    _mm_store_ps(temp_x, rx);
    _mm_store_ps(temp_y, ry);

    for (size_t j = 0; j < 4; ++j) {
      results[i + j].x = temp_x[j];
      results[i + j].y = temp_y[j];
    }
  }

  for (; i < count; ++i) {
    results[i] = a[i] - b[i];
  }
}

#endif // __SSE__

inline void BatchLengthSq(const Vec2 *vectors, float *results, size_t count) {
#ifdef __ARM_NEON
  BatchLengthSqNEON(vectors, results, count);
#elif defined(__SSE__)
  BatchLengthSqSSE(vectors, results, count);
#else
  for (size_t i = 0; i < count; ++i) {
    results[i] = vectors[i].LengthSq();
  }
#endif
}

inline void BatchDot(const Vec2 *a, const Vec2 *b, float *results,
                     size_t count) {
#ifdef __ARM_NEON
  BatchDotNEON(a, b, results, count);
#elif defined(__SSE__)
  BatchDotSSE(a, b, results, count);
#else
  for (size_t i = 0; i < count; ++i) {
    results[i] = a[i].Dot(b[i]);
  }
#endif
}

inline void BatchCross(const Vec2 *a, const Vec2 *b, float *results,
                       size_t count) {
#ifdef __ARM_NEON
  BatchCrossNEON(a, b, results, count);
#elif defined(__SSE__)
  BatchCrossSSE(a, b, results, count);
#else
  for (size_t i = 0; i < count; ++i) {
    results[i] = a[i].Cross(b[i]);
  }
#endif
}

inline void BatchTriArea2(const Vec2 *a, const Vec2 *b, const Vec2 *c,
                          float *results, size_t count) {
#ifdef __ARM_NEON
  BatchTriArea2NEON(a, b, c, results, count);
#elif defined(__SSE__)
  BatchTriArea2SSE(a, b, c, results, count);
#else
  for (size_t i = 0; i < count; ++i) {
    results[i] = TriArea2(a[i], b[i], c[i]);
  }
#endif
}

inline void BatchAdd(const Vec2 *a, const Vec2 *b, Vec2 *results,
                     size_t count) {
#ifdef __ARM_NEON
  BatchAddNEON(a, b, results, count);
#elif defined(__SSE__)
  BatchAddSSE(a, b, results, count);
#else
  for (size_t i = 0; i < count; ++i) {
    results[i] = a[i] + b[i];
  }
#endif
}

inline void BatchSub(const Vec2 *a, const Vec2 *b, Vec2 *results,
                     size_t count) {
#ifdef __ARM_NEON
  BatchSubNEON(a, b, results, count);
#elif defined(__SSE__)
  BatchSubSSE(a, b, results, count);
#else
  for (size_t i = 0; i < count; ++i) {
    results[i] = a[i] - b[i];
  }
#endif
}

} // namespace vec2_simd

#endif // NAVIGATOR_SRC_CORE_VEC2_SIMD_H_
