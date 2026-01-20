#ifndef NAVIGATOR_SRC_CORE_VEC2_H_
#define NAVIGATOR_SRC_CORE_VEC2_H_

#include <cmath>

struct alignas(8) Vec2 {
  float x, y;

  constexpr Vec2() : x(0), y(0) {}
  constexpr Vec2(float x_val, float y_val) : x(x_val), y(y_val) {}

  Vec2 operator+(Vec2 o) const { return {x + o.x, y + o.y}; }
  Vec2 operator-(Vec2 o) const { return {x - o.x, y - o.y}; }
  Vec2 operator*(float s) const { return {x * s, y * s}; }
  bool operator==(Vec2 o) const { return x == o.x && y == o.y; }

  float Dot(Vec2 o) const { return x * o.x + y * o.y; }
  float Cross(Vec2 o) const { return x * o.y - y * o.x; }
  float LengthSq() const { return x * x + y * y; }
  float Length() const { return std::sqrt(LengthSq()); }

  Vec2 Normalized() const {
    float len = Length();
    return len > 1e-6f ? Vec2{x / len, y / len} : Vec2{0, 0};
  }
};

inline float TriArea2(Vec2 a, Vec2 b, Vec2 c) {
  return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
}

#endif // NAVIGATOR_SRC_CORE_VEC2_H_
