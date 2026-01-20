#ifndef NAVIGATOR_SRC_UTIL_QUATERNARY_HEAP_H_
#define NAVIGATOR_SRC_UTIL_QUATERNARY_HEAP_H_

#include <algorithm>
#include <utility>
#include <vector>

template <typename T, typename Compare> class QuaternaryHeap {
public:
  void Clear() { data_.clear(); }
  bool Empty() const { return data_.empty(); }

  void Push(T val) {
    data_.push_back(val);
    SiftUp(data_.size() - 1);
  }

  T Pop() {
    T top = data_[0];
    if (data_.size() > 1) {
      data_[0] = data_.back();
      data_.pop_back();
      SiftDown(0);
    } else {
      data_.pop_back();
    }
    return top;
  }

private:
  std::vector<T> data_;
  Compare cmp_;

  void SiftUp(size_t idx) {
    while (idx > 0) {
      size_t parent = (idx - 1) >> 2;
      if (cmp_(data_[parent], data_[idx])) {
        std::swap(data_[parent], data_[idx]);
        idx = parent;
      } else {
        break;
      }
    }
  }

  void SiftDown(size_t idx) {
    size_t size = data_.size();
    while (true) {
      size_t child = (idx << 2) + 1;
      if (child >= size)
        break;

      size_t min = FindMinChild(child, size);

      if (cmp_(data_[idx], data_[min])) {
        std::swap(data_[idx], data_[min]);
        idx = min;
      } else {
        break;
      }
    }
  }

  inline size_t FindMinChild(size_t first, size_t size) const {
    size_t min = first;
    size_t c1 = first + 1;
    if (c1 < size && cmp_(data_[min], data_[c1]))
      min = c1;
    size_t c2 = first + 2;
    if (c2 < size && cmp_(data_[min], data_[c2]))
      min = c2;
    size_t c3 = first + 3;
    if (c3 < size && cmp_(data_[min], data_[c3]))
      min = c3;
    return min;
  }
};

#endif // NAVIGATOR_SRC_UTIL_QUATERNARY_HEAP_H_
