#pragma once

// LRUCache<T> — port of `depthMap/cuda/host/LRUCache.hpp`.
//
// A fixed-capacity LRU cache that maps values of type T to a
// stable zero-based slot index. The same T inserted twice
// returns the same slot. Once full, the least-recently-used
// value is evicted to make room for a new one.
//
// T must be:
//   * default-constructible (used to fill the "old value" output
//     when nothing was evicted)
//   * copyable + equality-comparable
//   * constructible from `int` with `T(-1)` representing "empty slot"
//
// For our depthMap use the type parameters are `int` (camera id)
// and `av::depth_map::CameraPair` (camera id + downscale). Both
// satisfy the contract trivially.

#include <algorithm>
#include <list>
#include <stdexcept>
#include <vector>

namespace av::depth_map {

template <typename T>
class LRUCache {
public:
    LRUCache() : max_size_(0) {}

    explicit LRUCache(int size)
        : owner_(static_cast<std::size_t>(size), T(-1)),
          max_size_(size)
    {}

    void resize(int size) {
        max_size_ = size;
        owner_.resize(static_cast<std::size_t>(size), T(-1));
    }

    int get_index(const T& val) const {
        const auto it = std::find(owner_.begin(), owner_.end(), val);
        return (it != owner_.end())
            ? static_cast<int>(it - owner_.begin())
            : -1;
    }

    // Insert `val`.
    //   * If `val` is already cached: returns false; sets `position`
    //     to its slot; `old_val` is left as T(-1).
    //   * If `val` is new and there's empty room: returns true;
    //     sets `position` to the new slot; `old_val` is T(-1).
    //   * If `val` is new and the cache is full: returns true;
    //     sets `position` to the (evicted) slot; `old_val` is set
    //     to the evicted value.
    bool insert(const T& val, int& position, T& old_val) {
        if (max_size_ == 0) {
            throw std::runtime_error(
                "LRUCache::insert: max size is 0");
        }

        auto o_it = std::find(owner_.begin(), owner_.end(), val);
        if (o_it != owner_.end()) {
            old_val = T(-1);
            auto c_it = std::find(cache_.begin(), cache_.end(), val);
            cache_.erase(c_it);
            cache_.push_back(val);
            position = static_cast<int>(o_it - owner_.begin());
            return false;
        }

        const int cell = static_cast<int>(cache_.size());
        if (cell < max_size_) {
            old_val = T(-1);
            cache_.push_back(val);
            owner_[static_cast<std::size_t>(cell)] = val;
            position = cell;
            return true;
        }

        // Full → evict LRU.
        old_val = cache_.front();
        cache_.pop_front();
        cache_.push_back(val);
        o_it = std::find(owner_.begin(), owner_.end(), old_val);
        *o_it = val;
        position = static_cast<int>(o_it - owner_.begin());
        return true;
    }

    void clear() {
        cache_.clear();
        std::fill(owner_.begin(), owner_.end(), T(-1));
    }

    int  size()     const noexcept { return static_cast<int>(cache_.size()); }
    int  capacity() const noexcept { return max_size_; }

private:
    std::list<T>   cache_;     // LRU order (front = least recent)
    std::vector<T> owner_;     // slot → owning T (T(-1) = empty)
    int            max_size_;
};

// CameraPair — (camera id, downscale) key for the camera-params
// cache. Constructible from `int(-1)` for the sentinel.
struct CameraPair {
    int camera_id;
    int downscale;

    constexpr CameraPair() noexcept              : camera_id(0),  downscale(0)  {}
    constexpr explicit CameraPair(int v) noexcept: camera_id(v),  downscale(v)  {}
    constexpr CameraPair(int id, int ds) noexcept: camera_id(id), downscale(ds) {}

    CameraPair& operator=(int v) noexcept {
        camera_id = downscale = v;
        return *this;
    }

    constexpr bool operator==(const CameraPair& o) const noexcept {
        return camera_id == o.camera_id && downscale == o.downscale;
    }
    constexpr bool operator!=(const CameraPair& o) const noexcept {
        return !(*this == o);
    }
    constexpr bool operator<(const CameraPair& o) const noexcept {
        return (camera_id < o.camera_id) ||
               (camera_id == o.camera_id && downscale < o.downscale);
    }
};

}  // namespace av::depth_map
