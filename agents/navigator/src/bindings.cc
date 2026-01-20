#include "navigator.h"
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;

NB_MODULE(navigator, m) {
  nb::class_<Vec2>(m, "Vec2")
      .def(nb::init<float, float>())
      .def_rw("x", &Vec2::x)
      .def_rw("y", &Vec2::y);

  nb::class_<AABB>(m, "AABB")
      .def_rw("min_x", &AABB::min_x)
      .def_rw("min_y", &AABB::min_y)
      .def_rw("max_x", &AABB::max_x)
      .def_rw("max_y", &AABB::max_y);

  nb::class_<Navigator>(m, "Navigator")
      .def(nb::init<>())
      .def("load_mesh",
           [](Navigator &self,
              nb::ndarray<float, nb::ndim<2>, nb::c_contig> verts,
              nb::ndarray<int32_t, nb::ndim<2>, nb::c_contig> tris,
              nb::ndarray<int32_t, nb::ndim<2>, nb::c_contig> nbrs) {
             if (verts.shape(1) != 2)
               throw std::runtime_error("Verts Nx2");
             if (tris.shape(1) != 3)
               throw std::runtime_error("Tris Nx3");
             if (nbrs.shape(1) != 3)
               throw std::runtime_error("Nbrs Nx3");

             self.LoadMesh(verts.data(), verts.shape(0), tris.data(),
                           tris.shape(0), nbrs.data());
           })
      .def("find_path",
           [](Navigator &self, float sx, float sy, float gx, float gy) {
             std::vector<Vec2> path = self.FindPath(sx, sy, gx, gy);
             std::vector<std::pair<float, float>> res;
             res.reserve(path.size());
             for (const auto &p : path)
               res.push_back({p.x, p.y});
             return res;
           })
      .def("find_path_array",
           [](Navigator &self, float sx, float sy, float gx, float gy) {
             std::vector<Vec2> path = self.FindPath(sx, sy, gx, gy);
             float *data = new float[path.size() * 2];
             for (size_t i = 0; i < path.size(); ++i) {
               data[i * 2] = path[i].x;
               data[i * 2 + 1] = path[i].y;
             }
             size_t shape[2] = {path.size(), 2};
             return nb::ndarray<nb::numpy, float, nb::ndim<2>, nb::c_contig>(
                 data, 2, shape, nb::capsule(data, [](void *p) noexcept {
                   delete[] (float *)p;
                 }));
           })
      .def("find_triangle", &Navigator::FindTriangle)
      .def("num_triangles", &Navigator::NumTriangles)
      .def("is_ready", &Navigator::IsReady)
      .def("get_triangle_vertices", [](Navigator &self, int id) {
        auto verts = self.GetTriangleVertices(static_cast<uint16_t>(id));
        return std::vector<std::pair<float, float>>{{verts[0].x, verts[0].y},
                                                    {verts[1].x, verts[1].y},
                                                    {verts[2].x, verts[2].y}};
      });
}
