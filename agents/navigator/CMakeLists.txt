cmake_minimum_required(VERSION 3.15)
project(navigator LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT MSVC)
  add_compile_options(-O3 -ffast-math -march=native -flto -fno-plt -fvisibility=hidden -Wall)
  add_link_options(-flto)
  if(APPLE)
    add_link_options(-Wl,-dead_strip)
  endif()
else()
  add_compile_options(/O2 /fp:fast /arch:AVX2 /GL)
  add_link_options(/LTCG)
endif()

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -c "import nanobind; print(nanobind.cmake_dir())"
  OUTPUT_VARIABLE nanobind_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

find_package(nanobind CONFIG REQUIRED)

set(NAVIGATOR_SOURCES
    src/bindings.cc
    src/navigator.cc
    src/core/navmesh.cc
    src/algorithm/bvh.cc
    src/algorithm/apsp.cc
    src/algorithm/funnel.cc
)

nanobind_add_module(_navigator ${NAVIGATOR_SOURCES})
target_include_directories(_navigator PRIVATE src)
install(TARGETS _navigator DESTINATION navigator)
install(FILES __init__.py mesh_converter.py DESTINATION navigator)
