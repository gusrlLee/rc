cmake_minimum_required(VERSION 3.30)

find_package(OpenGL REQUIRED)

add_library(glad 
    src/glad.c
)

target_include_directories( glad PUBLIC ${OpenGL_INCLUDE_DIRS})
target_include_directories( glad PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(glad PRIVATE ${OPENGL_LIBRARIES})