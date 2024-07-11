add_compile_options(-fsycl)
add_link_options(-fsycl)

add_executable(main main.cpp)

target_compile_definitions(main INTERFACE DR_FORMAT)
target_link_libraries(main DR::mpi fmt::fmt)
