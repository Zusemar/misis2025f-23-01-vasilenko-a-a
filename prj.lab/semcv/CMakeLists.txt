add_library(semcv semcv.hpp semcv.cpp)
target_include_directories(semcv PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_link_libraries(semcv PUBLIC ${OpenCV_LIBS})