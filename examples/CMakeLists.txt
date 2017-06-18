add_executable( dense_inference dense_inference.cpp)
target_link_libraries(dense_inference densecrf ${OpenCV_LIBS} ${DGM_LIBRARIES})
