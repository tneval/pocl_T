
if(HAVE_LIBJPEG_TURBO)
    list(APPEND POCL_DEVICES_LINK_LIST ${LIBJPEG_TARGET})
endif()

if(HAVE_ONNXRT)
    list(APPEND POCL_DEVICES_LINK_LIST onnxruntime::onnxruntime)
    list(APPEND POCL_DEVICES_SOURCES
            ${CMAKE_CURRENT_SOURCE_DIR}/pocl_dbk_khr_onnxrt_cpu.c ${CMAKE_CURRENT_SOURCE_DIR}/pocl_dbk_khr_onnxrt_cpu.h
    )
endif ()

list(APPEND POCL_DEVICES_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/pocl_dbk_khr_jpeg_cpu.c ${CMAKE_CURRENT_SOURCE_DIR}/pocl_dbk_khr_jpeg_cpu.h
        ${CMAKE_CURRENT_SOURCE_DIR}/pocl_dbk_khr_img_cpu.c ${CMAKE_CURRENT_SOURCE_DIR}/pocl_dbk_khr_img_cpu.h
)

if(HAVE_OPENCV)
    list(APPEND POCL_DEVICES_SOURCES
            ${CMAKE_CURRENT_SOURCE_DIR}/pocl_dbk_khr_dnn_utils.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/pocl_dbk_khr_dnn_utils.hh)

    list(APPEND POCL_DEVICES_LINK_LIST opencv_dnn)

endif ()

set(POCL_DEVICES_LINK_LIST ${POCL_DEVICES_LINK_LIST} PARENT_SCOPE)
set(POCL_DEVICES_SOURCES ${POCL_DEVICES_SOURCES} PARENT_SCOPE)


