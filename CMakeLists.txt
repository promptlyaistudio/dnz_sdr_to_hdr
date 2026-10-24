cmake_minimum_required(VERSION 3.20)

# macOS 27's CommandLineTools SDK ships a .tbd this linker rejects ("tapi error:
# malformed file"), so prefer Xcode's SDK. Must happen before project() enables C.
if(APPLE AND NOT CMAKE_OSX_SYSROOT
   AND EXISTS "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk")
    set(CMAKE_OSX_SYSROOT "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk")
endif()

project(dnz-sdr-to-hdr VERSION 1.0.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(APPLE)
    # Flame on Apple Silicon: native arm64 only keeps the build fast and the binary small.
    set(CMAKE_OSX_ARCHITECTURES "arm64" CACHE STRING "Build architectures")
    set(CMAKE_OSX_DEPLOYMENT_TARGET "11.0" CACHE STRING "Minimum deployment target")
endif()

set(OFX_SDK_DIR "" CACHE PATH "Path to the OpenFX SDK")
if(NOT OFX_SDK_DIR)
    foreach(cand "${CMAKE_SOURCE_DIR}/../openfx" "$ENV{HOME}/projets/openfx")
        if(EXISTS "${cand}/include/ofxCore.h")
            set(OFX_SDK_DIR "${cand}")
            break()
        endif()
    endforeach()
endif()
if(NOT OFX_SDK_DIR)
    message(FATAL_ERROR "OpenFX SDK not found; pass -DOFX_SDK_DIR=/path/to/openfx")
endif()
message(STATUS "OpenFX SDK: ${OFX_SDK_DIR}")

# The .ofx bundle itself is assembled (and ad-hoc signed) by build.sh: CMake's own
# bundle support leaves CFBundleExecutable empty for MODULE targets and skips signing.
add_library(dnz-sdr-to-hdr MODULE
    src/main.cpp
    src/effect.cpp
    src/render.cpp
    src/engine_client.cpp
)
set_target_properties(dnz-sdr-to-hdr PROPERTIES
    PREFIX ""
    SUFFIX ".ofx"
    OUTPUT_NAME "DNZSDRtoHDR"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)
target_include_directories(dnz-sdr-to-hdr PRIVATE
    "${OFX_SDK_DIR}/include"
    "${CMAKE_SOURCE_DIR}/src"
)
target_compile_options(dnz-sdr-to-hdr PRIVATE -O2 -Wall -Wno-unused-parameter)
