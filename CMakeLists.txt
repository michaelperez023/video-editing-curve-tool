cmake_minimum_required(VERSION 3.16)
project(VideoEditingCurveTool LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs videoio highgui)
message(STATUS "OpenCV ${OpenCV_VERSION} found in ${OpenCV_DIR}")

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-Wall -Wextra)
elseif(MSVC)
  add_compile_options(/W4)
endif()

# Tone curve, lookup table and curve window drawing
add_library(curve_core STATIC Curve.cpp Curve.h)
target_include_directories(curve_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(curve_core PUBLIC opencv_core opencv_imgproc)

# The curve tool (VideoProcessor.h and Histogram1D.h are from R. Laganiere's
# OpenCV 3 Computer Vision Application Programming Cookbook)
add_executable(curvetool main.cpp VideoProcessor.h Histogram1D.h)
target_link_libraries(curvetool PRIVATE curve_core opencv_imgcodecs opencv_videoio opencv_highgui)

option(BUILD_TESTING "Build the tests" ON)
if(BUILD_TESTING)
  enable_testing()

  add_executable(curve_tests tests/test_curve.cpp)
  target_link_libraries(curve_tests PRIVATE curve_core)
  foreach(name IN ITEMS identity control_points uneven_spacing monotone flat_ends smooth parse editing
                        apply drawing)
    add_test(NAME ${name} COMMAND curve_tests ${name})
  endforeach()

  add_executable(check_video tests/check_video.cpp)
  target_link_libraries(check_video PRIVATE curve_core opencv_imgcodecs opencv_videoio)

  # The program runs on copies of the sample video in the build directory, so
  # the one in the source tree is never at risk. barriers.avi has 399 frames.
  set(SAMPLE_VIDEO ${CMAKE_CURRENT_SOURCE_DIR}/barriers.avi)
  set(SAMPLE_FRAMES 399)
  set(TEST_DIR ${CMAKE_CURRENT_BINARY_DIR}/test_output)
  set(VIDEO ${TEST_DIR}/barriers.avi)
  set(CURVE 0:0,128:191,255:255)
  file(MAKE_DIRECTORY ${TEST_DIR} ${TEST_DIR}/default)

  add_test(NAME copy_sample COMMAND ${CMAKE_COMMAND} -E copy ${SAMPLE_VIDEO} ${VIDEO})
  add_test(NAME copy_sample_default COMMAND ${CMAKE_COMMAND} -E copy ${SAMPLE_VIDEO} ${TEST_DIR}/default/barriers.avi)
  set_tests_properties(copy_sample PROPERTIES FIXTURES_SETUP sample)
  set_tests_properties(copy_sample_default PROPERTIES FIXTURES_SETUP sample_default)

  # Headless run, then a check that every frame was written with the curve
  # applied (by decoding both videos and counting their frames) and that the
  # images match the first frame
  add_test(NAME cli_no_gui
           COMMAND curvetool --no-gui --curve ${CURVE} --out ${TEST_DIR}/curved.avi
                   --images ${TEST_DIR}/images ${VIDEO})
  set_tests_properties(cli_no_gui PROPERTIES FIXTURES_REQUIRED sample FIXTURES_SETUP curved)
  add_test(NAME cli_frames_written
           COMMAND check_video --frames ${SAMPLE_FRAMES} --images ${TEST_DIR}/images
                   ${VIDEO} ${TEST_DIR}/curved.avi ${CURVE})
  set_tests_properties(cli_frames_written PROPERTIES FIXTURES_REQUIRED curved)

  # Without --out the video is written next to the input as <name>_curves.avi
  add_test(NAME cli_default_output COMMAND curvetool --no-gui --curve 0:255,255:0 ${TEST_DIR}/default/barriers.avi)
  set_tests_properties(cli_default_output PROPERTIES FIXTURES_REQUIRED sample_default FIXTURES_SETUP inverted)
  add_test(NAME cli_default_output_frames
           COMMAND check_video --frames ${SAMPLE_FRAMES} ${TEST_DIR}/default/barriers.avi
                   ${TEST_DIR}/default/barriers_curves.avi 0:255,255:0)
  set_tests_properties(cli_default_output_frames PROPERTIES FIXTURES_REQUIRED inverted)

  # Error cases, recognised by their messages. The input may never be
  # written, however its path is spelled.
  add_test(NAME cli_same_file COMMAND curvetool --no-gui --out ${TEST_DIR}/./barriers.avi ${VIDEO})
  add_test(NAME cli_missing_video COMMAND curvetool --no-gui ${TEST_DIR}/missing.avi)
  add_test(NAME cli_bad_curve COMMAND curvetool --no-gui --curve 0:0,300:255 ${VIDEO})
  add_test(NAME cli_unknown_option COMMAND curvetool --no-gui --brightness 10 ${VIDEO})
  file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/not_a_directory "")
  add_test(NAME cli_bad_images_dir
           COMMAND curvetool --no-gui --out ${TEST_DIR}/unused.avi
                   --images ${CMAKE_CURRENT_BINARY_DIR}/not_a_directory ${VIDEO})
  set_tests_properties(cli_same_file PROPERTIES FIXTURES_REQUIRED sample
                       PASS_REGULAR_EXPRESSION "is the input video")
  set_tests_properties(cli_missing_video PROPERTIES PASS_REGULAR_EXPRESSION "Could not open the video")
  set_tests_properties(cli_bad_curve PROPERTIES
                       PASS_REGULAR_EXPRESSION "--curve: '300:255' is not a point x:y with levels from 0 to 255")
  set_tests_properties(cli_unknown_option PROPERTIES PASS_REGULAR_EXPRESSION "Unknown option: --brightness")
  # The directory must be checked before the video is processed
  set_tests_properties(cli_bad_images_dir PROPERTIES FIXTURES_REQUIRED sample
                       PASS_REGULAR_EXPRESSION "Could not create" FAIL_REGULAR_EXPRESSION "Wrote")

  # After all of the above, the copy of the input is unchanged
  add_test(NAME cli_input_unchanged COMMAND ${CMAKE_COMMAND} -E compare_files ${SAMPLE_VIDEO} ${VIDEO})
  set_tests_properties(cli_input_unchanged PROPERTIES FIXTURES_REQUIRED "sample;curved"
                       DEPENDS "cli_same_file;cli_bad_curve;cli_unknown_option;cli_bad_images_dir")
endif()
