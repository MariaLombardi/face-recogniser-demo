# Copyright (C) 2018 Fondazione Istituto Italiano di Tecnologia (IIT)
# All Rights Reserved.


cmake_minimum_required(VERSION 3.1)
project(face-recogniser-demo)

set(functions_path "${CMAKE_CURRENT_SOURCE_DIR}/functions")
set(models_path "${CMAKE_CURRENT_SOURCE_DIR}/models")
set(dataset_path "${CMAKE_CURRENT_SOURCE_DIR}/..")

if(NOT EXISTS ${functions_path})
	message(FATAL_ERROR "the python functions are not in the expected folder.")
endif()

if(NOT EXISTS ${models_path})
	message(FATAL_ERROR "the python models are not in the expected folder.")
endif()

if(NOT EXISTS ${dataset_path})
	message(FATAL_ERROR "the LWF dataset is not in the expected folder.")
endif()

configure_file(face-recogniser.py.in face-recogniser.py @ONLY)
configure_file(face-recogniser-realsense.py.in face-recogniser-realsense.py @ONLY)

set(python_bin ${CMAKE_CURRENT_SOURCE_DIR}/../build/src/face-recogniser.py
				${CMAKE_CURRENT_SOURCE_DIR}/../build/src/face-recogniser-realsense.py)

add_custom_target(copy_python_in_build ALL)
add_custom_command(TARGET copy_python_in_build POST_BUILD
	               COMMAND ${CMAKE_COMMAND} -E copy ${python_bin} ${CMAKE_BINARY_DIR}/${CMAKE_CFG_INTDIR}
	               COMMENT "Copying ${python_bin} to ${CMAKE_BINARY_DIR}/${CMAKE_CFG_INTDIR}/")

install(PROGRAMS ${python_bin} DESTINATION bin)

