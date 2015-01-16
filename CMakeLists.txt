# Copyright (c) 2015, Bruno Turcksin.
# 
# This file is subject to the Modified BSD License and may not be distributed
# without copyright and license information. Please refer to the file
# license.txt for the text and further information on this license.


MESSAGE("+++++++++++++++++++++++++++")
MESSAGE("+++ Configuring Ouranos +++")
MESSAGE("+++++++++++++++++++++++++++")

# Set the name of the project and target.
SET(TARGET "ouranos")

FILE(GLOB_RECURSE TARGET_SRC "src/*.cc")

CMAKE_MINIMUM_REQUIRED(VERSION 2.8.8)

FIND_PACKAGE(deal.II 8.3 QUIET
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
  )
IF(NOT ${deal.II_FOUND})
  MESSAGE(FATAL_ERROR "\n*** Could not find a suitably recent version of deal.II. ***\n"
    "You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake "
    "or set an environment variable \"DEAL_II_DIR\" that contains a path to a "
    "sufficiently recent version of deal.II."
    )
ENDIF()

MESSAGE(STATUS "Found deal.II version ${DEAL_II_PACKAGE_VERSION} at '${deal.II_DIR}'")

SET(_DEALII_GOOD ON)

IF(NOT DEAL_II_WITH_P4EST)
    MESSAGE(SEND_ERROR
      "\n-- deal.II was built without support for p4est!\n"
      )
    SET(_DEALII_GOOD OFF)
ENDIF()

IF(NOT DEAL_II_WITH_TRILINOS)
    MESSAGE(SEND_ERROR
      "\n-- deal.II was built without support for Trilinos!\n"
      )
    SET(_DEALII_GOOD OFF)
ENDIF()

IF (NOT _DEALII_GOOD)
  MESSAGE(FATAL_ERROR
    "\nOuranos requires a deal.II installation built with support for Trilinos and p4est but one or both of these appears to be missing!\n"
    )
ENDIF()

DEAL_II_INITIALIZE_CACHED_VARIABLES()
PROJECT(${TARGET})

# Version Number
SET(Ouranos_VERSION_MAJOR 0)
SET(Ouranos_VERSION_MINOR 1)

# Create a configure file
CONFIGURE_FILE(
  "${CMAKE_SOURCE_DIR}/src/OuranosConfig.h.in"
  "${CMAKE_BINARY_DIR}/OuranosConfig.h"
  )

# Set up the testsuite
SET(OURANOS_TEST_SUITE OFF CACHE BOOL "Build test suite if set to 'on'.")
IF(OURANOS_TEST_SUITE)
  ADD_SUBDIRECTORY(tests)
ENDIF()


DEAL_II_INVOKE_AUTOPILOT()

MESSAGE("+++++++++++++++++++++++++++")
MESSAGE("The test suite for Ouranos is set to: " ${OURANOS_TEST_SUITE})
MESSAGE("+++++++++++++++++++++++++++")
