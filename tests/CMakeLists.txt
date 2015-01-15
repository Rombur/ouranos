# Copyright (c) 2015, Bruno Turcksin.
# 
# This file is subject to the Modified BSD License and may not be distributed
# without copyright and license information. Please refer to the file
# license.txt for the text and further information on this license.


FILE(GLOB SOURCES "../src/*.cc")
SET(TESTING_SOURCES ${SOURCES})
LIST(REMOVE_ITEM TESTING_SOURCES ${CMAKE_SOURCE_DIR}/tests/../src/ouranos.cc)
FILE(GLOB HEARDERS "../src/*.hh")

ADD_EXECUTABLE(rtquadrature_test RTQuadrature_test.cc ${TESTING_SOURCES} ${HEARDERS})
DEAL_II_SETUP_TARGET(rtquadrature_test)

ADD_EXECUTABLE(parameters_test Parameters_test.cc)
DEAL_II_SETUP_TARGET(parameters_test)

ADD_EXECUTABLE(geometry_test Geometry_test.cc)
DEAL_II_SETUP_TARGET(geometry_test)

ADD_EXECUTABLE(fecell_test FECell_test.cc)
DEAL_II_SETUP_TARGET(fecell_test)

ADD_EXECUTABLE(rtmaterialproperties_test RTMaterialProperties_test.cc)
DEAL_II_SETUP_TARGET(rtmaterialproperties)

ADD_EXECUTABLE(radiativetransfer_test RadiativeTransfer_test.cc)
DEAL_II_SETUP_TARGET(radiativetransfer_test)

ADD_CUSTOM_TARGET(tests
  COMMAND ./rtquadrature_test
  COMMAND ./parameters_test
  COMMAND mpirun -np 1 ./geomety_test
  COMMAND ./fecell_test
  COMMAND ./rtmaterialproperties
  COMMAND mpirun -np ./radiativetransfer_test
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/tests
  )
