/* Copyright (c) 2013, Bruno Turcksin
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file
 * license.txt for the text and further information on this license.
 */

#define CATCH_CONFIG_RUNNER
#include "catch.hpp"

#include <cmath>
#include <string>
#include <vector>
#include "Epetra_Map.h"
#include "Epetra_MpiComm.h"
#include "Epetra_MultiVector.h"
#include "deal.II/base/utilities.h"
#include "deal.II/base/mpi.h"
#include "deal.II/fe/fe_dgq.h"
#include "deal.II/lac/trilinos_vector.h"
#include "../src/Geometry.hh"
#include "../src/LS.hh"
#include "../src/Parameters.hh"
#include "../src/RadiativeTransfer.hh"


TEST_CASE("Radiative Transfer","Check One-Group Radiative Transfer for 2D on 4 processors")
{
  std::string parameters_filename("./tests/rt_parameters.inp");
  Parameters parameters(parameters_filename);
  std::string geometry_filename(parameters.get_geometry_filename());
  std::string xs_filename(parameters.get_xs_filename());
  FE_DGQ<2> fe(parameters.get_fe_order());
  Geometry<2> geometry(geometry_filename,fe);
  RTMaterialProperties material_properties(xs_filename,geometry.get_n_materials(),
      parameters.get_n_groups());

  DoFHandler<2>* dof_handler(geometry.get_dof_handler());
  IndexSet index_set(dof_handler->locally_owned_dofs());
  LS quad(parameters.get_sn_order(),material_properties.get_L_max(),
      parameters.get_galerkin());
  quad.build_quadrature(parameters.get_weight_sum());
  
  std::vector<TrilinosWrappers::MPI::Vector> group_flux(1,
      TrilinosWrappers::MPI::Vector (index_set));
  Epetra_MpiComm comm(MPI_COMM_WORLD);
  Epetra_Map map(index_set.make_trilinos_map());
  Epetra_MultiVector flux_moments(map,quad.get_n_mom());
  Epetra_MultiVector group_flux_moments(map,quad.get_n_mom());

  // Creat the RadiativeTransfer object
  RadiativeTransfer<2,4> radiative_transfer(&fe,geometry.get_triangulation(),
      dof_handler,&parameters,&quad,&material_properties,&comm,&map);

  // Create the FECells and compute the sweep ordering
  radiative_transfer.setup();

  // Set the current group
  radiative_transfer.set_group(0);

  // Compute the right-hand side
  Epetra_MultiVector rhs(flux_moments);
  std::list<double*> buffers;
  std::list<MPI_Request*> requests;
  radiative_transfer.initialize_scheduler();
  while (radiative_transfer.get_n_tasks_to_execute()!=0)
  {
    radiative_transfer.sweep(*(radiative_transfer.get_next_task()),buffers,
        requests,rhs,&group_flux);
    radiative_transfer.free_buffers(buffers,requests);
  }

  while (buffers.size()!=0)
    radiative_transfer.free_buffers(buffers,requests);
  radiative_transfer.clear_scheduler();



  // Reference solution
  std::vector<double> solution(36,0.);
  solution[0] = 0.974449442413;
  solution[1] = 0.97444944203;
  solution[2] = 1.66326834888;
  solution[3] = 1.66326833258;
  solution[4] = 0.974449442389;
  solution[5] = 0.974449442389;
  solution[6] = 1.66326834662;
  solution[7] = 1.66326834662;
  solution[8] = 0.97444944209;
  solution[9] = 0.974449442413;
  solution[10] = 1.66326833258;
  solution[11] = 1.66326834888;
  solution[12] = 1.66833181666;
  solution[13] = 1.6683318057;
  solution[14] = 1.6683318057;
  solution[15] = 1.66833181666;
  solution[16] = 1.66833180693;
  solution[17] = 1.66833180693;
  solution[18] = 1.66833180693;
  solution[19] = 1.66833180693;
  solution[20] = 1.6683318057;
  solution[21] = 1.66833181666;
  solution[22] = 1.66833181666;
  solution[23] = 1.6683318057;
  solution[24] = 1.66326833258;
  solution[25] = 1.66326834888;
  solution[26] = 0.97444944209;
  solution[27] = 0.974449442413;
  solution[28] = 1.66326834662;
  solution[29] = 1.66326834662;
  solution[30] = 0.974449442389;
  solution[31] = 0.974449442389;
  solution[32] = 1.66326834888;
  solution[33] = 1.66326834888;
  solution[34] = 0.974449442413;
  solution[35] = 0.97444944209;


//  // Need reflective BC
//  for (unsigned int i=0; i<36; ++i)
//    REQUIRE(std::fabs(flux_moments[0][i]-solution[i])<1e-3);
}

int main (int argc, char** argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc,argv);

  int result = Catch::Main(argc,argv);

  return result;
}
