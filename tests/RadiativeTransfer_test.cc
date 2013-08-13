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
#include "deal.II/lac/solver_control.h"
#include "deal.II/lac/trilinos_precondition.h"
#include "deal.II/lac/trilinos_solver.h"
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
      TrilinosWrappers::MPI::Vector (index_set,MPI_COMM_WORLD));
  TrilinosWrappers::MPI::Vector flux_moments(index_set,MPI_COMM_WORLD);
  Epetra_MpiComm comm(MPI_COMM_WORLD);
  Epetra_Map map(index_set.make_trilinos_map());

  // Creat the RadiativeTransfer object
  RadiativeTransfer<2,4> radiative_transfer(&fe,geometry.get_triangulation(),
      dof_handler,&parameters,&quad,&material_properties,&comm,&map);

  // Create the FECells and compute the sweep ordering
  radiative_transfer.setup();

  // Set the current group
  radiative_transfer.set_group(0);

  // Compute the right-hand side
  TrilinosWrappers::MPI::Vector rhs(flux_moments);
  std::list<double*> buffers;
  std::list<MPI_Request*> requests;
  radiative_transfer.initialize_scheduler();
  Epetra_MultiVector psi(map,quad.get_n_dir());
  while (radiative_transfer.get_n_tasks_to_execute()!=0)
  {
    radiative_transfer.sweep(*(radiative_transfer.get_next_task()),buffers,
        requests,rhs.trilinos_vector(),psi,&group_flux);
    radiative_transfer.free_buffers(buffers,requests);
  }
  while (buffers.size()!=0)
    radiative_transfer.free_buffers(buffers,requests);
  radiative_transfer.clear_scheduler();

  SolverControl solver_control(parameters.get_max_inner_it(),
      parameters.get_inner_tolerance());
  TrilinosWrappers::SolverBicgstab solver(solver_control);
  TrilinosWrappers::PreconditionIdentity preconditioner;
  solver.solve(radiative_transfer,flux_moments,rhs,
      preconditioner);

  std::cout<<flux_moments.trilinos_vector()<<std::endl;

  // Reference solution
  std::vector<double> solution(64,0.);
  solution[0]=0.431292;
  solution[1]=0.660557;
  solution[2]=1.04838;
  solution[3]=0.660557;
  solution[4]=0.663296;
  solution[5]=0.698176;
  solution[6]=1.11381;
  solution[7]=1.05284;
  solution[8]=0.698176;
  solution[9]=0.663296;
  solution[10]=1.05284;
  solution[11]=1.11381;
  solution[12]=0.660557;
  solution[13]=0.431292;
  solution[14]=0.660557;
  solution[15]=1.04838;
  solution[16]=0.663296;
  solution[17]=1.05284;
  solution[18]=1.11381;
  solution[19]=0.698176;
  solution[20]=1.05753;
  solution[21]=1.11944;
  solution[22]=1.18834;
  solution[23]=1.11944;
  solution[24]=1.11944;
  solution[25]=1.05753;
  solution[26]=1.11944;
  solution[27]=1.18834;
  solution[28]=1.05284;
  solution[29]=0.663296;
  solution[30]=0.698176;
  solution[31]=1.11381;
  solution[32]=0.698176;
  solution[33]=1.11381;
  solution[34]=1.05284;
  solution[35]=0.663296;
  solution[36]=1.11944;
  solution[37]=1.18834;
  solution[38]=1.11944;
  solution[39]=1.05753;
  solution[40]=1.18834;
  solution[41]=1.11944;
  solution[42]=1.05753;
  solution[43]=1.11944;
  solution[44]=1.11381;
  solution[45]=0.698176;
  solution[46]=0.663296;
  solution[47]=1.05284;
  solution[48]=0.660557;
  solution[49]=1.04838;
  solution[50]=0.660557;
  solution[51]=0.431292;
  solution[52]=1.05284;
  solution[53]=1.11381;
  solution[54]=0.698176;
  solution[55]=0.663296;
  solution[56]=1.11381;
  solution[57]=1.05284;
  solution[58]=0.663296;
  solution[59]=0.698176;
  solution[60]=1.04838;
  solution[61]=0.660557;
  solution[62]=0.431292;
  solution[63]=0.660557;


//  for (unsigned int i=0; i<64; ++i)
//    REQUIRE(std::fabs(flux_moments[0][i]-solution[i])<1e-3);
}

int main (int argc, char** argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc,argv);

  int result = Catch::Main(argc,argv);

  return result;
}
