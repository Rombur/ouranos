/* Copyright (c) 2013 - 2015 Bruno Turcksin
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file
 * license.txt for the text and further information on this license.
 */

#define CATCH_CONFIG_RUNNER
#include "catch.hpp"

#include <iostream>
#include <cmath>
#include <memory>
#include <string>
#include <vector>
#include "Epetra_Map.h"
#include "Epetra_MpiComm.h"
#include "Epetra_MultiVector.h"
#include "deal.II/base/conditional_ostream.h"
#include "deal.II/base/utilities.h"
#include "deal.II/base/mpi.h"
#include "deal.II/fe/fe_dgq.h"
#include "deal.II/lac/solver_control.h"
#include "deal.II/lac/trilinos_precondition.h"
#include "deal.II/lac/trilinos_solver.h"
#include "deal.II/lac/trilinos_vector.h"
#include "../src/Geometry.hh"
#include "../src/GLC.hh"
#include "../src/Parameters.hh"
#include "../src/RadiativeTransfer.hh"
#include "../src/RandomScheduler.hh"
#include "../src/CAPPFBScheduler.hh"


TEST_CASE("Check One-Group Radiative Transfer for 2D on 4 processors using the random scheduler",
    "[one cell patch][random scheduler]")
{
  ConditionalOStream pcout(std::cout,Utilities::MPI::this_mpi_process(
        MPI_COMM_WORLD)==0);
  std::string parameters_filename("rt_parameters_single_cell_patch.inp");
  Parameters parameters(parameters_filename);
  std::string geometry_filename(parameters.get_geometry_filename());
  std::string xs_filename(parameters.get_xs_filename());
  FE_DGQ<2> fe(parameters.get_fe_order());
  Geometry<2> geometry(pcout,geometry_filename,fe);
  RTMaterialProperties material_properties(xs_filename,geometry.get_n_materials(),
      parameters.get_n_groups());

  DoFHandler<2>* dof_handler(geometry.get_dof_handler());
  IndexSet index_set(dof_handler->locally_owned_dofs());
  GLC quad(parameters.get_sn_order(),material_properties.get_L_max(),
      parameters.get_galerkin());
  quad.build_quadrature(parameters.get_weight_sum(),2);
  
  std::vector<TrilinosWrappers::MPI::Vector> group_flux(1,
      TrilinosWrappers::MPI::Vector (index_set,MPI_COMM_WORLD));
  TrilinosWrappers::MPI::Vector flux_moments(index_set,MPI_COMM_WORLD);
  Epetra_MpiComm comm(MPI_COMM_WORLD);
  Epetra_Map map(index_set.make_trilinos_map());

  // Create the scheduler.
  std::shared_ptr<Scheduler<2,4>> scheduler(new RandomScheduler<2,4> (&quad,&comm,pcout));

  // Create the RadiativeTransfer object
  RadiativeTransfer<2,4> radiative_transfer(1,dof_handler->n_dofs(),&fe,
      geometry.get_triangulation(),dof_handler,&parameters,&quad,&material_properties,
      &comm,&map,scheduler);

  // Create the FECells and compute the sweep ordering
  radiative_transfer.setup();

  // Set the current group
  radiative_transfer.set_group(0);

  // Compute the right-hand side
  TrilinosWrappers::MPI::Vector rhs(flux_moments);
  std::list<double*> buffers;
  std::list<MPI_Request*> requests;
  scheduler->start();
  while (scheduler->get_n_tasks_to_execute()!=0)
  {
    radiative_transfer.sweep(*(scheduler->get_next_task()),buffers,
        requests,rhs.trilinos_vector(),&group_flux);
    scheduler->free_buffers(buffers,requests);
  }

  SolverControl solver_control(parameters.get_max_inner_it(),
      parameters.get_inner_tolerance());
  TrilinosWrappers::SolverBicgstab solver(solver_control);
  TrilinosWrappers::PreconditionIdentity preconditioner;
  solver.solve(radiative_transfer,flux_moments,rhs,
      preconditioner);

  // Reference solution
  std::vector<double> solution(64,0.);
  solution[0]=0.407787;
  solution[1]= 0.62897;
  solution[2]= 0.62897;
  solution[3]= 1.02715;
  solution[4]=0.634493;
  solution[5]=0.680499;
  solution[6]= 1.03841;
  solution[7]= 1.11156;
  solution[8]=0.634493;
  solution[9]= 1.03841;
  solution[10]=0.680499;
  solution[11]= 1.11156;
  solution[12]= 1.05037;
  solution[13]= 1.12519;
  solution[14]= 1.12519;
  solution[15]= 1.20627;
  solution[16]=0.680499;
  solution[17]=0.634493;
  solution[18]= 1.11156;
  solution[19]= 1.03841;
  solution[20]= 0.62897;
  solution[21]=0.407787;
  solution[22]= 1.02715;
  solution[23]= 0.62897;
  solution[24]= 1.12519;
  solution[25]= 1.05037;
  solution[26]= 1.20627;
  solution[27]= 1.12519;
  solution[28]= 1.03841;
  solution[29]=0.634493;
  solution[30]= 1.11156;
  solution[31]=0.680499;
  solution[32]=0.680499;
  solution[33]= 1.11156;
  solution[34]=0.634493;
  solution[35]= 1.03841;
  solution[36]= 1.12519;
  solution[37]= 1.20627;
  solution[38]= 1.05037;
  solution[39]= 1.12519;
  solution[40]= 0.62897;
  solution[41]= 1.02715;
  solution[42]=0.407787;
  solution[43]= 0.62897;
  solution[44]= 1.03841;
  solution[45]= 1.11156;
  solution[46]=0.634493;
  solution[47]=0.680499;
  solution[48]= 1.20627;
  solution[49]= 1.12519;
  solution[50]= 1.12519;
  solution[51]= 1.05037;
  solution[52]= 1.11156;
  solution[53]=0.680499;
  solution[54]= 1.03841;
  solution[55]=0.634493;
  solution[56]= 1.11156;
  solution[57]= 1.03841;
  solution[58]=0.680499;
  solution[59]=0.634493;
  solution[60]= 1.02715;
  solution[61]= 0.62897;
  solution[62]= 0.62897;
  solution[63]=0.407787;

  for (unsigned int i=0; i<64; ++i)
    if (index_set.is_element(i)==true)
      REQUIRE(std::fabs(flux_moments[i]-solution[i])<1e-3);
}

TEST_CASE("Check One-Group Radiative Transfer for 2D on 4 processors using patches and the random scheduler",
    "[multiple cells patch][random scheduler]")
{
  ConditionalOStream pcout(std::cout,Utilities::MPI::this_mpi_process(
        MPI_COMM_WORLD)==0);
  std::string parameters_filename("rt_parameters_multiple_cells_patch.inp");
  Parameters parameters(parameters_filename);
  std::string geometry_filename(parameters.get_geometry_filename());
  std::string xs_filename(parameters.get_xs_filename());
  FE_DGQ<2> fe(parameters.get_fe_order());
  Geometry<2> geometry(pcout,geometry_filename,fe);
  RTMaterialProperties material_properties(xs_filename,geometry.get_n_materials(),
      parameters.get_n_groups());

  DoFHandler<2>* dof_handler(geometry.get_dof_handler());
  IndexSet index_set(dof_handler->locally_owned_dofs());
  GLC quad(parameters.get_sn_order(),material_properties.get_L_max(),
      parameters.get_galerkin());
  quad.build_quadrature(parameters.get_weight_sum(),2);
  
  std::vector<TrilinosWrappers::MPI::Vector> group_flux(1,
      TrilinosWrappers::MPI::Vector (index_set,MPI_COMM_WORLD));
  TrilinosWrappers::MPI::Vector flux_moments(index_set,MPI_COMM_WORLD);
  Epetra_MpiComm comm(MPI_COMM_WORLD);
  Epetra_Map map(index_set.make_trilinos_map());

  // Create the scheduler.
  std::shared_ptr<Scheduler<2,4>> scheduler(new RandomScheduler<2,4> (&quad,&comm,pcout));

  // Create the RadiativeTransfer object
  RadiativeTransfer<2,4> radiative_transfer(1,dof_handler->n_dofs(),&fe,
      geometry.get_triangulation(),dof_handler,&parameters,&quad,&material_properties,
      &comm,&map,scheduler);

  // Create the FECells and compute the sweep ordering
  radiative_transfer.setup();

  // Set the current group
  radiative_transfer.set_group(0);

  // Compute the right-hand side
  TrilinosWrappers::MPI::Vector rhs(flux_moments);
  std::list<double*> buffers;
  std::list<MPI_Request*> requests;
  scheduler->start();
  while (scheduler->get_n_tasks_to_execute()!=0)
  {
    radiative_transfer.sweep(*(scheduler->get_next_task()),buffers,
        requests,rhs.trilinos_vector(),&group_flux);
    scheduler->free_buffers(buffers,requests);
  }

  SolverControl solver_control(parameters.get_max_inner_it(),
      parameters.get_inner_tolerance());
  TrilinosWrappers::SolverBicgstab solver(solver_control);
  TrilinosWrappers::PreconditionIdentity preconditioner;
  solver.solve(radiative_transfer,flux_moments,rhs,
      preconditioner);

  // Reference solution
  std::vector<double> solution(64,0.);
  solution[0]=0.407787;
  solution[1]= 0.62897;
  solution[2]= 0.62897;
  solution[3]= 1.02715;
  solution[4]=0.634493;
  solution[5]=0.680499;
  solution[6]= 1.03841;
  solution[7]= 1.11156;
  solution[8]=0.634493;
  solution[9]= 1.03841;
  solution[10]=0.680499;
  solution[11]= 1.11156;
  solution[12]= 1.05037;
  solution[13]= 1.12519;
  solution[14]= 1.12519;
  solution[15]= 1.20627;
  solution[16]=0.680499;
  solution[17]=0.634493;
  solution[18]= 1.11156;
  solution[19]= 1.03841;
  solution[20]= 0.62897;
  solution[21]=0.407787;
  solution[22]= 1.02715;
  solution[23]= 0.62897;
  solution[24]= 1.12519;
  solution[25]= 1.05037;
  solution[26]= 1.20627;
  solution[27]= 1.12519;
  solution[28]= 1.03841;
  solution[29]=0.634493;
  solution[30]= 1.11156;
  solution[31]=0.680499;
  solution[32]=0.680499;
  solution[33]= 1.11156;
  solution[34]=0.634493;
  solution[35]= 1.03841;
  solution[36]= 1.12519;
  solution[37]= 1.20627;
  solution[38]= 1.05037;
  solution[39]= 1.12519;
  solution[40]= 0.62897;
  solution[41]= 1.02715;
  solution[42]=0.407787;
  solution[43]= 0.62897;
  solution[44]= 1.03841;
  solution[45]= 1.11156;
  solution[46]=0.634493;
  solution[47]=0.680499;
  solution[48]= 1.20627;
  solution[49]= 1.12519;
  solution[50]= 1.12519;
  solution[51]= 1.05037;
  solution[52]= 1.11156;
  solution[53]=0.680499;
  solution[54]= 1.03841;
  solution[55]=0.634493;
  solution[56]= 1.11156;
  solution[57]= 1.03841;
  solution[58]=0.680499;
  solution[59]=0.634493;
  solution[60]= 1.02715;
  solution[61]= 0.62897;
  solution[62]= 0.62897;
  solution[63]=0.407787;

  for (unsigned int i=0; i<64; ++i)
    if (index_set.is_element(i)==true)
      REQUIRE(std::fabs(flux_moments[i]-solution[i])<1e-3);
}

TEST_CASE("Check One-Group Radiative Transfer for 2D on 4 processors using the CAP-PFB scheduler",
    "[one cell patch][CAP-PFB scheduler]")
{
  ConditionalOStream pcout(std::cout,Utilities::MPI::this_mpi_process(
        MPI_COMM_WORLD)==0);
  std::string parameters_filename("rt_parameters_single_cell_patch.inp");
  Parameters parameters(parameters_filename);
  std::string geometry_filename(parameters.get_geometry_filename());
  std::string xs_filename(parameters.get_xs_filename());
  FE_DGQ<2> fe(parameters.get_fe_order());
  Geometry<2> geometry(pcout,geometry_filename,fe);
  RTMaterialProperties material_properties(xs_filename,geometry.get_n_materials(),
      parameters.get_n_groups());

  DoFHandler<2>* dof_handler(geometry.get_dof_handler());
  IndexSet index_set(dof_handler->locally_owned_dofs());
  GLC quad(parameters.get_sn_order(),material_properties.get_L_max(),
      parameters.get_galerkin());
  quad.build_quadrature(parameters.get_weight_sum(),2);
  
  std::vector<TrilinosWrappers::MPI::Vector> group_flux(1,
      TrilinosWrappers::MPI::Vector (index_set,MPI_COMM_WORLD));
  TrilinosWrappers::MPI::Vector flux_moments(index_set,MPI_COMM_WORLD);
  Epetra_MpiComm comm(MPI_COMM_WORLD);
  Epetra_Map map(index_set.make_trilinos_map());

  // Create the scheduler.
  unsigned int max_iter(4);
  std::shared_ptr<Scheduler<2,4>> scheduler(new CAPPFBScheduler<2,4> (&quad,&comm,pcout,max_iter));

  // Create the RadiativeTransfer object
  RadiativeTransfer<2,4> radiative_transfer(1,dof_handler->n_dofs(),&fe,
      geometry.get_triangulation(),dof_handler,&parameters,&quad,&material_properties,
      &comm,&map,scheduler);

  // Create the FECells and compute the sweep ordering
  radiative_transfer.setup();

  // Set the current group
  radiative_transfer.set_group(0);

  // Compute the right-hand side
  TrilinosWrappers::MPI::Vector rhs(flux_moments);
  std::list<double*> buffers;
  std::list<MPI_Request*> requests;
  scheduler->start();
  while (scheduler->get_n_tasks_to_execute()!=0)
  {
    radiative_transfer.sweep(*(scheduler->get_next_task()),buffers,
        requests,rhs.trilinos_vector(),&group_flux);
    scheduler->free_buffers(buffers,requests);
  }

  SolverControl solver_control(parameters.get_max_inner_it(),
      parameters.get_inner_tolerance());
  TrilinosWrappers::SolverBicgstab solver(solver_control);
  TrilinosWrappers::PreconditionIdentity preconditioner;
  solver.solve(radiative_transfer,flux_moments,rhs,
      preconditioner);

  // Reference solution
  std::vector<double> solution(64,0.);
  solution[0]=0.407787;
  solution[1]= 0.62897;
  solution[2]= 0.62897;
  solution[3]= 1.02715;
  solution[4]=0.634493;
  solution[5]=0.680499;
  solution[6]= 1.03841;
  solution[7]= 1.11156;
  solution[8]=0.634493;
  solution[9]= 1.03841;
  solution[10]=0.680499;
  solution[11]= 1.11156;
  solution[12]= 1.05037;
  solution[13]= 1.12519;
  solution[14]= 1.12519;
  solution[15]= 1.20627;
  solution[16]=0.680499;
  solution[17]=0.634493;
  solution[18]= 1.11156;
  solution[19]= 1.03841;
  solution[20]= 0.62897;
  solution[21]=0.407787;
  solution[22]= 1.02715;
  solution[23]= 0.62897;
  solution[24]= 1.12519;
  solution[25]= 1.05037;
  solution[26]= 1.20627;
  solution[27]= 1.12519;
  solution[28]= 1.03841;
  solution[29]=0.634493;
  solution[30]= 1.11156;
  solution[31]=0.680499;
  solution[32]=0.680499;
  solution[33]= 1.11156;
  solution[34]=0.634493;
  solution[35]= 1.03841;
  solution[36]= 1.12519;
  solution[37]= 1.20627;
  solution[38]= 1.05037;
  solution[39]= 1.12519;
  solution[40]= 0.62897;
  solution[41]= 1.02715;
  solution[42]=0.407787;
  solution[43]= 0.62897;
  solution[44]= 1.03841;
  solution[45]= 1.11156;
  solution[46]=0.634493;
  solution[47]=0.680499;
  solution[48]= 1.20627;
  solution[49]= 1.12519;
  solution[50]= 1.12519;
  solution[51]= 1.05037;
  solution[52]= 1.11156;
  solution[53]=0.680499;
  solution[54]= 1.03841;
  solution[55]=0.634493;
  solution[56]= 1.11156;
  solution[57]= 1.03841;
  solution[58]=0.680499;
  solution[59]=0.634493;
  solution[60]= 1.02715;
  solution[61]= 0.62897;
  solution[62]= 0.62897;
  solution[63]=0.407787;

  for (unsigned int i=0; i<64; ++i)
    if (index_set.is_element(i)==true)
      REQUIRE(std::fabs(flux_moments[i]-solution[i])<1e-3);
}

TEST_CASE("Check One-Group Radiative Transfer for 2D on 4 processors using patches and the CAP-PFB scheduler",
    "[multiple cells patch][CAP-PFB scheduler]")
{
  ConditionalOStream pcout(std::cout,Utilities::MPI::this_mpi_process(
        MPI_COMM_WORLD)==0);
  std::string parameters_filename("rt_parameters_multiple_cells_patch.inp");
  Parameters parameters(parameters_filename);
  std::string geometry_filename(parameters.get_geometry_filename());
  std::string xs_filename(parameters.get_xs_filename());
  FE_DGQ<2> fe(parameters.get_fe_order());
  Geometry<2> geometry(pcout,geometry_filename,fe);
  RTMaterialProperties material_properties(xs_filename,geometry.get_n_materials(),
      parameters.get_n_groups());

  DoFHandler<2>* dof_handler(geometry.get_dof_handler());
  IndexSet index_set(dof_handler->locally_owned_dofs());
  GLC quad(parameters.get_sn_order(),material_properties.get_L_max(),
      parameters.get_galerkin());
  quad.build_quadrature(parameters.get_weight_sum(),2);
  
  std::vector<TrilinosWrappers::MPI::Vector> group_flux(1,
      TrilinosWrappers::MPI::Vector (index_set,MPI_COMM_WORLD));
  TrilinosWrappers::MPI::Vector flux_moments(index_set,MPI_COMM_WORLD);
  Epetra_MpiComm comm(MPI_COMM_WORLD);
  Epetra_Map map(index_set.make_trilinos_map());

  // Create the scheduler.
  unsigned int max_iter(4);
  std::shared_ptr<Scheduler<2,4>> scheduler(new CAPPFBScheduler<2,4> (&quad,&comm,pcout,max_iter));

  // Create the RadiativeTransfer object
  RadiativeTransfer<2,4> radiative_transfer(1,dof_handler->n_dofs(),&fe,
      geometry.get_triangulation(),dof_handler,&parameters,&quad,&material_properties,
      &comm,&map,scheduler);

  // Create the FECells and compute the sweep ordering
  radiative_transfer.setup();

  // Set the current group
  radiative_transfer.set_group(0);

  // Compute the right-hand side
  TrilinosWrappers::MPI::Vector rhs(flux_moments);
  std::list<double*> buffers;
  std::list<MPI_Request*> requests;
  scheduler->start();
  while (scheduler->get_n_tasks_to_execute()!=0)
  {
    radiative_transfer.sweep(*(scheduler->get_next_task()),buffers,
        requests,rhs.trilinos_vector(),&group_flux);
    scheduler->free_buffers(buffers,requests);
  }

  SolverControl solver_control(parameters.get_max_inner_it(),
      parameters.get_inner_tolerance());
  TrilinosWrappers::SolverBicgstab solver(solver_control);
  TrilinosWrappers::PreconditionIdentity preconditioner;
  solver.solve(radiative_transfer,flux_moments,rhs,
      preconditioner);

  // Reference solution
  std::vector<double> solution(64,0.);
  solution[0]=0.407787;
  solution[1]= 0.62897;
  solution[2]= 0.62897;
  solution[3]= 1.02715;
  solution[4]=0.634493;
  solution[5]=0.680499;
  solution[6]= 1.03841;
  solution[7]= 1.11156;
  solution[8]=0.634493;
  solution[9]= 1.03841;
  solution[10]=0.680499;
  solution[11]= 1.11156;
  solution[12]= 1.05037;
  solution[13]= 1.12519;
  solution[14]= 1.12519;
  solution[15]= 1.20627;
  solution[16]=0.680499;
  solution[17]=0.634493;
  solution[18]= 1.11156;
  solution[19]= 1.03841;
  solution[20]= 0.62897;
  solution[21]=0.407787;
  solution[22]= 1.02715;
  solution[23]= 0.62897;
  solution[24]= 1.12519;
  solution[25]= 1.05037;
  solution[26]= 1.20627;
  solution[27]= 1.12519;
  solution[28]= 1.03841;
  solution[29]=0.634493;
  solution[30]= 1.11156;
  solution[31]=0.680499;
  solution[32]=0.680499;
  solution[33]= 1.11156;
  solution[34]=0.634493;
  solution[35]= 1.03841;
  solution[36]= 1.12519;
  solution[37]= 1.20627;
  solution[38]= 1.05037;
  solution[39]= 1.12519;
  solution[40]= 0.62897;
  solution[41]= 1.02715;
  solution[42]=0.407787;
  solution[43]= 0.62897;
  solution[44]= 1.03841;
  solution[45]= 1.11156;
  solution[46]=0.634493;
  solution[47]=0.680499;
  solution[48]= 1.20627;
  solution[49]= 1.12519;
  solution[50]= 1.12519;
  solution[51]= 1.05037;
  solution[52]= 1.11156;
  solution[53]=0.680499;
  solution[54]= 1.03841;
  solution[55]=0.634493;
  solution[56]= 1.11156;
  solution[57]= 1.03841;
  solution[58]=0.680499;
  solution[59]=0.634493;
  solution[60]= 1.02715;
  solution[61]= 0.62897;
  solution[62]= 0.62897;
  solution[63]=0.407787;

  for (unsigned int i=0; i<64; ++i)
    if (index_set.is_element(i)==true)
      REQUIRE(std::fabs(flux_moments[i]-solution[i])<1e-3);
}

int main (int argc, char** argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc,argv,1);

  int result = Catch::Session().run(argc,argv);

  return result;
}
