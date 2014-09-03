/* Copyright (c) 2013, 2014 Bruno Turcksin.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file
 * license.txt for the text and further information on this license.
 */

#include <iostream>
#include <cmath>
#include <memory>
#include <string>
#include "deal.II/base/conditional_ostream.h"
#include "deal.II/base/mpi.h"
#include "deal.II/base/logstream.h"
#include "deal.II/base/utilities.h"
#include "deal.II/fe/fe_dgq.h"
#include "deal.II/grid/grid_tools.h"
#include "deal.II/lac/solver_control.h"
#include "deal.II/lac/trilinos_precondition.h"
#include "deal.II/lac/trilinos_solver.h"
#include "deal.II/lac/trilinos_vector.h"
#include "deal.II/numerics/data_out.h"
#include "Geometry.hh"
#include "GLC.hh"
#include "LS.hh"
#include "Parameters.hh"
#include "RadiativeTransfer.hh"
#include "RTQuadrature.hh"
#include "RandomScheduler.hh"

using namespace dealii;

void create_epetra_map(std::vector<TrilinosWrappers::types::int_type> &indices,
    IndexSet const &index_set,unsigned int const n_locally_owned_dofs,
    unsigned int const n_mom,types::global_dof_index const n_dofs)
{
  std::vector<types::global_dof_index> deal_ii_indices;
  index_set.fill_index_vector(deal_ii_indices);
  indices.resize(n_locally_owned_dofs*n_mom);
  for (unsigned int mom=0; mom<n_mom; ++mom)
    for (unsigned int i=0; i<n_locally_owned_dofs; ++i)
      indices[mom*n_locally_owned_dofs+i] = deal_ii_indices[i]+mom*n_dofs;
}


template<int dim,int tensor_dim>
void solve(ConditionalOStream const &pcout,unsigned int const n_mom, 
    RadiativeTransfer<dim,tensor_dim> &radiative_transfer,
    std::shared_ptr<Scheduler<dim,tensor_dim>> scheduler,
    Parameters const &parameters,Epetra_Map &map,
    std::vector<TrilinosWrappers::MPI::Vector> &group_flux)
{
  // Create the FECells and compute the sweep ordering
  pcout<<"Set up radiative transfer."<<std::endl;
  radiative_transfer.setup();

  if (parameters.get_sn_order()==SI)
  {
    const unsigned int max_inner_it(parameters.get_max_inner_it());
    const unsigned int max_outer_it(parameters.get_max_outer_it());
    const unsigned int n_groups(parameters.get_n_groups());
    const double group_tol(parameters.get_outer_tolerance());
    const double inner_tol(parameters.get_inner_tolerance());
    TrilinosWrappers::MPI::Vector flux_moments(map);
    std::vector<TrilinosWrappers::MPI::Vector> old_group_flux(group_flux);
        
    pcout<<"Start SI solver."<<std::endl;
    for (unsigned int out_it=0; out_it<max_outer_it; ++out_it)
    {
      // Loop over the groups
      for (unsigned int g=0; g<n_groups; ++g)
      {
        // Set the current group
        radiative_transfer.set_group(g);

        TrilinosWrappers::MPI::Vector old_flux_moments(flux_moments);

        for (unsigned int it=0; it<max_inner_it; ++it)
        {
          radiative_transfer.compute_scattering_source(
              flux_moments.trilinos_vector());
          std::list<double*> buffers;
          std::list<MPI_Request*> requests;
          scheduler->initialize_scheduling();
          while (scheduler->get_n_tasks_to_execute()!=0)
          {
            radiative_transfer.sweep(*(scheduler->get_next_task()),buffers,
                requests,flux_moments.trilinos_vector(),&group_flux);
            scheduler->free_buffers(buffers,requests);
          }

          old_flux_moments -= flux_moments;
          const double num(old_flux_moments.l2_norm());
          const double denom(flux_moments.l2_norm());
          if ((num/denom)<inner_tol)
          {
            pcout<<"SI converged at iteration: "<<it<<std::endl;
            break;
          }
          old_flux_moments = flux_moments;
        }

        // Copy flux_moments in group_flux
        TrilinosWrappers::MPI::Vector::iterator flux_moments_it(flux_moments.begin());
        for (unsigned int mom=0; mom<n_mom; ++mom)
        {
          TrilinosWrappers::MPI::Vector::iterator mg_it(
              group_flux[g*n_mom+mom].begin());
          TrilinosWrappers::MPI::Vector::iterator mg_end(
              group_flux[g*n_mom+mom].end());
          for (; mg_it!=mg_end; ++mg_it,++flux_moments_it)
            *mg_it = *flux_moments_it;
        }
      }

      double num(0.);
      double denom(0.);
      for (unsigned int g=0; g<n_groups; ++g)
        for (unsigned int mom=0; mom<n_mom; ++mom)
        {
          old_group_flux[g*n_mom+mom] -= group_flux[g*n_mom+mom];
          num += std::pow(old_group_flux[g*n_mom+mom].l2_norm(),2);
          denom += std::pow(group_flux[g*n_mom+mom].l2_norm(),2);
        }

      if (std::sqrt(num/denom)<group_tol)
      {
        pcout<<"Groups converged at iteration: "<<out_it<<std::endl;
        break;
      }

      for (unsigned int g=0; g<n_groups; ++g)
        for (unsigned int mom=0; mom<n_mom; ++mom)
          old_group_flux[g*n_mom+mom] = group_flux[g*n_mom+mom];
    }
  }
  else
  {
    const unsigned int max_inner_it(parameters.get_max_inner_it());
    const unsigned int max_outer_it(parameters.get_max_outer_it());
    const unsigned int n_groups(parameters.get_n_groups());
    const double group_tol(parameters.get_outer_tolerance());
    const double inner_tol(parameters.get_inner_tolerance());
    TrilinosWrappers::MPI::Vector flux_moments(map);
    std::vector<TrilinosWrappers::MPI::Vector> old_group_flux(group_flux);

    pcout<<"Start Krylov solver."<<std::endl;
    for (unsigned int out_it=0; out_it<max_outer_it; ++out_it)
    {
      // Loop over the groups
      for (unsigned int g=0; g<n_groups; ++g)
      {
        // Set the current group
        radiative_transfer.set_group(g);

        TrilinosWrappers::MPI::Vector rhs(flux_moments);
        std::list<double*> buffers;
        std::list<MPI_Request*> requests;
        scheduler->initialize_scheduling();
        while (scheduler->get_n_tasks_to_execute()!=0)
        {
          radiative_transfer.sweep(*(scheduler->get_next_task()),buffers,
              requests,rhs.trilinos_vector(),&group_flux);
          scheduler->free_buffers(buffers,requests);
        }

        SolverControl solver_control(max_inner_it,inner_tol);
        TrilinosWrappers::PreconditionIdentity preconditioner;
        if (parameters.get_solver_type()==BICGSTAB)
        {
          TrilinosWrappers::SolverBicgstab solver(solver_control);
          solver.solve(radiative_transfer,flux_moments,rhs,preconditioner);
        }
        else
        {
          TrilinosWrappers::SolverGMRES solver(solver_control);
          solver.solve(radiative_transfer,flux_moments,rhs,preconditioner);
        }

        // Copy flux_moments in group_flux
        TrilinosWrappers::MPI::Vector::iterator flux_moments_it(flux_moments.begin());
        for (unsigned int mom=0; mom<n_mom; ++mom)
        {
          TrilinosWrappers::MPI::Vector::iterator mg_it(
              group_flux[g*n_mom+mom].begin());
          TrilinosWrappers::MPI::Vector::iterator mg_end(
              group_flux[g*n_mom+mom].end());
          for (; mg_it!=mg_end; ++mg_it,++flux_moments_it)
            *mg_it = *flux_moments_it;
        }
      }

      double num(0.);
      double denom(0.);
      for (unsigned int g=0; g<n_groups; ++g)
      {
        for (unsigned int mom=0; mom<n_mom; ++mom)
        {
          old_group_flux[g*n_mom+mom] -= group_flux[g*n_mom+mom];
          num += std::pow(old_group_flux[g*n_mom+mom].l2_norm(),2);
          denom += std::pow(group_flux[g*n_mom+mom].l2_norm(),2);
        }
      }

      if (std::sqrt(num/denom)<group_tol)
      {
        pcout<<"Groups converged at iteration: "<<out_it<<std::endl;
        break;
      }

      for (unsigned int g=0; g<n_groups; ++g)
        for (unsigned int mom=0; mom<n_mom; ++mom)
          old_group_flux[g*n_mom+mom] = group_flux[g*n_mom+mom];
    }
  }
}

template<int dim>
void output_results(std::string const &filename,unsigned int const n_mom,
    unsigned int const n_groups,parallel::distributed::Triangulation<dim> const*
    triangulation,DoFHandler<dim> const* dof_handler,
    std::vector<TrilinosWrappers::MPI::Vector> const &group_flux,
    MPI_Comm const &mpi_communicator)
{
  std::ofstream output((filename+Utilities::int_to_string(
          triangulation->locally_owned_subdomain(),4)+".vtu").c_str());
  DataOut<dim> data_out;
  // Atttach dof_handler
  data_out.attach_dof_handler(*dof_handler);

  // Add group_flux
  for (unsigned int g=0; g<n_groups; ++g)
    for (unsigned int mom=0; mom<n_mom; ++mom)
    {
      std::string solution_name("flux_"+Utilities::int_to_string(g,3)+"g_"+
          Utilities::int_to_string(mom,3)+"m");
      data_out.add_data_vector(group_flux[g*n_mom+mom],solution_name);
    }

  // Add subdomain_id
  std::vector<unsigned int> partition_int(triangulation->n_active_cells());
  GridTools::get_subdomain_association(*triangulation,partition_int);
  const Vector<double> partitioning(partition_int.begin(),partition_int.end());
  data_out.add_data_vector(partitioning,"partitioning");

  // Build the patcthes
  data_out.build_patches();

  // Write the output
  data_out.write_vtu(output);

  if (Utilities::MPI::this_mpi_process(mpi_communicator)==0)
  {
    std::vector<std::string> filenames;
    for (unsigned int i=0; i<Utilities::MPI::n_mpi_processes(mpi_communicator); ++i)
      filenames.push_back(filename+Utilities::int_to_string(i,4)+".vtu");
    std::ofstream master_output ((filename+".pvtu").c_str());
    data_out.write_pvtu_record(master_output,filenames);
  }
}

int main(int argc,char **argv)
{  
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc,argv);

  try
  {
    // Suppress output on the screen
    deallog.depth_console(0);

    ConditionalOStream pcout(std::cout,Utilities::MPI::this_mpi_process(
          MPI_COMM_WORLD)==0);

    // Read the parameters
    pcout<<"Read parameters."<<std::endl;
    std::string parameters_filename(argv[argc-1]);
    unsigned int found(parameters_filename.find_last_of("."));
    std::string extension(parameters_filename.substr(found));
    AssertThrow((extension.compare(".inp")==0) || (extension.compare(".txt")==0), 
          ExcMessage("The parameters filename has to use .inp or .txt as extension."));
    Parameters parameters(parameters_filename);
    std::string geometry_filename(parameters.get_geometry_filename());
    std::string xs_filename(parameters.get_xs_filename());
    
    if (parameters.get_dimension()==2)
    {
      FE_DGQ<2> fe(parameters.get_fe_order());
      // Read the geometry
      pcout<<"Read geometry."<<std::endl;
      Geometry<2> geometry(pcout,geometry_filename,fe);
      // Read the material properties for the radiative transfer problem
      pcout<<"Read material properties."<<std::endl;
      RTMaterialProperties material_properties(xs_filename,geometry.get_n_materials(),
          parameters.get_n_groups());

      // Create the DoFHandler
      pcout<<"Create DoFHandler"<<std::endl;
      DoFHandler<2>* dof_handler(geometry.get_dof_handler());
      types::global_dof_index n_dofs(dof_handler->n_dofs());
      unsigned int n_locally_owned_dofs(dof_handler->n_locally_owned_dofs());
      IndexSet index_set(dof_handler->locally_owned_dofs());
      pcout<<"Number of cells: "<<geometry.get_triangulation()->n_global_active_cells()<<std::endl;
      pcout<<"Number of dofs: "<<dof_handler->n_dofs()<<std::endl;

      // Build the quadrature
      RTQuadrature* quad(nullptr);
      if (parameters.get_quad_type()==GLC_QUAD)
        quad = new GLC(parameters.get_sn_order(),material_properties.get_L_max(),
            parameters.get_galerkin());
      else
        quad = new LS(parameters.get_sn_order(),material_properties.get_L_max(),
            parameters.get_galerkin());
      pcout<<"Build the quadarature."<<std::endl;
      quad->build_quadrature(parameters.get_weight_sum(),2);
      const unsigned int n_mom(quad->get_n_mom());

      std::vector<TrilinosWrappers::MPI::Vector> group_flux(parameters.get_n_groups()*
          quad->get_n_mom(),TrilinosWrappers::MPI::Vector (index_set,MPI_COMM_WORLD));
      Epetra_MpiComm comm(MPI_COMM_WORLD);
      // The map is different than the index_set because each moment of the
      // flux has n_dofs
      std::vector<TrilinosWrappers::types::int_type> indices;
      create_epetra_map(indices,index_set,n_locally_owned_dofs,n_mom,n_dofs);
      Epetra_Map map(n_dofs,n_locally_owned_dofs,&indices[0],0,comm);

      // Create the RadiativeTransfer object and solve the problem
      switch (parameters.get_fe_order())
      {
        case 1 :
          {
            //TODO: use a factory function which return a shared_ptr
            std::shared_ptr<Scheduler<2,4>> scheduler(
                new RandomScheduler<2,4> (quad,&comm));
            RadiativeTransfer<2,4> radiative_transfer(parameters.get_n_groups(),
                n_dofs,&fe,geometry.get_triangulation(),dof_handler,&parameters,
                quad,&material_properties,&comm,&map,scheduler);
            solve(pcout,n_mom,radiative_transfer,scheduler,parameters,map,group_flux);
            break;
          }
        case 2 :
          {
            std::shared_ptr<Scheduler<2,9>> scheduler(
                new RandomScheduler<2,9> (quad,&comm));
            RadiativeTransfer<2,9> radiative_transfer(parameters.get_n_groups(),
                n_dofs,&fe,geometry.get_triangulation(),dof_handler,&parameters,
                quad,&material_properties,&comm,&map,scheduler);
            solve(pcout,n_mom,radiative_transfer,scheduler,parameters,map,group_flux);
            break;
          }
        case 3 :
          {
            std::shared_ptr<Scheduler<2,16>> scheduler(
                new RandomScheduler<2,16> (quad,&comm));
            RadiativeTransfer<2,16> radiative_transfer(parameters.get_n_groups(),
                n_dofs,&fe,geometry.get_triangulation(),dof_handler,&parameters,
                quad,&material_properties,&comm,&map,scheduler);
            solve(pcout,n_mom,radiative_transfer,scheduler,parameters,map,group_flux);
            break;
          }
        case 4 :
          {
            std::shared_ptr<Scheduler<2,25>> scheduler(
                new RandomScheduler<2,25> (quad,&comm));
            RadiativeTransfer<2,25> radiative_transfer(parameters.get_n_groups(),
                n_dofs,&fe,geometry.get_triangulation(),dof_handler,&parameters,
                quad,&material_properties,&comm,&map,scheduler);
            solve(pcout,n_mom,radiative_transfer,scheduler,parameters,map,group_flux);
            break;
          }
        case 5 :
          {
            std::shared_ptr<Scheduler<2,36>> scheduler(
                new RandomScheduler<2,36> (quad,&comm));
            RadiativeTransfer<2,36> radiative_transfer(parameters.get_n_groups(),
                n_dofs,&fe,geometry.get_triangulation(),dof_handler,&parameters,
                quad,&material_properties,&comm,&map,scheduler);
            solve(pcout,n_mom,radiative_transfer,scheduler,parameters,map,group_flux);
            break;
          }
        default :
          {
            AssertThrow(false,ExcMessage("FE order should be less or equal to 5."));
          }
      }         

      pcout<<"Output results."<<std::endl;
      output_results(parameters.get_output_filename(),n_mom,parameters.get_n_groups(),
          geometry.get_triangulation(),dof_handler,group_flux,MPI_COMM_WORLD);

      if (quad!=nullptr)
      {
        delete quad;
        quad = nullptr;
      }
    }
    else
    {
      FE_DGQ<3> fe(parameters.get_fe_order());
      // Read the geometry
      pcout<<"Read geometry."<<std::endl;
      Geometry<3> geometry(pcout,geometry_filename,fe);
      // Read the material properties for the radiative transfer problem
      pcout<<"Read material properties."<<std::endl;
      RTMaterialProperties material_properties(xs_filename,geometry.get_n_materials(),
          parameters.get_n_groups());

      // Create the DoFHandler
      DoFHandler<3>* dof_handler(geometry.get_dof_handler());
      types::global_dof_index n_dofs(dof_handler->n_dofs());
      unsigned int n_locally_owned_dofs(dof_handler->n_locally_owned_dofs());
      IndexSet index_set(n_locally_owned_dofs);

      // Build the quadrature
      RTQuadrature* quad(nullptr);
      if (parameters.get_quad_type()==GLC_QUAD)
        quad = new GLC(parameters.get_sn_order(),material_properties.get_L_max(),
            parameters.get_galerkin());
      else
        quad = new LS(parameters.get_sn_order(),material_properties.get_L_max(),
            parameters.get_galerkin());
      pcout<<"Build the quadarature."<<std::endl;
      quad->build_quadrature(parameters.get_weight_sum(),3);
      const unsigned int n_mom(quad->get_n_mom());

      std::vector<TrilinosWrappers::MPI::Vector> group_flux(parameters.get_n_groups()*
          quad->get_n_mom(),TrilinosWrappers::MPI::Vector (index_set,MPI_COMM_WORLD));
      Epetra_MpiComm comm(MPI_COMM_WORLD);
      // The map is different than the index_set because each moment of the
      // flux has n_dofs
      std::vector<TrilinosWrappers::types::int_type> indices;
      create_epetra_map(indices,index_set,n_locally_owned_dofs,quad->get_n_mom(),
          n_dofs);
      Epetra_Map map(n_dofs,n_locally_owned_dofs,&indices[0],0,comm);

      // Create the RadiativeTransfer object and solve the problem
      switch (parameters.get_fe_order())
      {
        case 1 :
          {
            std::shared_ptr<Scheduler<3,8>> scheduler(
                new RandomScheduler<3,8> (quad,&comm));
            RadiativeTransfer<3,8> radiative_transfer(parameters.get_n_groups(),
                n_dofs,&fe,geometry.get_triangulation(),dof_handler,&parameters,
                quad,&material_properties,&comm,&map,scheduler);
            solve(pcout,n_mom,radiative_transfer,scheduler,parameters,map,group_flux);
            break;
          }
        case 2 :
          {
            std::shared_ptr<Scheduler<3,27>> scheduler(
                new RandomScheduler<3,27> (quad,&comm));
            RadiativeTransfer<3,27> radiative_transfer(parameters.get_n_groups(),
                n_dofs,&fe,geometry.get_triangulation(),dof_handler,&parameters,
                quad,&material_properties,&comm,&map,scheduler);
            solve(pcout,n_mom,radiative_transfer,scheduler,parameters,map,group_flux);
            break;
          }
        case 3 :
          {
            std::shared_ptr<Scheduler<3,64>> scheduler(
                new RandomScheduler<3,64> (quad,&comm));
            RadiativeTransfer<3,64> radiative_transfer(parameters.get_n_groups(),
                n_dofs,&fe,geometry.get_triangulation(),dof_handler,&parameters,
                quad,&material_properties,&comm,&map,scheduler);
            solve(pcout,n_mom,radiative_transfer,scheduler,parameters,map,group_flux);
            break;
          }
        case 4 :
          {
            std::shared_ptr<Scheduler<3,125>> scheduler(
                new RandomScheduler<3,125> (quad,&comm));
            RadiativeTransfer<3,125> radiative_transfer(parameters.get_n_groups(),
                n_dofs,&fe,geometry.get_triangulation(),dof_handler,&parameters,
                quad,&material_properties,&comm,&map,scheduler);
            solve(pcout,n_mom,radiative_transfer,scheduler,parameters,map,group_flux);
            break;
          }
        case 5 :
          {
            std::shared_ptr<Scheduler<3,216>> scheduler(
                new RandomScheduler<3,216> (quad,&comm));
            RadiativeTransfer<3,216> radiative_transfer(parameters.get_n_groups(),
                n_dofs,&fe,geometry.get_triangulation(),dof_handler,&parameters,
                quad,&material_properties,&comm,&map,scheduler);
            solve(pcout,n_mom,radiative_transfer,scheduler,parameters,map,group_flux);
            break;
          }
        default :
          {
            AssertThrow(false,ExcMessage("FE order should be less or equal to 5."));
          }
      }

      pcout<<"Output results."<<std::endl;
      output_results(parameters.get_output_filename(),n_mom,parameters.get_n_groups(),
          geometry.get_triangulation(),dof_handler,group_flux,MPI_COMM_WORLD);

      if (quad!=nullptr)
      {
        delete quad;
        quad = nullptr;
      }
    }
  }
  catch(std::exception &exc)
  {
    std::cerr<<std::endl<<std::endl
             <<"-----------------------------------"
             <<std::endl;
    std::cerr<<"Exception on processing: "<<std::endl
             <<exc.what()<<std::endl
             <<std::endl
             <<"Aborting!"<<std::endl
             <<"-----------------------------------"
             <<std::endl;

    return 1;
  }
  catch(...)
  {
    std::cerr<<std::endl<<std::endl
             <<"-----------------------------------"
             <<std::endl;
    std::cerr<<"Unknown exception!" <<std::endl
             <<"Aborting!"<<std::endl
             <<"-----------------------------------"
             <<std::endl;

    return 1;
  }

  return 0;
}
