/* Copyright (c) 2013, Bruno Turcksin.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file
 * license.txt for the text and further information on this license.
 */

#include <iostream>
#include <cmath>
#include <string>
#include "deal.II/base/mpi.h"
#include "deal.II/base/logstream.h"
#include "deal.II/base/utilities.h"
#include "deal.II/fe/fe_dgq.h"
#include "deal.II/lac/solver_control.h"
#include "deal.II/lac/trilinos_precondition.h"
#include "deal.II/lac/trilinos_solver.h"
#include "deal.II/lac/trilinos_vector.h"
#include "Geometry.hh"
#include "GLC.hh"
#include "LS.hh"
#include "Parameters.hh"
#include "RadiativeTransfer.hh"
#include "RTQuadrature.hh"

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
void solve(RadiativeTransfer<dim,tensor_dim> &radiative_transfer,
    Parameters const &parameters,IndexSet &index_set,
    std::vector<TrilinosWrappers::MPI::Vector> &group_flux)
{
  // Create the FECells and compute the sweep ordering
  radiative_transfer.setup();

  if (parameters.get_sn_order()==SI)
  {
    const unsigned int max_inner_it(parameters.get_max_inner_it());
    const unsigned int max_outer_it(parameters.get_max_outer_it());
    const unsigned int n_groups(parameters.get_n_groups());
    const double group_tol(parameters.get_outer_tolerance());
    const double inner_tol(parameters.get_inner_tolerance());
    TrilinosWrappers::MPI::Vector flux_moments(index_set,MPI_COMM_WORLD);
    std::vector<TrilinosWrappers::MPI::Vector> old_group_flux(group_flux);

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
          radiative_transfer.initialize_scheduler();
          while (radiative_transfer.get_n_tasks_to_execute()!=0)
          {
            radiative_transfer.sweep(*(radiative_transfer.get_next_task()),buffers,
                requests,flux_moments.trilinos_vector(),&group_flux);
            radiative_transfer.free_buffers(buffers,requests);
          }

          old_flux_moments -= flux_moments;
          const double num(old_flux_moments.l2_norm());
          const double denom(flux_moments.l2_norm());
          if ((num/denom)<inner_tol)
          {
            std::cout<<"SI converged at iteration: "<<it<<std::endl;
            break;
          }
          old_flux_moments = flux_moments;
        }
        group_flux[g] = flux_moments;
      }

      double num(0.);
      double denom(0.);
      for (unsigned int g=0; g<n_groups; ++g)
      {
        old_group_flux[g] -= group_flux[g];
        num += std::pow(old_group_flux[g].l2_norm(),2);
        denom += std::pow(group_flux[g].l2_norm(),2);
      }

      if (std::sqrt(num/denom)<group_tol)
      {
        std::cout<<"Groups converged at iteration: "<<out_it<<std::endl;
        break;
      }

      for (unsigned int g=0; g<n_groups; ++g)
        old_group_flux[g] = group_flux[g];
    }
  }
  else
  {
    const unsigned int max_inner_it(parameters.get_max_inner_it());
    const unsigned int max_outer_it(parameters.get_max_outer_it());
    const unsigned int n_groups(parameters.get_n_groups());
    const double group_tol(parameters.get_outer_tolerance());
    const double inner_tol(parameters.get_inner_tolerance());
    TrilinosWrappers::MPI::Vector flux_moments(index_set,MPI_COMM_WORLD);
    std::vector<TrilinosWrappers::MPI::Vector> old_group_flux(group_flux);

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
        radiative_transfer.initialize_scheduler();
        while (radiative_transfer.get_n_tasks_to_execute()!=0)
        {
          radiative_transfer.sweep(*(radiative_transfer.get_next_task()),buffers,
              requests,rhs.trilinos_vector(),&group_flux);
          radiative_transfer.free_buffers(buffers,requests);
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
        group_flux[g] = flux_moments;
      }

      double num(0.);
      double denom(0.);
      for (unsigned int g=0; g<n_groups; ++g)
      {
        old_group_flux[g] -= group_flux[g];
        num += std::pow(old_group_flux[g].l2_norm(),2);
        denom += std::pow(group_flux[g].l2_norm(),2);
      }

      if (std::sqrt(num/denom)<group_tol)
      {
        std::cout<<"Groups converged at iteration: "<<out_it<<std::endl;
        break;
      }

      for (unsigned int g=0; g<n_groups; ++g)
        old_group_flux[g] = group_flux[g];
    }
  }
}

int main(int argc,char **argv)
{  
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc,argv);

  try
  {
    // Suppress output on the screen
    deallog.depth_console(0);

    // Read the parameters
    Parameters parameters(argv[argc-1]);
    std::string geometry_filename(parameters.get_geometry_filename());
    std::string xs_filename(parameters.get_xs_filename());
    
    if (parameters.get_dimension()==2)
    {
      FE_DGQ<2> fe(parameters.get_fe_order());
      // Read the geometry
      Geometry<2> geometry(geometry_filename,fe);
      // Read the material properties for the radiative transfer problem
      RTMaterialProperties material_properties(xs_filename,geometry.get_n_materials(),
          parameters.get_n_groups());

      // Create the DoFHandler
      DoFHandler<2>* dof_handler(geometry.get_dof_handler());
      types::global_dof_index n_dofs(dof_handler->n_dofs());
      unsigned int n_locally_owned_dofs(dof_handler->n_locally_owned_dofs());
      IndexSet index_set(n_locally_owned_dofs);

      // Buil the quadrature
      RTQuadrature* quad(nullptr);
      if (parameters.get_quad_type()==GLC_QUAD)
        quad = new GLC(parameters.get_sn_order(),material_properties.get_L_max(),
            parameters.get_galerkin());
      else
        quad = new LS(parameters.get_sn_order(),material_properties.get_L_max(),
            parameters.get_galerkin());
      quad->build_quadrature(parameters.get_weight_sum());

      std::vector<TrilinosWrappers::MPI::Vector> group_flux(parameters.get_n_groups(),
          TrilinosWrappers::MPI::Vector (index_set,MPI_COMM_WORLD));
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
            RadiativeTransfer<2,4> radiative_transfer(parameters.get_n_groups(),
                n_dofs,&fe,geometry.get_triangulation(),dof_handler,&parameters,
                quad,&material_properties,&comm,&map);
            solve(radiative_transfer,parameters,index_set,group_flux);
            break;
          }
        case 2 :
          {
            RadiativeTransfer<2,9> radiative_transfer(parameters.get_n_groups(),
                n_dofs,&fe,geometry.get_triangulation(),dof_handler,&parameters,
                quad,&material_properties,&comm,&map);
            solve(radiative_transfer,parameters,index_set,group_flux);
            break;
          }
        case 3 :
          {
            RadiativeTransfer<2,16> radiative_transfer(parameters.get_n_groups(),
                n_dofs,&fe,geometry.get_triangulation(),dof_handler,&parameters,
                quad,&material_properties,&comm,&map);
            solve(radiative_transfer,parameters,index_set,group_flux);
            break;
          }
        case 4 :
          {
            RadiativeTransfer<2,25> radiative_transfer(parameters.get_n_groups(),
                n_dofs,&fe,geometry.get_triangulation(),dof_handler,&parameters,
                quad,&material_properties,&comm,&map);
            solve(radiative_transfer,parameters,index_set,group_flux);
            break;
          }
        case 5 :
          {
            RadiativeTransfer<2,36> radiative_transfer(parameters.get_n_groups(),
                n_dofs,&fe,geometry.get_triangulation(),dof_handler,&parameters,
                quad,&material_properties,&comm,&map);
            solve(radiative_transfer,parameters,index_set,group_flux);
            break;
          }
      }         

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
      Geometry<3> geometry(geometry_filename,fe);
      // Read the material properties for the radiative transfer problem
      RTMaterialProperties material_properties(xs_filename,geometry.get_n_materials(),
          parameters.get_n_groups());

      // Create the DoFHandler
      DoFHandler<3>* dof_handler(geometry.get_dof_handler());
      types::global_dof_index n_dofs(dof_handler->n_dofs());
      unsigned int n_locally_owned_dofs(dof_handler->n_locally_owned_dofs());
      IndexSet index_set(n_locally_owned_dofs);

      // Buil the quadrature
      RTQuadrature* quad(nullptr);
      if (parameters.get_quad_type()==GLC_QUAD)
        quad = new GLC(parameters.get_sn_order(),material_properties.get_L_max(),
            parameters.get_galerkin());
      else
        quad = new LS(parameters.get_sn_order(),material_properties.get_L_max(),
            parameters.get_galerkin());
      quad->build_quadrature(parameters.get_weight_sum());

      std::vector<TrilinosWrappers::MPI::Vector> group_flux(parameters.get_n_groups(),
          TrilinosWrappers::MPI::Vector (index_set,MPI_COMM_WORLD));
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
            RadiativeTransfer<3,8> radiative_transfer(parameters.get_n_groups(),
                n_dofs,&fe,geometry.get_triangulation(),dof_handler,&parameters,
                quad,&material_properties,&comm,&map);
            solve(radiative_transfer,parameters,index_set,group_flux);
            break;
          }
        case 2 :
          {
            RadiativeTransfer<3,27> radiative_transfer(parameters.get_n_groups(),
                n_dofs,&fe,geometry.get_triangulation(),dof_handler,&parameters,
                quad,&material_properties,&comm,&map);
            solve(radiative_transfer,parameters,index_set,group_flux);
            break;
          }
        case 3 :
          {
            RadiativeTransfer<3,64> radiative_transfer(parameters.get_n_groups(),
                n_dofs,&fe,geometry.get_triangulation(),dof_handler,&parameters,
                quad,&material_properties,&comm,&map);
            solve(radiative_transfer,parameters,index_set,group_flux);
            break;
          }
        case 4 :
          {
            RadiativeTransfer<3,125> radiative_transfer(parameters.get_n_groups(),
                n_dofs,&fe,geometry.get_triangulation(),dof_handler,&parameters,
                quad,&material_properties,&comm,&map);
            solve(radiative_transfer,parameters,index_set,group_flux);
            break;
          }
        case 5 :
          {
            RadiativeTransfer<3,216> radiative_transfer(parameters.get_n_groups(),
                n_dofs,&fe,geometry.get_triangulation(),dof_handler,&parameters,
                quad,&material_properties,&comm,&map);
            solve(radiative_transfer,parameters,index_set,group_flux);
            break;
          }
      }

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
