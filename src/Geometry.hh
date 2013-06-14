/* Copyright (c) 2013, Bruno Turcksin.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file
 * license.txt for the text and further information on this license.
 */

#ifndef _GEOMETRY_HH_
#define _GEOMETRY_HH_

#include <fstream>
#include <string>
#include <vector>
#include "deal.II/base/exceptions.h"
#include "deal.II/base/parameter_handler.h"
#include "deal.II/distributed/tria.h"
#include "deal.II/dofs/dof_handler.h"
#include "deal.II/grid/grid_generator.h"
#include "deal.II/grid/tria_accessor.h"
#include "deal.II/grid/tria_iterator.h"

using namespace dealii;

typedef std::vector<unsigned int> ui_vector;
typedef std::vector<double> d_vector;

/**
 * This class reads the geometry file and creates the distributed
 * triangulation and the dof_handler.
 */

template<int dim>
class Geometry
{
  public :
    Geometry(std::string &geometry_filename);

    /// Return the number of subdivisions of dimension i.
    unsigned int get_n_subdivisions(unsigned int i) const;

    /// Return the material id for cell (i,j,k).
    unsigned int get_material_ids(unsigned int i,unsigned int j,
        unsigned int k=0) const;

    /// Return the source id for cell (i,j,k).
    unsigned int get_source_ids(unsigned int i,unsigned int j,
        unsigned int k=0) const;

    /// Return a pointer to the distributed triangulation.
    parallel::distributed::Triangulation<dim>* get_triangulation();

    /// Return a pointer to the dof handler.
    DoFHandler<dim>* get_dof_handler();

  private :
    /// Declare the parameters and assign default values.
    void declare_parameters(ParameterHandler &prm);

    /// Parse the parameters.
    void parse_parameters(ParameterHandler &prm,d_vector &bot_left,
        d_vector &up_right);

    /// Return a list of n_elements unsigned int from a given string.
    ui_vector get_list_uint(std::string &input,unsigned int n_elements);
    
    /// Return a list of n_elements doubles from a given string.
    d_vector get_list_double(std::string &input,unsigned int n_elements);

    /// Number of subdivisions in each direction.
    ui_vector n_subdivisions;
    /// Material IDs.
    ui_vector material_ids;
    /// Source IDs.
    ui_vector source_ids;
    /// Distributed triangulation.
    parallel::distributed::Triangulation<dim> triangulation;
    /// DoF handler.
    DoFHandler<dim> dof_handler;
};

template<int dim>
inline unsigned int Geometry<dim>::get_n_subdivisions(unsigned int i) const
{
  AssertIndexRange(i,dim);
  return n_subdivisions[i];
}

template<int dim>
inline unsigned int Geometry<dim>::get_material_ids(unsigned int i,unsigned int j,
    unsigned int k) const
{
  AssertIndexRange(i+j*n_subdivisions[0]+k*n_subdivisions[0]*n_subdivisions[1],
      material_ids.size());
  return material_ids[i+j*n_subdivisions[0]+k*n_subdivisions[0]*n_subdivisions[1]];
}

template<int dim>
inline unsigned int Geometry<dim>::get_source_ids(unsigned int i,unsigned int j,
    unsigned int k) const
{
  AssertIndexRange(i+j*n_subdivisions[0]+k*n_subdivisions[0]*n_subdivisions[1],
      material_ids.size());
  return source_ids[i+j*n_subdivisions[0]+k*n_subdivisions[0]*n_subdivisions[1]];
}

template<int dim>
inline parallel::distributed::Triangulation<dim>* Geometry<dim>::get_triangulation()
{
  return &triangulation;
}

template<int dim>
inline DoFHandler<dim>* Geometry<dim>::get_dof_handler()
{
  return &dof_handler;
}

#endif

