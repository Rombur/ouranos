/* Copyright (c) 2013, Bruno Turcksin.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file
 * license.txt for the text and further information on this license.
 */

#ifndef _FECELL_HH_
#define _FECELL_HH_

#include <vector>
#include "deal.II/base/exceptions.h"
#include "deal.II/base/tensor.h"
#include "deal.II/dofs/dof_handler.h"
#include "deal.II/fe/fe_values.h"

using namespace dealii;

/**
 * This class builds and stores all the finite element matrices on a given
 * cell needed to solve the radiative transfer.
 */

template <int dim,int tensor_dim>
class FECell
{
  public :
    FECell(const unsigned int n_q_points,const unsigned int n_face_q_points,
        FEValues<dim> &fe_values,FEFaceValues<dim> &fe_face_values,
        FEFaceValues<dim> &fe_neighbor_face_values,
        typename DoFHandler<dim>::active_cell_iterator &cell,
        typename DoFHandler<dim>::active_cell_iterator &end_cell);

    /// Return the material id of the current cell
    unsigned int get_material_id() const;

    /// Return the source id of the current cell
    unsigned int get_source_id() const;

    /// Return the active_cell_itertor of the current cell.
    typename DoFHandler<dim>::active_cell_iterator& get_cell();

    /// Return a pointer to the mass matrix.
    Tensor<2,tensor_dim> const* const get_mass_matrix() const;

    /// Return a pointer to the kth components of the gradient matrix.
    Tensor<2,tensor_dim> const* const get_grad_matrix(unsigned int k) const;

    /// Return a pointer to the downwind matrix associated to a given face.
    Tensor<2,tensor_dim> const* const get_downwind_matrix(unsigned int face) const;

    /// Return a pointer to the upwind matrix associated to a given face.
    Tensor<2,tensor_dim> const* const get_upwind_matrix(unsigned int face) const;

  private :
    /// Material id of the current cell.
    unsigned int material_id;
    /// Source id of the current cell.
    unsigned int source_id;
    /// Current cell.
    typename DoFHandler<dim>::active_cell_iterator cell;
    /// Mass matrix \f$\int_D b_i\ b_j\ dr\f$.
    Tensor<2,tensor_dim> mass_matrix;
    /// Vector of the matrices correspondant to the components of gradient
    /// matrix \f$\int_D b_i\ \nabla b_j\ dr\f$.
    std::vector<Tensor<2,tensor_dim> > grad_matrices;
    /// Downwind matrices $\f\int_E b_i\ b_j\ dr\f$ where \f$b_i\f$ and
    /// \f$b_j\f$ are definde on the same cell.
    std::vector<Tensor<2,tensor_dim> > downwind_matrices;
    /// Upwind matrices \f$\int_{E_c} b_i\ b_j\ dr\f$ where \f$b_i\f$ and
    /// \f$b_j\f$ are defined on different cell.
    std::vector<Tensor<2,tensor_dim> > upwind_matrices;
};

template <int dim,int tensor_dim>
inline unsigned int FECell<dim,tensor_dim>::get_material_id() const
{
  return material_id;
}

template <int dim,int tensor_dim>
inline unsigned int FECell<dim,tensor_dim>::get_source_id() const
{
  return source_id;
}

template <int dim,int tensor_dim>
inline typename DoFHandler<dim>::active_cell_iterator& 
FECell<dim,tensor_dim>::get_cell()
{
  return cell;
}

template <int dim,int tensor_dim>
inline Tensor<2,tensor_dim> const* const 
FECell<dim,tensor_dim>::get_mass_matrix() const
{
  return &mass_matrix;
}

template <int dim,int tensor_dim>
inline Tensor<2,tensor_dim> const* const 
FECell<dim,tensor_dim>::get_grad_matrix(unsigned int k) const
{
  AssertIndexRange(i,dim);
  return &grad_matrices[k];
}

template <int dim,int tensor_dim>
inline Tensor<2,tensor_dim> const* const
FECell<dim,tensor_dim>::get_downwind_matrix(unsigned int face) const
{
  AssertIndexRange(i,2*dim);
  return &downwind_matrices[face];
}

template <int dim,int tensor_dim>
inline Tensor<2,tensor_dim> const* const
FECell<dim,tensor_dim>::get_upwind_matrix(unsigned int face) const
{
  AssertIndexRange(i,2*dim);
  return &upwind_matrices[face];
}

#endif
