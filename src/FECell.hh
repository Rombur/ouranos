/* Copyright (c) 2013, Bruno Turcksin.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file
 * license.txt for the text and further information on this license.
 */

#ifndef _FECELL_HH_
#define _FECELL_HH_

#include <vector>
#include "deal.II/base/tensor.h"
#include "deal.II/dofs/dof_handler.h"
#include "deal.II/fe/fe_values.h"
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
        typename DoFHandler<dim>::active_cell_iterator &cell);

    /// Return the mass matrix \f$\int_D b_i\ b_j\ dr\f$.
    Tensor<2,tensor_dim> &get_mass_matrix();

    /// Return the kth components of the gradient matrix, i.e.,
    /// \f$\int_D b_i\ \partial_k b_j\ dr\f$.
    Tensor<2,tensor_dim> &get_grad_matrix(unsigned int k);

  private :
    /// Mass matrix \f$\int_D b_i\ b_j\ dr\f$.
    Tensor<2,tensor_dim> mass_matrix;

    /// Vector of the matrices correspondant to the components of gradient
    /// matrix \f$\int_D b_i\ \nabla b_j\ dr\f$.
    std::vector<Tensor<2,tensor_dim> > grad_matrices;
};

template <int dim,int tensor_dim>
inline Tensor<2,tensor_dim>& FECell<dim,tensor_dim>::get_mass_matrix()
{
  return mass_matrix;
}

template <int dim,int tensor_dim>
inline Tensor<2,tensor_dim>& FECell<dim,tensor_dim>::get_grad_matrix(unsigned int k)
{
  return grad_matrices[k];
}

#endif
