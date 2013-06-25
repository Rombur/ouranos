/* Copyright (c) 2013, Bruno Turcksin.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file
 * license.txt for the text and further information on this license.
 */

#ifndef _RADIATIVETRANSFER_HH_
#define _RADIATIVETRANSFER_HH_

#include <algorithm>
#include <list>
#include <set>
#include <vector>
#include "Epetra_Comm.h"
#include "Epetra_Map.h"
#include "Epetra_MultiVector.h"
#include "Epetra_Operator.h"
#include "deal.II/base/exceptions.h"
#include "deal.II/base/quadrature_lib.h"
#include "deal.II/distributed/tria.h"
#include "deal.II/dofs/dof_handler.h"
#include "deal.II/fe/fe_dgq.h"
#include "deal.II/fe/fe_values.h"
#include "deal.II/lac/full_matrix.h"
#include "deal.II/lac/vector.h"
#include "FECell.hh"
#include "Parameters.hh"
#include "RTQuadrature.hh"
#include "RTMaterialProperties.hh"

using namespace dealii;

typedef std::vector<unsigned int> ui_vector;

/**
 *  This class derives from Epetra_Operator and implement the function Apply
 *  needed by the AztecOO solvers.
 */

template <int dim,int tensor_dim>
class RadiativeTransfer : public Epetra_Operator
{
  public :
    RadiativeTransfer(parallel::distributed::Triangulation<dim>* triangulation,
        DoFHandler<dim>* dof_handler,Parameters* parameters,RTQuadrature* quad,
        RTMaterialProperties* material_properties);


    /// Create all the matrices that are need to solve the transport equation
    /// and compute the sweep ordering.
    void setup();

    /// Return the result of the transport operator applied to x in. Return 0
    /// if successful.
    int Apply(Epetra_MultiVector const &x,Epetra_MultiVector &y) const;

    /// Compute the scattering given a flux.
    void compute_scattering_source(Epetra_MultiVector const &x) const;

    /// Perform a sweep. If group_flux is nullptr is false, the surfacic and 
    /// the volumetric sources are not included in the sweep.
    /// @todo The sweep can be optimized to use less memory (only keep the
    /// front wave).
    void sweep(Epetra_MultiVector &flux_moments,unsigned int idir,
        Epetra_MultiVector const* const group_flux=nullptr) const;

    /// This method is not implemented.
    int SetUseTranspose(bool UseTranspose) {return 0;};

    /// This method is not implemented.
    int ApplyInverse(Epetra_MultiVector const &x,Epetra_MultiVector &y) const
    {return 0;};

    /// This method is not implemented.
    double NormInf() const {return 0.;};

    /// Return a character string describing the operator.
    char const* Label() const {return "radiative_transfer";};

    /// Return the UseTranspose setting (always false).
    bool UseTranspose() const {return "false";};

    /// Retun true if this object can provide an approximate Inf-norm (always
    /// false).
    bool HasNormInf() const {return false;};

    /// Return pointer to the Epetra_Comm communicator associated with this
    /// operator.
//    Epetra_Comm const& Comm() const;

    /// Return the Epetra_Map object associated with the domain of this
    /// operator.
//    Epetra_Map const& OperatorDomainMap() const;

    /// Return the Epetra_Map object associated with the range of this
    /// operator.
//    Epetra_Map const& OperatorRangeMap() const;
    
    /// Set the current group.
    void Set_group(unsigned int g);    

  private :
    /// Compute the ordering of the cell for the sweeps.
    void compute_sweep_ordering();
    
    /// Compute the scattering source due to the upscattering and the
    /// downscattering to the current group.
    void compute_outer_scattering_source(Tensor<1,tensor_dim> &b,
        Epetra_MultiVector const* const group_flux,
        FECell<dim,tensor_dim> const* const fecell,const unsigned int idir) const;

    /// Number of moments.
    unsigned int n_mom;
    /// Current group.
    unsigned int group;
    /// Number of groups.
    unsigned int n_groups;
    /// Number of cells owned by the current processor.
    unsigned int n_cells;
    /// Pointer to Epetra_Map associated to flux_moments and group_flux
    Epetra_Map* map;
    /// Sweep ordering associated to the different direction.
    std::vector<ui_vector> sweep_order;
    /// Scattering source for each moment.
    std::vector<Vector<double>*> scattering_source;
    /// FECells owned by the current processor.
    std::vector<FECell<dim,tensor_dim> > fecell_mesh;
    /// Pointer to the distributed triangulation.
    parallel::distributed::Triangulation<dim>* triangulation;
    /// Pointer to the dof_handler.
    DoFHandler<dim>* dof_handler;
    /// Pointer to the parameters.
    Parameters* parameters;
    /// Pointer to the quadrature.
    RTQuadrature* quad;
    /// Pointer to the material property.
    RTMaterialProperties* material_properties;
};

#endif
