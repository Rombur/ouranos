/* Copyright (c) 2013, Bruno Turcksin.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file
 * license.txt for the text and further information on this license.
 */

#ifndef _RADIATIVETRANSFER_HH_
#define _RADIATIVETRANSFER_HH_

#include <algorithm>
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
#include "FECell.hh"
#include "Parameters.hh"
#include "RTQuadrature.hh"

using namespace dealii;

/**
 *  This class derives from Epetra_Operator and implement the function Apply
 *  needed by the AztecOO solvers.
 */

template <int dim,int tensor_dim>
class RadiativeTransfer : public Epetra_Operator
{
  public :
    RadiativeTransfer(parallel::distributed::Triangulation<dim>* triangulation,
        DoFHandler<dim>* dof_handler,Parameters* parameters,RTQuadrature* quad);


    /// Create all the matrices that are need to solve the transport equation
    /// and compute the sweep ordering.
    void setup();

    /// Return the result of the transport operator applied to x in. Return 0
    /// if successful.
//    int Apply(Epetra_MultiVector const &x,Epetra_MultiVector &y) const;

    /// Compute the scattering given a flux.
//    void Compute_scattering_source(Epetra_MultiVector const &x) const;

    /// Perform a sweep. If rhs is false, the surfacic and the volumetric
    /// sources are not included in the sweep.
//    void Sweep(Epetra_MultiVector &flux_moments,bool rhs=false) const;

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
    
  private :
    /// Compute the ordering of the cell for the sweeps.
    void compute_sweep_ordering();
                      
    /// Sweep ordering associated to the different direction.
    std::vector<unsigned int> sweep_order;
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
};

#endif
