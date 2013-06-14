/* Copyright (c) 2013, Bruno Turcksin.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file
 * license.txt for the text and further information on this license.
 */

#ifndef _RADIATIVETRANSFER_HH_
#define _RADIATIVETRANSFER_HH_

template<int dim>
class RadiativeTransfer:
{
  public :
    RadiativeTransfer(parallel:distributed::Triangulation<dim>* triangulation,
        DoFHandler<dim>* dof_handler,Parameters* parameters,
        RTMaterialProperties* material_properties,Quadrature* quad);

    /// Create all the matrices that are need to solve the transport equation
    /// and compute the sweep ordering.
    void setup();
    
  private :
    /// Compute the ordering of the cell for the sweeps.
    void compute_sweep_ordering();
};

#endif
