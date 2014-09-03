/* Copyright (c) 2013, Bruno Turcksin.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file
 * license.txt for the text and further information on this license.
 */

#ifndef _PARAMETERS_HH_
#define _PARAMETERS_HH_

#include <string>
#include <vector>
#include "deal.II/base/exceptions.h"
#include "deal.II/base/parameter_handler.h"

using namespace dealii;


/// Enum on the different types of boundary conditions.
enum BC_TYPE{VACUUM,ISOTROPIC,MOST_NORMAL,REFLECTIVE};

/// Enum on the quadrature type: GLC (Gauss-Legendre-Chebyshev) or LS (Level
/// Symmetric.
enum QUAD_TYPE{GLC_QUAD,LS_QUAD};

/// Enum on the solver type: SI (Source Iteration), GMRES, or BICGSTAB.
enum SOLVER_TYPE{SI,GMRES,BICGSTAB};

/**
 * This class reads and parses all the parameters needed by ouranos.
 */

class Parameters
{
  public :
    Parameters(const std::string &parameters_filename);

    /// Return the flag on Galerkin quadrature.
    bool get_galerkin() const;

    /// Return the dimension of the problem.
    unsigned int get_dimension() const;

    /// Return the macimum number of inner_iterations.
    unsigned int get_max_inner_it() const;

    /// Return the maximum nimber of outer iterations.
    unsigned int get_max_outer_it() const;

    /// Return Sn order.
    unsigned int get_sn_order() const;

    /// Return the number of adaptive refinement to perform.
    unsigned int get_n_refinements() const;

    /// Return the number of levels of cells when creating the patches.
    unsigned int get_n_levels_patch() const;

    /// Return the order of the finite elements.
    unsigned int get_fe_order() const;

    /// Return the number of groups.
    unsigned int get_n_groups() const;

    /// Return the number of sources.
    unsigned int get_n_src() const;

    /// Return the relative tolerance for the inner solver.
    double get_inner_tolerance() const;

    /// Return the relavive tolerance for the outer solver.
    double get_outer_tolerance() const;

    /// Return the incoming flux on a given boundary for a given group.
    double get_inc_flux(unsigned int face,unsigned int group) const;

    /// Return the sum of the weighrs (1, 2pi, or 4pi).
    double get_weight_sum() const;

    /// Return the refinement factor used in AMR.
    double get_refinement_factor() const;

    /// Return the coarsening factor used in AMR.
    double get_coarsening_factor() const;

    /// Return the intensity of the source i for the group g.
    double get_src(unsigned int i,unsigned int g) const;

    /// Return the boundary condition for a given face (vacuum, isotropic,
    /// most_normal, or reflective).
    BC_TYPE get_bc_type(unsigned int face) const;
    
    /// Return the type of quadrature used by the RT (GLC or LS).
    QUAD_TYPE get_quad_type() const;

    /// Return the type of solver used by the RT (SI, BICGSTAB, or GMRES).
    SOLVER_TYPE get_solver_type() const;

    /// Return the name of the geometry file.
    std::string get_geometry_filename() const;

    /// Return the name of the output file.
    std::string get_output_filename() const;

    /// Return the name of the cross section file.
    std::string get_xs_filename() const;

  private :
    /// Declare the parameters and assign default values.
    void declare_parameters(ParameterHandler &prm);

    /// Parse the parameters.
    void parse_parameters(ParameterHandler &prm);

    /// Return a list of n_elements doubles from a given string.
    std::vector<double> get_list_double(std::string &input,unsigned int n_elements);

    /// Flag for Galerkin quadrature.
    bool galerkin;
    /// Dimension of the problem (two or three).
    unsigned int dim;
    /// Finite elements order.
    unsigned fe_order;
    /// Sn order.
    unsigned int sn_order;
    /// Maximum number of outer iterations for RT solver.
    unsigned int max_out_it;
    /// Maximum number of inner iterations for RT solver.
    unsigned int max_in_it;
    /// Number of groups.
    unsigned int n_groups;
    /// Number of adaptive refinements to perform.
    unsigned int n_refinements;
    /// Number of levels of cells to go up when creating the patches.
    unsigned int n_levels;
    /// Number of sources.
    unsigned int n_src;
    /// Type of quadrature: GLC or LS.
    QUAD_TYPE quad_type;
    /// Type of solver for radiative transfer: SI, GMRES, or BiCGSTAB.
    SOLVER_TYPE solver_type;
    /// Tolerance for the outer radiative transfer solver.
    double outer_tol;
    /// Tolerance for the inner radiative transfer solver.
    double inner_tol;
    /// Sum of the weights for the RT quadrature.
    double weight_sum;
    /// Threshold used for adaptive mesh refinement.
    double refinement_factor;
    /// Threshold used for adaptive mesh coarsening.
    double coarsening_factor;
    /// Name of the geometry file.
    std::string geometry_filename;
    /// Name of the output file.
    std::string output_filename;
    /// Name of the cross sections file.
    std::string xs_filename;
    /// Intensity of the sources.
    std::vector<std::vector<double>> src;
    /// Boundary conditions (xmin,xmax,ymin,ymax,zmin,zmax).
    std::vector<BC_TYPE> bc_type;
    /// Incoming fluxes on the boundaries (faces,groups).
    std::vector<std::vector<double>> inc_flux;
};

inline bool Parameters::get_galerkin() const
{
  return galerkin;
}

inline unsigned int Parameters::get_dimension() const
{
  return dim;
}

inline unsigned int Parameters::get_max_inner_it() const
{
  return max_in_it;
}

inline unsigned int Parameters::get_max_outer_it() const
{
  return max_out_it;
}

inline unsigned int Parameters::get_sn_order() const
{
  return sn_order;
}

inline unsigned int Parameters::get_n_refinements() const
{
  return n_refinements;
}

inline unsigned int Parameters::get_n_levels_patch() const
{
  return n_levels;
}

inline unsigned int Parameters::get_fe_order() const
{
  return fe_order;
}
    
inline unsigned int Parameters::get_n_groups() const
{
  return n_groups;
}
    
inline unsigned int Parameters::get_n_src() const
{
  return n_src;
}

inline std::string Parameters::get_geometry_filename() const
{
  return geometry_filename;
}

inline std::string Parameters::get_output_filename() const
{
  return output_filename;
}

inline std::string Parameters::get_xs_filename() const
{
  return xs_filename;
}
    
inline  double Parameters::get_inner_tolerance() const
{
  return inner_tol;
}

inline double Parameters::get_outer_tolerance() const
{
  return outer_tol;
}

inline double Parameters::get_inc_flux(unsigned int face,unsigned int group) const
{
  AssertIndexRange(face,inc_flux.size());
  AssertIndexRange(group,inc_flux[face].size());
  return inc_flux[face][group];
}

inline double Parameters::get_weight_sum() const
{
  return weight_sum;
}

inline double Parameters::get_refinement_factor() const
{
  return refinement_factor;
}

inline double Parameters::get_coarsening_factor() const
{
  return coarsening_factor;
}

inline double Parameters::get_src(unsigned int i,unsigned int g) const
{
  AssertIndexRange(i,src.size());
  AssertIndexRange(g,src[i].size());
  return src[i][g];
}

inline BC_TYPE Parameters::get_bc_type(unsigned int face) const
{
  AssertIndexRange(face,2*dim);
  return bc_type[face];
}
    
inline QUAD_TYPE Parameters::get_quad_type() const
{
  return quad_type;
}

inline SOLVER_TYPE Parameters::get_solver_type() const
{
  return solver_type;
}

#endif
