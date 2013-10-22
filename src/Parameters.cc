/* Copyright (c) 2013, Bruno Turcksin.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file
 * license.txt for the text and further information on this license.
 */

#include "Parameters.hh"

Parameters::Parameters(const std::string &parameters_filename)
{
  ParameterHandler prm;

  // Declare the parameters 
  declare_parameters(prm);

  // Read the file
  std::ifstream parameters_file(parameters_filename.c_str());
  AssertThrow(parameters_file,ExcMessage("Input parameters file not found."));
  const bool success = prm.read_input(parameters_file);
  AssertThrow(success,ExcMessage("Invalid input parameters file."));
  parameters_file.close();

  // Parse the parameters
  parse_parameters(prm);
}

void Parameters::declare_parameters(ParameterHandler &prm)
{  
  prm.declare_entry("Dimension","2",Patterns::Integer(2,3),
      "Dimension of the problem.");
  prm.declare_entry("FE order","1",Patterns::Integer(1,5),
      "Order of the finite elements.");
  prm.declare_entry("Geometry file","geometry.inp",Patterns::FileName(),
      "Name of the geometry file.");
  prm.declare_entry("Number of refinements","0",Patterns::Integer(0),
      "Number of refinement cycles used for AMR.");
  prm.declare_entry("Refinement factor","0.",Patterns::Double(0.,1.),
      "Refinement factor used for AMR.");
  prm.declare_entry("Coarsening factor","0.",Patterns::Double(0.,1.),
      "Coarsening factor used for AMR.");
  prm.declare_entry("Output file","output",Patterns::FileName(),
      "Name of the output file.");
  prm.enter_subsection("Radiative Transfer parameters");
  {
    prm.declare_entry("Cross sections file","xs.inp",Patterns::FileName(),
        "Name of the cross sections file.");
    prm.declare_entry("Solver type","SI",Patterns::Selection("SI|GMRES|BICGSTAB"),
        "Type of solver (SI, GMRES, or BICGSTAB).");
    prm.declare_entry("Max outer iterations","1",Patterns::Integer(1),
        "Maximum number of outer iterations.");
    prm.declare_entry("Max inner iterations","1",Patterns::Integer(1),
        "Maximum number of inner iterations.");
    prm.declare_entry("Tolerance outer solver","1e-4",Patterns::Double(0.),
        "Tolerance for the outer solver.");
    prm.declare_entry("Tolerance inner solver","1e-6",Patterns::Double(0.),
        "Tolerance for the inner solver.");
    prm.declare_entry("Sn order","2",Patterns::Integer(2),"Sn order.");
    prm.declare_entry("Quadrature type","GLC",Patterns::Selection("LS|GLC"),
        "Type of quadrature (LS or GLC).");
    prm.declare_entry("Sum weight","4PI",Patterns::Selection("1|2PI|4PI"),
        "Sum of the weight of the quadrature.");
    prm.declare_entry("Galerkin","false",Patterns::Bool(),
        "Flag for Galerkin quadrature.");
    prm.declare_entry("Xmin BC","vacuum",
        Patterns::Selection("vacuum|isotropic|most normal|reflective"),
        "Boundary condition for the face with the smallest x.");
    prm.declare_entry("Xmax BC","vacuum",
        Patterns::Selection("vacuum|isotropic|most normal|reflective"),
        "Boundary condition for the face with the largest x.");
    prm.declare_entry("Ymin BC","vacuum",
        Patterns::Selection("vacuum|isotropic|most normal|reflective"),
        "Boundary condition for the face with the smallest y.");
    prm.declare_entry("Ymax BC","vacuum",
        Patterns::Selection("vacuum|isotropic|most normal|reflective"),
        "Boundary condition for the face with the largest y.");
    prm.declare_entry("Zmin BC","vacuum",
        Patterns::Selection("vacuum|isotropic|most normal|reflective"),
        "Boundary condition for the face with the smallest z.");
    prm.declare_entry("Zmax BC","vacuum",
        Patterns::Selection("vacuum|isotropic|most normal|reflective"),
        "Boundary condition for the face with the largest z.");
    prm.declare_entry("Xmin BC values","0",Patterns::List(Patterns::Double(0.)),
        "Values of the incoming flux on xmin face.");
    prm.declare_entry("Xmax BC values","0",Patterns::List(Patterns::Double(0.)),
        "Values of the incoming flux on xmax face.");
    prm.declare_entry("Ymin BC values","0",Patterns::List(Patterns::Double(0.)),
        "Values of the incoming flux on ymin face.");
    prm.declare_entry("Ymax BC values","0",Patterns::List(Patterns::Double(0.)),
        "Values of the incoming flux on ymax face.");
    prm.declare_entry("Zmin BC values","0",Patterns::List(Patterns::Double(0.)),
        "Values of the incoming flux on zmin face.");
    prm.declare_entry("Zmax BC values","0",Patterns::List(Patterns::Double(0.)),
        "Values of the incoming flux on zmax face.");
    prm.declare_entry("Number of sources","0",Patterns::Integer(0),
        "Number of sources.");
    prm.declare_entry("Number of groups","1",Patterns::Integer(1),
        "Number of groups.");
    prm.declare_entry("Source intensity","1.",Patterns::List(
          Patterns::Double(0.)),"Intensity of the different sources.");
  }                
  prm.leave_subsection();
}

void Parameters::parse_parameters(ParameterHandler &prm)
{
  dim = prm.get_integer("Dimension");
  fe_order = prm.get_integer("FE order");
  geometry_filename = prm.get("Geometry file");
  n_refinements = prm.get_integer("Number of refinements");
  refinement_factor = prm.get_double("Refinement factor");
  coarsening_factor = prm.get_double("Coarsening factor");
  output_filename = prm.get("Output file");
  prm.enter_subsection("Radiative Transfer parameters");
  {
    std::string input;
    
    xs_filename = prm.get("Cross sections file");
    
    input = prm.get("Solver type");
    if (input.compare("SI")==0)
      solver_type = SI;
    else
    {
      if (input.compare("BICGSTAB")==0)
        solver_type = BICGSTAB;
      else
        solver_type = GMRES;
    }
    
    max_out_it = prm.get_integer("Max outer iterations");
    
    max_in_it = prm.get_integer("Max inner iterations");
    
    outer_tol = prm.get_double("Tolerance outer solver");
    
    inner_tol = prm.get_double("Tolerance inner solver");
    
    sn_order = prm.get_integer("Sn order");
    
    input = prm.get("Quadrature type");
    if (input.compare("LS")==0)
      quad_type = LS_QUAD;
    else
      quad_type = GLC_QUAD;
    
    input = prm.get("Sum weight");
    if (input.compare("1")==0)
      weight_sum = 1.;
    else
    {
      if (input.compare("2PI")==0)
        weight_sum = 2.*M_PI;
      else
        weight_sum = 4.*M_PI;
    }

    galerkin = prm.get_bool("Galerkin");

    bc_type.resize(2.*dim);
    input = prm.get("Xmin BC");
    if (input.compare("vacuum")==0)
      bc_type[0] = VACUUM;
    else
    {
      if (input.compare("reflective")==0)
        bc_type[0] = REFLECTIVE;
      else 
      {
        if (input.compare("isotropic")==0)
          bc_type[0] = ISOTROPIC;
        else
          bc_type[0] = MOST_NORMAL;
      }
    }
    input = prm.get("Xmax BC");
    if (input.compare("vacuum")==0)
      bc_type[1] = VACUUM;
    else
    {
      if (input.compare("reflective")==0)
        bc_type[1] = REFLECTIVE;
      else 
      {
        if (input.compare("isotropic")==0)
          bc_type[1] = ISOTROPIC;
        else
          bc_type[1] = MOST_NORMAL;
      }
    }
    input = prm.get("Ymin BC");
    if (input.compare("vacuum")==0)
      bc_type[2] = VACUUM;
    else
    {
      if (input.compare("reflective")==0)
        bc_type[2] = REFLECTIVE;
      else 
      {
        if (input.compare("isotropic")==0)
          bc_type[2] = ISOTROPIC;
        else
          bc_type[2] = MOST_NORMAL;
      }
    }
    input = prm.get("Ymax BC");
    if (input.compare("vacuum")==0)
      bc_type[3] = VACUUM;
    else
    {
      if (input.compare("reflective")==0)
        bc_type[3] = REFLECTIVE;
      else 
      {
        if (input.compare("isotropic")==0)
          bc_type[3] = ISOTROPIC;
        else
          bc_type[3] = MOST_NORMAL;
      }
    }
    if (dim==3)
    {
      input = prm.get("Zmin BC");
      if (input.compare("vacuum")==0)
        bc_type[4] = VACUUM;
      else
      {
        if (input.compare("reflective")==0)
          bc_type[4] = REFLECTIVE;
        else 
        {
          if (input.compare("isotropic")==0)
            bc_type[4] = ISOTROPIC;
          else
            bc_type[4] = MOST_NORMAL;
        }
      }
      input = prm.get("Zmax BC");
      if (input.compare("vacuum")==0)
        bc_type[5] = VACUUM;
      else
      {
        if (input.compare("reflective")==0)
          bc_type[5] = REFLECTIVE;
        else 
        {
          if (input.compare("isotropic")==0)
            bc_type[5] = ISOTROPIC;
          else
            bc_type[5] = MOST_NORMAL;
        }
      }
    }

    n_groups = prm.get_integer("Number of groups");

    inc_flux.resize(2*dim,std::vector<double>(n_groups,0.));
    if ((bc_type[0]==ISOTROPIC) || (bc_type[0]==MOST_NORMAL))
    {
      input = prm.get("Xmin BC values");
      inc_flux[0] = get_list_double(input,n_groups);
    }
    if ((bc_type[1]==ISOTROPIC) || (bc_type[1]==MOST_NORMAL))
    {
      input = prm.get("Xmax BC values");
      inc_flux[1] = get_list_double(input,n_groups);
    }
    if ((bc_type[2]==ISOTROPIC) || (bc_type[2]==MOST_NORMAL))
    {
      input = prm.get("Ymin BC values");
      inc_flux[2] = get_list_double(input,n_groups);
    }
    if ((bc_type[3]==ISOTROPIC) || (bc_type[3]==MOST_NORMAL))
    {
      input = prm.get("Ymax BC values");
      inc_flux[3] = get_list_double(input,n_groups);
    }
    if (dim==3)
    {
      if ((bc_type[4]==ISOTROPIC) || (bc_type[4]==MOST_NORMAL))
      {
        input = prm.get("Zmin BC values");
        inc_flux[4] = get_list_double(input,n_groups);
      }
      if ((bc_type[5]==ISOTROPIC) || (bc_type[5]==MOST_NORMAL))
      {
        input = prm.get("Zmax BC values");
        inc_flux[5] = get_list_double(input,n_groups);
      }
    }
  
    n_src = prm.get_integer("Number of sources");
    
    src.resize(n_src,std::vector<double>(n_groups,0.));
    input = prm.get("Source intensity");
    std::vector<double> values(get_list_double(input,n_src*n_groups));
    for (unsigned int i=0; i<n_src; ++i)
      for (unsigned int g=0; g<n_groups; ++g)
        src[i][g] = values[i*n_groups+g];
  }
  prm.leave_subsection();
}

std::vector<double> Parameters::get_list_double(std::string &input,
    unsigned int n_elements)
{
  std::vector<double> values(n_elements,0.);

  // Replace , by blank
  for (unsigned int i=0; i<input.size(); ++i)
    if (input[i]==',')
      input.replace(i,1," ");

  char* end_ptr_1;
  for (unsigned int i=0; i<n_elements; ++i)
  {
    if (i==0)
      values[i] = std::strtod(input.c_str(),&end_ptr_1);
    else
    {
      char* end_ptr_2;
      values[i] = std::strtod(end_ptr_1,&end_ptr_2);
      end_ptr_1 = end_ptr_2;
    }
  }

  return values;
}
