/* Copyright (c) 2013, Bruno Turcksin.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file
 * license.txt for the text and further information on this license.
 */

#include "Geometry.hh"

template<int dim>
Geometry<dim>::Geometry(std::string &geometry_filename) :
  triangulation(MPI_COMM_WORLD),
  dof_handler(triangulation)
{
  d_vector bot_left,up_right;
  ParameterHandler prm;

  // Declare the parameters 
  declare_parameters(prm);

  // Read the file
  std::ifstream geometry_file(geometry_filename.c_str());
  AssertThrow(geometry_file,ExcMessage("Geometry file not found."));
  const bool success = prm.read_input(geometry_file);
  AssertThrow(success,ExcMessage("Invalid geometry file."));
  geometry_file.close();

  // Parse the parameters
  parse_parameters(prm,bot_left,up_right);

  // Generate the grid   
  if (dim==2)
  {
    Point<dim,double> bottom_left(bot_left[0],bot_left[1]);
    Point<dim,double> upper_right(up_right[0],up_right[1]);
    GridGenerator::subdivided_hyper_rectangle(triangulation,n_subdivisions,
        bottom_left,upper_right,true);
  }
  else
  {
    Point<dim,double> bottom_left(bot_left[0],bot_left[1],bot_left[2]);
    Point<dim,double> upper_right(up_right[0],up_right[1],up_right[2]);
    GridGenerator::subdivided_hyper_rectangle(triangulation,n_subdivisions,
        bottom_left,upper_right,true);
  }
}
  
template<int dim>
void Geometry<dim>::declare_parameters(ParameterHandler &prm)
{  
  prm.declare_entry("Bottom left point","0.,0.",Patterns::List(Patterns::Double()),
      "Coordinate of bottom left point.");
  prm.declare_entry("Upper right point","0.,0.",Patterns::List(Patterns::Double()),
      "Coordinate of upper right point.");
  prm.declare_entry("Number of subdivisions","1,1",
      Patterns::List(Patterns::Integer(1)),"Number of subdivisions in x,y,z.");
  prm.declare_entry("Material IDs","0",Patterns::List(Patterns::Integer(0)),
      "Material IDs.");
  prm.declare_entry("Source IDs","0",Patterns::List(Patterns::Integer(0)),
      "Source IDs.");
}

template<int dim>
void Geometry<dim>::parse_parameters(ParameterHandler &prm,d_vector &bot_left,
    d_vector &up_right)
{
  std::string input;

  input = prm.get("Bottom left point");
  bot_left = get_list_double(input,dim);

  input = prm.get("Upper right point");
  up_right = get_list_double(input,dim);

  input = prm.get("Number of subdivisions");
  n_subdivisions = get_list_uint(input,dim);

  input = prm.get("Material IDs");
  unsigned int n_sub(n_subdivisions[0]);
  for (unsigned int i=1; i<dim; ++i)
    n_sub *= n_subdivisions[i];
  material_ids = get_list_uint(input,n_sub);

  input = prm.get("Source IDs");
  source_ids = get_list_uint(input,n_sub);
}

template<int dim>
ui_vector Geometry<dim>::get_list_uint(std::string &input,unsigned int n_elements)
{
  ui_vector values(n_elements,0.);

  // Replace , by blank
  for (unsigned int i=0; i<input.size(); ++i)
    if (input[i]==',')
      input.replace(i,1," ");

  char* end_ptr_1;
  for (unsigned int i=0; i<n_elements; ++i)
  {
    if (i==0)
      values[i] = std::strtol(input.c_str(),&end_ptr_1,0);
    else
    {
      char* end_ptr_2;
      values[i] = std::strtol(end_ptr_1,&end_ptr_2,0);
      end_ptr_1 = end_ptr_2;
    }
  }

  return values;
}

template<int dim>
d_vector Geometry<dim>::get_list_double(std::string &input,unsigned int n_elements)
{
  d_vector values(n_elements,0.);

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

//*****Explicit instantiations*****//
template class Geometry<2>;
template class Geometry<3>;
