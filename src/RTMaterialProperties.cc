/* Copyright (c) 2013, Bruno Turcksin.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file
 * license.txt for the text and further information on this license.
 */

#include "RTMaterialProperties.hh"

RTMaterialProperties::RTMaterialProperties(
    std::string &material_properties_filename,unsigned int n_materials,
    unsigned int n_groups) :
  n_materials(n_materials),
  n_groups(n_groups)
{
  ParameterHandler prm;

  // Declare the parameters 
  declare_parameters(prm);

  // Read the file
  std::ifstream material_properties_file(material_properties_filename.c_str());
  AssertThrow(material_properties_file,ExcMessage(
        "Radiative transfer material properties file not found."));
  const bool success = prm.read_input(material_properties_file);
  AssertThrow(success,ExcMessage(
        "Invalid radiative transfer material properties file."));
  material_properties_file.close();

  // Parse the parameters
  parse_parameters(prm);
}

void RTMaterialProperties::declare_parameters(ParameterHandler &prm)
{
  prm.declare_entry("L_max","0",Patterns::Integer(0),"L_max");
  prm.declare_entry("Sigma_t","1.",Patterns::List(Patterns::Double(0.)),
      "Total cross sections.");
  prm.declare_entry("Sigma_s","0.",Patterns::List(Patterns::Double(0.)),
      "Scattering cross sections.");
}

void RTMaterialProperties::parse_parameters(ParameterHandler &prm)
{
  std::string input;

  L_max = prm.get_integer("L_max");

  input = prm.get("Sigma_t");
  sigma_t = get_list_double(input,n_materials*n_groups);

  input = prm.get("Sigma_s");
  sigma_s = get_list_double(input,n_materials*n_groups*n_groups*(L_max+1));
}


std::vector<double> RTMaterialProperties::get_list_double(std::string &input,
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
