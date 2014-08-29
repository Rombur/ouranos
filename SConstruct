def read_deal_II_options(filename,debug) :
  """Read the Make.global_options of deal.II and extract the important data
  for SCons."""
  file = open(filename,'r')
  debug_ld_flags_next_line = False
  debug_libs_next_line = False
  release_ld_flags_next_line = False
  release_libs_next_line = False
  first_include = True
  for line in file :
    if 'ifeq ($(debug-mode), on)' in line :
      debug_ld_flags_next_line = True
      continue
    if debug_ld_flags_next_line==True :
      if debug==1 :
        ld_flags = line.split()[2:]
        debug_ld_flags_next_line = False
        debug_libs_next_line = True
        continue
    if debug_libs_next_line==True :
      libs = line.split()[3:]
      debug_libs_next_line = False
    if 'else' in line and debug_ld_flags_next_line==True :
      release_ld_flags_next_line = True
      debug_ld_flags_next_line = False
      continue
    if release_ld_flags_next_line==True :
      ld_flags = line.split()[2:]
      release_ld_flags_next_line = False
      release_libs_next_line = True
      continue
    if release_libs_next_line==True :
      libs = line.split()[3:]
      release_libs_next_line = False
    if 'INCLUDE' in line and first_include==True :
      include = line.split()[5:]
      first_include = False
    if 'CXXFLAGS.g' in line and debug==1 :
      cxx_flags = line.split()[2:]
      cxx_flags = ['-std=c++11',cxx_flags[:-1]]
    if 'CXXFLAGS.o' in line and debug==0 :
      cxx_flags = line.split()[2:]
      cxx_flags = ['-std=c++11',cxx_flags[:-1]]

  return ld_flags,libs,include,cxx_flags
      

# Check if the debug or the release version must be compile
debug = int(ARGUMENTS.get('debug',1))

# Path to deal.II
deal_II_path = ARGUMENTS.get('deal_II_path','/w/turcksin/dealii/install')

# Read Make.global_options of deal.II 
ld_flags,partial_libs,partial_include,cxx_flags = read_deal_II_options(deal_II_path+'/common/Make.global_options',debug)

# Change the format of libs to be compatible with SCons
libs = []
libpath = [deal_II_path+'/lib']
for elem in partial_libs :
  if elem=='-Wl,-rpath,$(D)/lib' :
    ld_flags += ['-Wl,-rpath,'+deal_II_path+'/lib']
  else :
    if '-Wl,-rpath,' in elem :
      ld_flags += [elem]
    else :
# The BLAS and ATLAS libraries are a special case because their suffix is
# .so.3gf So the entire path is used and the standard SCons method is
# by-passed.
      if elem[-4:]=='.3gf' :
        libs += [File(elem)]
      else :
        split_elem = elem.split('/')
        libpath_elem = ''
        for i in xrange(1,len(split_elem)-1) :
          libpath_elem += '/'+split_elem[i]
# If the first three characters are lib, we strip lib.
        if 'lib' in split_elem[-1][:3] :
          lib_elem = split_elem[-1][3:]
# If the first two characters are -l, we strip -l.
        elif '-l' in split_elem[-1][:2] :
          lib_elem = split_elem[-1][2:]
        if libpath_elem not in libpath :
          libpath += [libpath_elem]
# Strip .so if necessary
        if '.so' in lib_elem[:-3]:
          libs += [lib_elem[:-3]]          
        else :
          libs += [lib_elem]

# Add libdeal_II.g.so or libdeal_II.so to libs
if debug==1 :
  libs = ['deal_II.g'] + libs
else :
  libs = ['deal_II'] + libs

# Strip -I from include elements
include = [deal_II_path+'/include',deal_II_path+'/include/deal.II',deal_II_path+'/include/deal.II/bundled']
for elem in partial_include :
  include += [elem[2:]]

# Create the environment
# CPPFLAGS are for C PreProcessing
# CXXFLAGS are for C++ compiler
# Import the environment of the user
import os
env = Environment(
  ENV=os.environ,
  CXX='mpic++',
  CXXFLAGS=cxx_flags,
  CPPPATH=include,
  LIBPATH=libpath,
  LIBS=libs,
  LINKFLAGS=ld_flags)

SConscript(['src/SConscript','tests/SConscript'],exports=['env'])
