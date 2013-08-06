# www.scons.org/doc/1.1.0/HTML/scons-user/x2361.html


# Check if the debug or the release version must be compile
debug = ARGUMENTS.get('debug',1)

# Path to deal.II
deal_II_path = ARGUMENTS.get('deal_II_path','/home/bruno/Documents/deal.ii/modified_deal_ii/deal.II/installed')

# Path to boost
boost_path = ARGUMENTS.get('boost_path','/usr/include')

# Path to trilinos
trilinos_path = ARGUMENTS.get('trilinos_path','/home/bruno/Documents/vendors/trilinos/bin')

# Path to mpi.h
mpi_path = ARGUMENTS.get('mpi_path','/usr/include/mpi')

# Path to p4est
p4est_path = ARGUMENTS.get('p4est_path','/home/bruno/Documents/vendors/p4est-0.3.4/bin')

SConscript('tests/SConscript',exports=['deal_II_path','boost_path','trilinos_path',
  'mpi_path','p4est_path'])
SConscript('src/SConscript',exports=['debug','deal_II_path','boost_path',
  'trilinos_path','mpi_path','p4est_path'])
