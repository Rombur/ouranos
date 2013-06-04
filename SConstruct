# Check is the debug or the release version must be compile
debug = ARGUMENTS.get('debug',1)

# Path to deal.II
deal_II_path = ARGUMENTS.get('deal_II_path','/home/bruno/Documents/deal.II/branch_bigger_global_dof_indices_4/deal.II/installed')

# Path to boost
boost_path = ARGUMENTS.get('boost_path','/usr/include')

SConscript('src/SConscript',exports=['debug','deal_II_path','boost_path'])
