# Check is the debug or the release version must be compile
debug = ARGUMENTS.get('debug',1)

# Path for deal.II
deal_II_path = ARGUMENTS.get('deal_II_path','/home/bruno/Documents/deal.II/branch_bigger_global_dof_indices_4/deal.II/installed')

SConscript('src/SConscript',exports=['debug','deal_II_path'])
