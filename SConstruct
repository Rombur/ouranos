# Check is the debug or the release version must be compile
debug = ARGUMENTS.get('debug',1)

# Path to deal.II
deal_II_path = ARGUMENTS.get('deal_II_path','/home/bruno/Documents/deal.ii/trunk/deal.II/installed')

# Path to boost
boost_path = ARGUMENTS.get('boost_path','/usr/include')

# Path to trilinos
trilinos_path = ARGUMENTS.get('trilinos_path','/home/bruno/Documents/vendors/trilinos/bin')

SConscript('tests/SConscript',exports=['deal_II_path','boost_path','trilinos_path'])
SConscript('src/SConscript',exports=['debug','deal_II_path','boost_path','trilinos_path'])
