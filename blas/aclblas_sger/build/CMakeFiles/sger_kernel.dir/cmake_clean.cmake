file(REMOVE_RECURSE
  "lib/libsger_kernel.a"
  "lib/libsger_kernel.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang CXX)
  include(CMakeFiles/sger_kernel.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
