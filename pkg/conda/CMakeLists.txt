# Conda package building for NCCL

find_program(SED_EXECUTABLE sed)

# Set package directories
set(TGZPREPDIR "${CMAKE_BINARY_DIR}/conda")
set(PKGDIR "${CMAKE_BINARY_DIR}/pkg/conda")
set(TGZTARGETS ${TGZPREPDIR}/create_tarball.sh)

# Get package architecture
set(PKG_ARCH ${CMAKE_HOST_SYSTEM_PROCESSOR})

# Custom target for conda package preparation
add_custom_target(conda_prep ALL
  # Create conda prep directory
  COMMAND ${CMAKE_COMMAND} -E make_directory ${TGZPREPDIR}
  COMMENT "Preparing Conda package files"
)

# Generate create_tarball.sh from template
add_custom_command(
  TARGET conda_prep
  COMMAND ${SED_EXECUTABLE} -e 's/\\$$\{nccl:Major\}/${NCCL_MAJOR}/g'
                            -e 's/\\$$\{nccl:Minor\}/${NCCL_MINOR}/g'
                            -e 's/\\$$\{nccl:Patch\}/${NCCL_PATCH}/g'
                            -e 's/\\$$\{pkg:Arch\}/${PKG_ARCH}/g'
            ${CMAKE_CURRENT_SOURCE_DIR}/create_tarball.sh.in > ${TGZPREPDIR}/create_tarball.sh
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/create_tarball.sh.in
)

# Build the package
add_custom_target(conda_build
  COMMAND cd ${CMAKE_BINARY_DIR} && bash conda/create_tarball.sh
  COMMAND ${CMAKE_COMMAND} -E make_directory ${PKGDIR}
  COMMAND mv ${CMAKE_SOURCE_DIR}/nccl*.tar.gz ${PKGDIR}
  WORKING_DIRECTORY ${TGZPREPDIR}
  DEPENDS conda_prep nccl_static
  COMMENT "Building Conda package with create_tarball.sh"
)

# Custom target for cleaning conda package
add_custom_target(conda_clean
  COMMAND ${CMAKE_COMMAND} -E remove_directory ${TGZPREPDIR}
  COMMAND ${CMAKE_COMMAND} -E remove_directory ${PKGDIR}
  COMMENT "Cleaning Conda package build files"
)

# Add conda targets to the main project
add_dependencies(conda_build nccl_static)
