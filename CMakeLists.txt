cmake_minimum_required(VERSION 3.18)
project(Genesis4_CUDA LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Developer options
set(CMAKE_VERBOSE_MAKEFILE ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
add_compile_options($<$<CONFIG:Debug>:-g>)

# OpenMPI: disable C++ interface
add_compile_definitions(OMPI_SKIP_MPICXX)

# ──────────────────────────────────────────────────────
# External source setup
# ──────────────────────────────────────────────────────
option(GENESIS_USE_CUDA "Enable native CUDA acceleration" ON)
option(G4_AUTO_FETCH_DEPS "Automatically clone upstream Genesis4 if missing" ON)
option(G4_RENAME_OVERLAPPED_HEADERS "Rename upstream Genesis4 headers shadowed by this port to *.orig" ON)


set(G4_UPSTREAM_REPOSITORY "https://github.com/svenreiche/Genesis-1.3-Version4.git" CACHE STRING "Upstream Genesis4 git repository")
set(G4_UPSTREAM_GIT_REF "v4.6.14" CACHE STRING "Genesis4 branch/tag to clone")
set(G4_UPSTREAM_SOURCE_DIR "${CMAKE_SOURCE_DIR}/Genesis-1.3-Version4" CACHE PATH "Upstream Genesis4 source directory")

find_package(Git QUIET)

function(git_clone_if_missing name repo ref dst required_file)
    if (EXISTS "${dst}/${required_file}")
        message(STATUS "${name}: using existing source at ${dst}")
        return()
    endif()

    if (EXISTS "${dst}")
        message(FATAL_ERROR
            "${name}: ${dst} exists but ${required_file} was not found. "
            "Remove the incomplete directory or set the corresponding *_SOURCE_DIR cache variable.")
    endif()

    if (NOT G4_AUTO_FETCH_DEPS)
        message(FATAL_ERROR
            "${name}: source not found at ${dst}. "
            "Either enable G4_AUTO_FETCH_DEPS or provide the source directory manually.")
    endif()

    if (NOT GIT_FOUND)
        message(FATAL_ERROR "${name}: git executable not found; cannot auto-fetch ${repo}")
    endif()

    get_filename_component(_dst_parent "${dst}" DIRECTORY)
    file(MAKE_DIRECTORY "${_dst_parent}")

    set(_clone_cmd "${GIT_EXECUTABLE}" clone --depth 1)
    if (NOT "${ref}" STREQUAL "")
        list(APPEND _clone_cmd --branch "${ref}")
    endif()
    list(APPEND _clone_cmd "${repo}" "${dst}")

    message(STATUS "${name}: cloning ${repo} into ${dst}")
    execute_process(
        COMMAND ${_clone_cmd}
        RESULT_VARIABLE _clone_result
        OUTPUT_VARIABLE _clone_output
        ERROR_VARIABLE _clone_error
    )

    if (NOT _clone_result EQUAL 0)
        message(FATAL_ERROR
            "${name}: git clone failed with exit code ${_clone_result}\n"
            "Command: ${_clone_cmd}\n"
            "Output: ${_clone_output}\n"
            "Error: ${_clone_error}")
    endif()

    if (NOT EXISTS "${dst}/${required_file}")
        message(FATAL_ERROR
            "${name}: clone completed, but ${required_file} was not found in ${dst}")
    endif()
endfunction()

function(g4_restore_shadowed_headers overlay_include_dir upstream_include_dir)
    if (NOT IS_DIRECTORY "${upstream_include_dir}")
        message(FATAL_ERROR "Genesis4 include directory not found: ${upstream_include_dir}")
    endif()

    # If a header was preserved by an older port overlay but that overlay was
    # later removed, restore the upstream header so this port can fall back to
    # the unmodified Genesis4 implementation.
    file(GLOB _preserved_headers CONFIGURE_DEPENDS
        "${upstream_include_dir}/*.orig"
    )
    foreach(_orig_header IN LISTS _preserved_headers)
        get_filename_component(_orig_name "${_orig_header}" NAME)
        string(REGEX REPLACE "\.orig$" "" _header_name "${_orig_name}")
        set(_overlay_header "${overlay_include_dir}/${_header_name}")
        set(_upstream_header "${upstream_include_dir}/${_header_name}")
        file(RENAME "${_orig_header}" "${_upstream_header}")
        message(STATUS "Genesis4 header restored: include/${_header_name}.orig -> include/${_header_name}")
    endforeach()
endfunction()

function(g4_preserve_shadowed_headers overlay_include_dir upstream_include_dir)
    if (NOT G4_RENAME_OVERLAPPED_HEADERS)
        return()
    endif()

    if (NOT IS_DIRECTORY "${overlay_include_dir}")
        return()
    endif()

    if (NOT IS_DIRECTORY "${upstream_include_dir}")
        message(FATAL_ERROR "Genesis4 include directory not found: ${upstream_include_dir}")
    endif()

    # If a header was preserved by an older port overlay but that overlay was
    # later removed, restore the upstream header so this port can fall back to
    # the unmodified Genesis4 implementation.
    file(GLOB _preserved_headers CONFIGURE_DEPENDS
        "${upstream_include_dir}/*.orig"
    )
    foreach(_orig_header IN LISTS _preserved_headers)
        get_filename_component(_orig_name "${_orig_header}" NAME)
        string(REGEX REPLACE "\.orig$" "" _header_name "${_orig_name}")
        set(_overlay_header "${overlay_include_dir}/${_header_name}")
        set(_upstream_header "${upstream_include_dir}/${_header_name}")
        if ((NOT EXISTS "${_overlay_header}") AND (NOT EXISTS "${_upstream_header}"))
            file(RENAME "${_orig_header}" "${_upstream_header}")
            message(STATUS "Genesis4 header restored: include/${_header_name}.orig -> include/${_header_name}")
        endif()
    endforeach()

    file(GLOB _overlay_headers CONFIGURE_DEPENDS
        "${overlay_include_dir}/*.h"
        "${overlay_include_dir}/*.hh"
        "${overlay_include_dir}/*.hpp"
        "${overlay_include_dir}/*.hxx"
    )

    foreach(_overlay_header IN LISTS _overlay_headers)
        get_filename_component(_header_name "${_overlay_header}" NAME)
        set(_upstream_header "${upstream_include_dir}/${_header_name}")
        set(_orig_header "${_upstream_header}.orig")

        if (EXISTS "${_upstream_header}")
            if (EXISTS "${_orig_header}")
                message(WARNING
                    "Genesis4 header ${_upstream_header} is shadowed, but ${_orig_header} already exists. "
                    "Leaving both files untouched to avoid overwriting the preserved original.")
            else()
                file(RENAME "${_upstream_header}" "${_orig_header}")
                message(STATUS "Genesis4 header preserved: include/${_header_name} -> include/${_header_name}.orig")
            endif()
        elseif (EXISTS "${_orig_header}")
            message(STATUS "Genesis4 header already preserved: include/${_header_name}.orig")
        endif()
    endforeach()
endfunction()

git_clone_if_missing("Genesis4" "${G4_UPSTREAM_REPOSITORY}" "${G4_UPSTREAM_GIT_REF}" "${G4_UPSTREAM_SOURCE_DIR}" "include/version.h.in")

# ──────────────────────────────────────────────────────
# Genesis4 source paths
# ──────────────────────────────────────────────────────
set(G4 ${G4_UPSTREAM_SOURCE_DIR})

# ──────────────────────────────────────────────────────
# Find dependencies
# ──────────────────────────────────────────────────────
find_package(MPI REQUIRED)

SET(HDF5_PREFER_PARALLEL TRUE)
find_package(HDF5 REQUIRED)
if (HDF5_IS_PARALLEL)
    message(STATUS "Parallel HDF5 supported")
else()
    message(STATUS "WARNING: No parallel HDF5 support")
endif()

# ──────────────────────────────────────────────────────
# Generate version.h
# ──────────────────────────────────────────────────────
set(_GENESIS_VERSION_MAJOR 4)
set(_GENESIS_VERSION_MINOR 6)
set(_GENESIS_VERSION_REV 2)
set(_GENESIS_VERSION_BETA false)

execute_process(
    COMMAND git -C ${G4} describe --tags --always
    OUTPUT_VARIABLE GIT_HASH
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)
if(NOT GIT_HASH)
    set(GIT_HASH "unknown")
endif()

execute_process(
    COMMAND whoami
    OUTPUT_VARIABLE _user_name
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
string(TIMESTAMP _config_time "%Y-%m-%d %H:%M:%S")

configure_file(
    ${G4}/include/version.h.in
    ${CMAKE_BINARY_DIR}/include/version.h
    @ONLY
)

# ──────────────────────────────────────────────────────
# Files reused directly from Genesis4 (no modification)
# ──────────────────────────────────────────────────────
set(G4_UNCHANGED_SRC
    # Util
    ${G4}/src/Util/BesselJ.cpp
    ${G4}/src/Util/GaussHermite.cpp
    ${G4}/src/Util/Hammerslay.cpp
    ${G4}/src/Util/Inverfc.cpp
    ${G4}/src/Util/RandomU.cpp
    ${G4}/src/Util/SimpleHandshake.cpp
    ${G4}/src/Util/Sorting.cpp
    ${G4}/src/Util/StringProcessing.cpp
    ${G4}/src/Util/SemaFile.cpp
    ${G4}/src/Util/FFTObj.cpp
    # Lattice
    ${G4}/src/Lattice/AlterLattice.cpp
    ${G4}/src/Lattice/Lattice.cpp
    ${G4}/src/Lattice/LatticeElements.cpp
    ${G4}/src/Lattice/LatticeParser.cpp
    ${G4}/src/Lattice/Optics.cpp
    # Core (unchanged)
    ${G4}/src/Core/Undulator.cpp
    ${G4}/src/Core/Incoherent.cpp
    ${G4}/src/Core/Collective.cpp
    ${G4}/src/Core/Control.cpp
    ${G4}/src/Core/Diagnostic.cpp
    ${G4}/src/Core/DiagnosticUser.cpp
    ${G4}/src/Core/BeamSolver.cpp
    ${G4}/src/Core/TrackBeam.cpp
    ${G4}/src/Core/FieldSolverADI.cpp
    ${G4}/src/Core/FieldSolverFFT.cpp
    ${G4}/src/Core/EFieldSolver.cpp
    # IO
    ${G4}/src/IO/HDF5base.cpp
    ${G4}/src/IO/Output.cpp
    ${G4}/src/IO/readBeamHDF5.cpp
    ${G4}/src/IO/readFieldHDF5.cpp
    ${G4}/src/IO/SDDSBeam.cpp
    ${G4}/src/IO/writeBeamHDF5.cpp
    ${G4}/src/IO/writeFieldHDF5.cpp
    ${G4}/src/IO/writeWakeHDF5.cpp
    ${G4}/src/IO/readMapHDF5.cpp
    # Loading
    ${G4}/src/Loading/ImportBeam.cpp
    ${G4}/src/Loading/ImportField.cpp
    ${G4}/src/Loading/LoadBeam.cpp
    ${G4}/src/Loading/LoadField.cpp
    ${G4}/src/Loading/GenProfile.cpp
    ${G4}/src/Loading/QuietLoading.cpp
    ${G4}/src/Loading/Series.cpp
    ${G4}/src/Loading/ShotNoise.cpp
    ${G4}/src/Loading/SeriesManager.cpp
    ${G4}/src/Loading/SeriesParser.cpp
    ${G4}/src/Loading/Sequence.cpp
    ${G4}/src/Loading/ImportTransformation.cpp
    # Main
    ${G4}/src/Main/AlterSetup.cpp
    ${G4}/src/Main/Dump.cpp
    ${G4}/src/Main/AlterBeam.cpp
    ${G4}/src/Main/EField.cpp
    ${G4}/src/Main/FieldManipulator.cpp
    ${G4}/src/Main/GenMain.cpp
    ${G4}/src/Main/Parser.cpp
    ${G4}/src/Main/Setup.cpp
    ${G4}/src/Main/SponRad.cpp
    ${G4}/src/Main/GenTime.cpp
    # ${G4}/src/Main/Track.cpp
    ${G4}/src/Main/Wake.cpp
)

# ──────────────────────────────────────────────────────
# Build configuration: native CUDA vs plain CPU
# ──────────────────────────────────────────────────────
if(GENESIS_USE_CUDA)
    # ──────────────────────────────────────────────────────
    # CUDA architecture selection
    # ──────────────────────────────────────────────────────
    set(GENESIS_CUDA_ARCHITECTURES "86" CACHE STRING
        "CUDA architectures: 86 for RTX A4000, 75 for Quadro RTX 6000, 89 for RTX 6000 Ada, 90 for H100")

    set(CMAKE_CUDA_ARCHITECTURES "${GENESIS_CUDA_ARCHITECTURES}" CACHE STRING
        "CUDA architectures for Genesis4_CUDA")

    enable_language(CUDA)

    # Native CUDA-ported source files (new implementations).
    set(CUDA_PORTED_CUDA_SRC
        src/Core/TrackBeamCUDA.cpp
        src/Core/BeamSolverCUDA.cpp
        src/Core/FieldSolverADICUDA.cpp
        src/Core/FieldSolverFFTCUDA.cpp
        src/Core/ControlCUDA.cpp
        src/Core/DiagnosticCUDA.cpp
    )

    # Port glue.  CPU solvers are compiled directly from upstream Genesis4 and are
    # selected when &track/use_cuda is false.
    set(CUDA_PORTED_CXX_SRC
        src/Core/Gencore.cpp
        src/Core/Beam.cpp
        src/Core/Field.cpp
        src/Main/Track.cpp
    )

    g4_preserve_shadowed_headers("${CMAKE_SOURCE_DIR}/include" "${G4}/include")

    find_package(CUDAToolkit REQUIRED)

    add_library(genesis4_cuda STATIC
        ${G4_UNCHANGED_SRC}
        ${CUDA_PORTED_CUDA_SRC}
        ${CUDA_PORTED_CXX_SRC}
    )
    set_target_properties(genesis4_cuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "${GENESIS_CUDA_ARCHITECTURES}"
    )

    set_source_files_properties(${CUDA_PORTED_CUDA_SRC} PROPERTIES LANGUAGE CUDA)
    target_compile_options(genesis4_cuda PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda --expt-relaxed-constexpr>
    )

    target_include_directories(genesis4_cuda
        PUBLIC
        ${CMAKE_SOURCE_DIR}/include
        ${G4}/include
        ${CMAKE_BINARY_DIR}/include
    )

    target_link_libraries(genesis4_cuda
        PUBLIC
        MPI::MPI_C
        CUDA::cudart
        CUDA::cufft
    )
    target_compile_definitions(genesis4_cuda PUBLIC GENESIS_USE_CUDA=1)

else()
    set(G4_CHANGED_SRC
        ${G4}/src/Main/Track.cpp
        ${G4}/src/Core/Gencore.cpp
        ${G4}/src/Core/Beam.cpp
        ${G4}/src/Core/Field.cpp
    )
    
    # Plain CPU build: no CUDA, upstream Genesis4 sources only
    g4_restore_shadowed_headers("${CMAKE_SOURCE_DIR}/include" "${G4}/include")

    add_library(genesis4_cuda STATIC
        ${G4_UNCHANGED_SRC}
        ${G4_CHANGED_SRC}
    )

    target_include_directories(genesis4_cuda
        PUBLIC
        ${G4}/include
        ${CMAKE_BINARY_DIR}/include
    )

    target_link_libraries(genesis4_cuda
        PUBLIC
        MPI::MPI_C
    )
endif()

# ──────────────────────────────────────────────────────
# Optional dependencies
# ──────────────────────────────────────────────────────
find_package(PkgConfig REQUIRED)
list(APPEND CMAKE_PREFIX_PATH $ENV{FFTW_ROOT})
set(ENV{PKG_CONFIG_PATH} "$ENV{PKG_CONFIG_PATH}:$ENV{FFTW_ROOT}/lib/pkgconfig/")
pkg_check_modules(FFTW QUIET fftw3 IMPORTED_TARGET)
if (TARGET PkgConfig::FFTW)
    message(STATUS "FFTW found")
    target_link_libraries(genesis4_cuda PUBLIC PkgConfig::FFTW)
else()
        message(STATUS "FFTW not found")
        message(STATUS "Excluding FFTW specific source code")
endif()

# add optional macro definitions
if (TARGET PkgConfig::FFTW)
    message(STATUS "Defining the macro FFTW in the source code")
    target_compile_definitions(genesis4_cuda PRIVATE FFTW=1)
endif()

if (TARGET HDF5::HDF5)
    target_link_libraries(genesis4_cuda PUBLIC HDF5::HDF5)
else()
    target_include_directories(genesis4_cuda PUBLIC ${HDF5_C_INCLUDE_DIRS})
    target_link_libraries(genesis4_cuda PUBLIC ${HDF5_C_LIBRARIES})
endif()

# ──────────────────────────────────────────────────────
# Build executable
# ──────────────────────────────────────────────────────
add_executable(genesis4 ${G4}/src/Main/mainwrap.cpp)
target_link_libraries(genesis4 genesis4_cuda)