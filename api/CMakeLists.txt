# api/ -- Tutti runtime public API
#
# v0.1 status: headers only. The Runtime implementation
# (api/src/runtime.cpp) lands when the dispatch logic is wired up
# against the concrete IDeviceRegistry / IMemorySubsystem / backend
# implementations.
#
# This file declares the api as an INTERFACE library so other
# subsystems can link against the headers without pulling in code
# that doesn't exist yet.

add_library(tutti_api INTERFACE)

target_include_directories(tutti_api INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/..        # for sibling subsystem includes
)

# When api/src/ lands, swap INTERFACE -> STATIC and add sources here.
