cmake_minimum_required(VERSION 3.18.1)

project("stuffmods")

add_library(stuffmods SHARED
    Menu/Menu.cpp
    Menu/ImGui/imgui.cpp
    Menu/ImGui/imgui_draw.cpp
    Menu/ImGui/imgui_tables.cpp
    Menu/ImGui/imgui_widgets.cpp
    Menu/ImGui/backends/imgui_impl_android.cpp
    Menu/ImGui/backends/imgui_impl_vulkan.cpp
)

add_library(dobby STATIC IMPORTED)
set_target_properties(dobby PROPERTIES
    IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/Dobby/${ANDROID_ABI}/libdobby.a)

target_include_directories(stuffmods PRIVATE
    Menu/ImGui
    Menu/ImGui/backends
    Dobby
)

target_compile_options(stuffmods PRIVATE -O2 -std=c++17)

target_link_libraries(stuffmods PRIVATE dobby log android z vulkan)
