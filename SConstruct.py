import os
import sys

env = SConscript("godot-cpp/SConstruct")

env.Append(CPPPATH=["src/"])
sources = Glob("src/*.cpp") # <--- Captures both NEATAgent.cpp and NetworkAgent.cpp

if env["platform"] == "macos":
    # Creates bin/NEAT.macos.debug.framework/...
    library = env.SharedLibrary(
        "demoproject/bin/NEAT.{}.{}.framework/NEAT.{}.{}".format(
            env["platform"], env["target"], env["platform"], env["target"]
        ),
        source=sources,
    )
    env.Append(CPPDEFINES=["ACCELERATE_NEW_LAPACK"])
    env.Append(LINKFLAGS=["-framework", "Accelerate"])
else:
    library = env.SharedLibrary(
        "demoproject/bin/NEAT{}{}".format(env["suffix"], env["SHLIBSUFFIX"]),
        source=sources,
    )

Default(library)