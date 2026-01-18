import os
import sys

env = SConscript("godot-cpp/SConstruct")

if env["target"] == "template_debug":
    env.Append(CCFLAGS=["-O0", "-g"])

env.Append(CPPPATH=["src/"])
sources = Glob("src/*.cpp")

if env["platform"] == "macos":
    library = env.SharedLibrary(
        "addons/NEAT/bin/NEAT.{}.{}.framework/NEAT.{}.{}".format(
            env["platform"], env["target"], env["platform"], env["target"]
        ),
        source=sources,
    )
else:
    library = env.SharedLibrary(
        "addons/NEAT/bin/libNEAT{}{}".format(env["suffix"], env["SHLIBSUFFIX"]),
        source=sources,
    )

Default(library)