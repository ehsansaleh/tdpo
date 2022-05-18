# the compiler: gcc for C program, define as g++ for C++
CC = module load gcc; g++

# compiler flags:
#  -g    adds debugging information to the executable file
#  -Wall turns on most, but not all, compiler warnings

CFLAGS  = -O2 --std=c++11 -mavx -pthread -Wl,-rpath,'$$ORIGIN:./lib' -fPIC -g -Wall -faligned-new

# define any directories containing header files other than /usr/include
INCLUDES = -I./include

# define library paths in addition to /usr/lib
#   if I wanted to include libraries not in /usr/lib I'd specify
#   their path using -Lpath, something like:
LFLAGS = -L./lib

# This can be used to add the lib directory to the linker's search path
EXPORT_LDLIB = LD_LIBRARY_PATH=$$LD_LIBRARY_PATH:$(shell realpath ./lib)

# define any libraries to link into executable:
#   if I want to link in libraries (libx.so or libx.a) I use the -llibname
#   option, something like (this will link in libmylib.so and libm.so:
SIMIFLIBS = -lmujoco200nogl
MLPLIBS =  -lsleef -lblas

# Define any pre-processor directives in this variable:
XMLPATH = $(shell realpath ./src/leg.xml)
SIMIFDEFS = -D xml_type=content_type -D xml_file="$(XMLPATH)"

SHELL:=/bin/bash

# the build target executable:
SIMIFTARGET = SimIF
MLPTARGET = MlpIF
ROLLTARGET = Rollout

all: roll_so

################################################
################# SimInterface #################
################################################

simif: clean_simif xml2hpp $(XMLHPPPATH) ./src/$(SIMIFTARGET).cpp
	$(CC) $(CFLAGS) $(INCLUDES) $(SIMIFDEFS) -D __MAINPROG__=SimIF_CPP -o ./bin/$(SIMIFTARGET) \
	./src/$(SIMIFTARGET).cpp $(LFLAGS) $(SIMIFLIBS)
	./bin/$(SIMIFTARGET)

XMLFILENAME := $(shell basename $(abspath $(XMLPATH)))
XMLBYTESEXACT = $(shell du -sb $(XMLPATH) | cut -f1)
XMLBYTES = $$(( $(XMLBYTESEXACT) + 10 ))
XMLHPPPATH = ./include/mj_xml.hpp

xml2hpp: $(XMLPATH)
	@echo "Input XML File -> " $(XMLPATH)
	@echo "#pragma once" > $(XMLHPPPATH)
	@echo "const int xml_bytes = "$(XMLBYTES)"; " >> $(XMLHPPPATH)
	@echo -n "const char xml_content["$(XMLBYTES)"] = R\"\"\"(" >> $(XMLHPPPATH)
	@cat $(XMLPATH) >> $(XMLHPPPATH)
	@echo ")\"\"\";" >> $(XMLHPPPATH)

################################################
#################### MLP #######################
################################################

mlp: clean_mlp ./src/$(MLPTARGET).cpp
	$(CC) $(CFLAGS) $(INCLUDES) -D __MAINPROG__=MlpIF_CPP -o ./bin/$(MLPTARGET) \
	./src/$(MLPTARGET).cpp $(LFLAGS) $(MLPLIBS)
	$(EXPORT_LDLIB) ./bin/$(MLPTARGET)

################################################
################### Rollout ####################
################################################

roll: clean_roll ./src/$(ROLLTARGET).cpp ./src/$(MLPTARGET).cpp ./src/$(SIMIFTARGET).cpp
	$(CC) $(CFLAGS) $(INCLUDES) $(SIMIFDEFS) -D __MAINPROG__=Rollout_CPP -o ./bin/$(ROLLTARGET) \
	./src/$(ROLLTARGET).cpp ./src/$(SIMIFTARGET).cpp ./src/$(MLPTARGET).cpp $(LFLAGS) $(MLPLIBS) $(SIMIFLIBS)
	$(EXPORT_LDLIB) ./bin/$(ROLLTARGET)

roll_so: clean_roll xml2hpp $(XMLHPPPATH) ./src/$(ROLLTARGET).cpp ./src/$(MLPTARGET).cpp ./src/$(SIMIFTARGET).cpp
	$(CC) $(CFLAGS) -shared $(INCLUDES) \
	$(SIMIFDEFS) -D __MAINPROG__=Shared_Obj \
	-o ./bin/lib$(ROLLTARGET).so ./src/$(ROLLTARGET).cpp ./src/$(SIMIFTARGET).cpp ./src/$(MLPTARGET).cpp \
	$(LFLAGS) $(MLPLIBS) $(SIMIFLIBS)

################################################
################# Clearning ####################
################################################

clean_simif:
	$(RM) ./bin/$(SIMIFTARGET)
clean_mlp:
	$(RM) ./bin/$(MLPTARGET)
clean_roll:
	$(RM) ./bin/$(ROLLTARGET)
	$(RM) ./bin/lib$(ROLLTARGET).so

clean: clean_simif clean_mlp clean_roll
	$(RM) ./bin/core.*
