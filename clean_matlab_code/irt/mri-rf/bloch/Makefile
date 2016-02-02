# bloch/Makefile
# compile Bloch simulator

# we need a unique name for the binary for each OS type
# base it on matlab's "mexext" command

ifeq ($(Arch), mi64)
	suff = maci
endif

ifeq ($(Arch), x86_64)
	suff = a64
endif

ifeq ($(Arch), i686)
	suff = glx
endif

#ifeq ($(suff), unknown)
#	@echo "unknown extension $(ext)"
#endif

goal = blochmulti_spsp-$(suff)

all:	$(goal)
	@echo "done suff=$(suff)"

$(goal):	blochmulti_spsp_2008_12_03.c
		gcc -o $@ $? -lm
