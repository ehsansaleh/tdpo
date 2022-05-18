SHELL := /bin/bash
PROJBASE := $(shell dirname $(abspath $(lastword $(MAKEFILE_LIST))))
MPIPRECMD := mpirun -n 2 --oversubscribe

.PHONY: summary

summary:
	source .env.sh && cd usr && ${MPIPRECMD} python csv2summ.py

hpo_trpo: summary
	source .env.sh && cd usr && PYTHONHASHSEED=0 ${MPIPRECMD} python hpo.py -c 4_hpo5b_eng/1_trpo_optuna/hpo

fix_crlf:
	find ${PROJBASE} -maxdepth 3 -type f -name "*.md5" \
	  -o -name "*.py" -o -name "*.sh" -o -name "*.json" | xargs dos2unix

TODAYDATE := $(shell date +%m_%d_%Y)
clean:
	@if [[ "aa"$(CFGTREE) == "aa" ]]; then \
	  echo 'usage: make clean CFGTREE="0_scratch/ppo1"'; \
		exit 1; \
	fi;
	@for dirkind in "results" "storage" "joblogs"; do \
		if [[ ! -d "$${dirkind}/$(CFGTREE)" ]]; then \
		  echo "The directory $${dirkind}/$(CFGTREE) does not exist"; \
		else \
		  echo mkdir -p trash/${TODAYDATE}/$${dirkind}; \
		  mkdir -p trash/${TODAYDATE}/$${dirkind}; \
			echo mv $${dirkind}/$(CFGTREE) trash/${TODAYDATE}/$${dirkind}; \
			mv results/$(CFGTREE) trash/${TODAYDATE}/$${dirkind}; \
		fi; \
	done
