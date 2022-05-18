#!/bin/bash
set -e

# slurm-related variables
nodes=15                                  # Run all processes on a single node
ntaskspernode=8                           # Number of available threads in a job
mpiworkers=4                              # Number of MPI ranks (<= ntaskspernode)
time=72:00:00                             # Time limit hrs:min:sec
partition="west"                          # The submission queue
TIMEOUTVAL="2.5h"
slurmemail="ehsans2@illinois.edu"

# HPO specification
HPOITERSARR=($(seq 24 24))
CFGTREE="3_lfhpo_west/1_td3_optuna"

SCRIPTDIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
source $SCRIPTDIR/../.env.sh
cd $SCRIPTDIR

join_by () {
  local IFS="$1"; shift; echo "$*";
}
pushd () {
    command pushd "$@" > /dev/null
}
popd () {
    command popd "$@" > /dev/null
}

for HPOITER in "${HPOITERSARR[@]}"; do
  HPOCFGTREEID="${CFGTREE}/hpo"
  MYCFGPREFIX="${CFGTREE}/round_${HPOITER}"
  [[ $HPOCFGTREEID == *skopt* ]] && MPICMD="" || MPICMD="mpirun -n 20"
  echo "Bash: HPO iteration $HPOITER"
  pushd ../usr
  PYTHONHASHSEED=0 $MPICMD python hpo.py -c $HPOCFGTREEID
  popd

  if [[ ! -f "../configs/${MYCFGPREFIX}.json" ]]; then
    echo "cannot find ../configs/${MYCFGPREFIX}.json"
    break
  fi
  if [[ -d "../storage/${MYCFGPREFIX}" ]]; then
    echo "the storage dir ../storage/${MYCFGPREFIX} is not empty."
    break
  fi
  if [[ -d "../results/${MYCFGPREFIX}" ]]; then
    echo "the results dir ../results/${MYCFGPREFIX} is not empty."
    break
  fi

  CFGPREFIXARR=($MYCFGPREFIX)
  CFGPREFIXARRSTR=$(join_by ":" "${CFGPREFIXARR[@]}")
  EXPORTVARS="ALL,JOBMODE=true"
  EXPORTVARS="${EXPORTVARS},SCRIPTDIR=$(realpath ../)"
  EXPORTVARS="${EXPORTVARS},NODESIZE=${nodes}"
  EXPORTVARS="${EXPORTVARS},CFGPREFIXARRSTR=${CFGPREFIXARRSTR}"
  EXPORTVARS="${EXPORTVARS},TIMEOUTVAL=${TIMEOUTVAL}"
  EXPORTVARS="${EXPORTVARS},MPIWORKERS=${mpiworkers}"
  mkdir -p ../joblogs/slurm
  output="../joblogs/slurm/log_%j.txt"   # Standard output and error log
  output=$(realpath $output)
  for CFGPREFIX in ${CFGPREFIXARR[@]}; do
    python redivide_runbooks.py -r ../storage/${CFGPREFIX}/runbooks -s ${nodes}
  done

  sbatch --wait --job-name="abc" --mail-type="BEGIN,END,FAIL" \
     --mail-user=${slurmemail} \
     --nodes=1 --cpus-per-task=1 --exclude=ccc0156 \
     --ntasks-per-node=$ntaskspernode --time=$time \
     --partition=$partition --output=$output --open-mode=append \
     --array=0-$(($nodes-1)) --export=$EXPORTVARS \
     $(realpath ../trainer.sh)

  jobstatus=$?
  if [ $jobstatus -eq 0 ] ; then
    echo "slurm job successful"
    echo "-----------------------------"
  else
    echo "slurm job unsuccessful"
    break
  fi

  [[ $HPOCFGTREEID == *skopt* ]] && (cd ../ && make summary && cd -)
done
