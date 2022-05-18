#!/bin/bash

# slurm-related variables
nodes=30                                  # Run all processes on a single node
ntaskspernode=4                           # Number of available threads in a job
mpiworkers=4                              # Number of MPI ranks (<= ntaskspernode)
time=12:00:00                             # Time limit hrs:min:sec
partition="west"                          # The submission queue
slurmemail="ehsans2@illinois.edu"
TIMEOUTVAL="12h"
# experiment specification
CFGPREFIXARR=("3_lfhpo_west/0_ppo1_ovat")

set -e
SCRIPTDIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
source $SCRIPTDIR/../.env.sh
cd $SCRIPTDIR

join_by () { local IFS="$1"; shift; echo "$*"; }
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

sbatch --job-name="abc" --mail-type="BEGIN,END,FAIL" \
       --mail-user=${slurmemail} \
       --nodes=1 --cpus-per-task=1 --exclude=ccc0156 \
       --ntasks-per-node=$ntaskspernode --time=$time \
       --partition=$partition --output=$output --open-mode=append \
       --array=0-$(($nodes-1)) --export=$EXPORTVARS \
       $(realpath ../trainer.sh)
