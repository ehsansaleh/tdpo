#!/bin/bash
HOSTNAME=$(hostname)
NODENAME="${HOSTNAME%%.*}"

if [[ "$JOBMODE" == "true" ]]; then
  SCRIPTDIR=${SCRIPTDIR}
  NODESIZE=${NODESIZE}
  NODERANK=${SLURM_ARRAY_TASK_ID}
  CFGPREFIXARRSTR=${CFGPREFIXARRSTR}
  TIMEOUTVAL=${TIMEOUTVAL}
  MPIWORKERS=${MPIWORKERS}
  IFS=':' read -r -a CFGPREFIXARR <<< "$CFGPREFIXARRSTR"
  OPENJUP="false"
  ONLYONERUN="false"
else
  SCRIPTDIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
  NODERANK=0
  NODESIZE=1
  CFGPREFIXARR=("0_scratch/td3_lfeasync")
  TIMEOUTVAL="120m"
  MPIWORKERS=4
  ONLYONERUN="true"
  OPENJUP="false"
fi

echo $NODENAME
cd $SCRIPTDIR
source .env.sh

if [[ $(type -t launchjupyter) != function ]]; then
  function launchjupyter {
    jupyter notebook --no-browser &
  }
fi

if [[ "$OPENJUP" == "true" ]]; then
  launchjupyter
fi

for CFGPREFIX in ${CFGPREFIXARR[@]}; do
  if [[ $CFGPREFIX == *ppo2* ]]; then
    PYTHONCMD="timeout $TIMEOUTVAL python"
  else
    PYTHONCMD="timeout $TIMEOUTVAL mpirun -n $MPIWORKERS python -m mpi4py"
  fi

  EXECID=$(date +"%F-%T.%N")
  EXECID="${EXECID}-${NODENAME}"

  export RUNCNTFILE=$(mktemp)
  trap 'rm -rf -- "$RUNCNTFILE"' EXIT
  echo "1" > $RUNCNTFILE

  OUTERRDIR="./joblogs/${CFGPREFIX}"
  mkdir -p $OUTERRDIR
  OUTERRFILE="${OUTERRDIR}/rank${NODERANK}_${NODENAME}.txt"

  RUNCNT=0
  while [[ $RUNCNT -lt $(< ${RUNCNTFILE}) ]]; do
    echo "----------" >> $OUTERRFILE
    echo $(date;hostname;pwd) >> $OUTERRFILE

    source .env.sh
    $PYTHONCMD trainer.py -c "./configs/${CFGPREFIX}.json" -s $NODESIZE -r $NODERANK --exec_id $EXECID >> $OUTERRFILE 2>&1
    source .deenv.sh
    RUNCNT=$((RUNCNT+1))

    [[ "$ONLYONERUN" == "true" ]] && RUNCNT=$(< ${RUNCNTFILE})

    a=$(($(< ${RUNCNTFILE})-${RUNCNT}))
    [[ $a -lt 0 ]] && a=0
    echo "$a calls to go!" >> $OUTERRFILE
    echo "--------------------" >> $OUTERRFILE
  done
  rm -rf -- "$RUNCNTFILE"
done

wait
