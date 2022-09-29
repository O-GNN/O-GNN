#!/bin/bash
set -x
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

cd $SCRIPT_DIR/
if [ -z "$(pip list | grep ogb)" ]; then
    pip install -e . --user
fi

cuda=0
POSITIONAL=()
prefix=pcqm4m0
dist=false
nproc=none
reload=false
while [[ $# -gt 0 ]]; do
    key=$1
    case $key in
    -c | --cuda)
        cuda=$2
        shift 2
        ;;
    --dist)
        dist=true
        shift
        ;;
    --prefix)
        prefix=$2
        shift 2
        ;;
    --nproc)
        nproc=$2
        shift 2
        ;;
    --reload)
        reload=true
        shift
        ;;
    *)
        POSITIONAL+=("$1")
        shift
        ;;
    esac
done
SUFFIX=$(echo ${POSITIONAL[*]} | sed -r 's/-//g' | sed -r 's/\s+/-/g')
SAVEDIR=/tmp/model/ringnew/checkpoints/${prefix}
if [ -n "$SUFFIX" ]; then
    SAVEDIR=${SAVEDIR}-${SUFFIX}
fi

mkdir -p $SAVEDIR
export KMP_WARNINGS=FALSE
if [ "$dist" == true ]; then
    if [ "$proc" == "none" ]; then
        cudaa=$(echo $cuda | sed -r 's/,//g')
        nproc=${#cudaa}
        CUDA_VISIBLE_DEVICES=$cuda python -m torch.distributed.launch --nproc_per_node=$nproc examples/lsc/pcqm4m/main_dg.py \
            --checkpoint-dir $SAVEDIR ${POSITIONAL[@]} | tee $SAVEDIR/training.log
    else
        python -m torch.distributed.launch --nproc_per_node=$nproc examples/lsc/pcqm4m/main_dg.py \
            --checkpoint-dir $SAVEDIR ${POSITIONAL[@]} | tee $SAVEDIR/training.log
    fi
else
    if [ "$reload" == "true" ]; then
        CUDA_VISIBLE_DEVICES=$cuda python examples/lsc/pcqm4m/main_dg.py \
            --checkpoint-dir $SAVEDIR ${POSITIONAL[@]} --reload | tee -a $SAVEDIR/training.log
    else
        CUDA_VISIBLE_DEVICES=$cuda python examples/lsc/pcqm4m/main_dg.py \
            --checkpoint-dir $SAVEDIR ${POSITIONAL[@]} | tee $SAVEDIR/training.log
    fi
fi
