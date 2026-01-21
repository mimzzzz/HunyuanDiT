model='DiT-g/2'
params=" \
            --qk-norm \
            --model ${model} \
            --rope-img base1024 \
            --rope-real \
            "
deepspeed hydit/train_deepspeed.py ${params}  "$@"