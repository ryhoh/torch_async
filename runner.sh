. env/bin/activate

export PYTHONPATH="$PYTHONPATH:~/torch_async/models"
export PYTHONPATH="$PYTHONPATH:~/torch_async/preprocess"

GPU_IDX=0
EPOCHS=50

for seed in $(seq 0 3)
do
  python3 procedure/learn.py --seed "$seed" --gpu "$GPU_IDX" --epochs "$EPOCHS" 2>&1 | tee res"${seed}".txt
done
