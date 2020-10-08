. env/bin/activate

export PYTHONPATH="$PYTHONPATH:~/torch_async/models"
export PYTHONPATH="$PYTHONPATH:~/torch_async/preprocess"

GPU_IDX=0
EPOCHS=100

for seed in $(seq 0 4)
do
  python3 procedure/learn.py --seed "$seed" --gpu "$GPU_IDX" 2>&1 --epochs "$EPOCHS" | tee res"${seed}".txt
done
