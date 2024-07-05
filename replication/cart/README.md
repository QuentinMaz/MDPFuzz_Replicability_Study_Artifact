# Cart Pole

## Installation (if not using the Docker image)

Setup the environment:
```bash
conda create -n cart python=3.10.12
conda activate cart
pip install -r requirements.txt
pip install git+https://github.com/DennisGross/gimitest.git
```

If you are using the Docker image, simply activate the latter with `conda activate cart`.

## Experiments

### RQ2: Fault Discovery Evaluation

To replicate one method, use the script `test_cart.py PATH METHOD`, whose arguments are the folder where to save the results and the method's name (`fuzzer`, `mdpfuzz` or `rt`).
The script sequentially repeats the executions 5 times (with the random seeds used in the paper).
Besides, we provide the script `launch_rq2.sh` to conveniently launch the three methods in seperate threads.
Even if you don't use the aforementioned script, we strongly recommend using the default path for logging the results ``../data_rq2/cart/``, i.e.:
```python
python test_cart.py ../data_rq2/cart/ mdpfuzz
python test_cart.py ../data_rq2/cart/ fuzzer
python test_cart.py ../data_rq2/cart/ rt
```
This path is expected by ``main.py`` for retrieving the data.

### RQ3: Parameter Analysis

MDPFuzz can be executed for a particular configuration and seed with the script `test_mdpfuzz_cart.py K TAU GAMMA SEED PATH` (where `PATH` defines where to record the results).
As before, please use the one assumed by the artifact: `../data_rq3/cart`.
The script can be also provided with a positive integer which indicates the line in the file `../parameters.txt` to read the previous arguments (i.e., `test_mdpfuzz_cart.py I PATH`).
That way you can execute at your own pace the 110 executions.
This alternative input is used by the script `launch_rq3.sh`, which starts all the configurations studied for this use case.