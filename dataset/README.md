```bash

python -m venv .env

source .env/bin/activate

docker run -ti --rm \
-v ~/work/code/py_code/deep-learning/02-deep-learning-advanced:/02-deep-learning-advanced \
-w /02-deep-learning-advanced \
docker-mirrors.alauda.cn/library/python:3.10.12-bullseye \
bash

pip install -i https://pypi.tuna.tsinghua.edu.cn/simple yapf

find . -name "*.py" -exec yapf -i {} \;


```
