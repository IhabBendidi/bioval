poetry build
pip3 install dist/bioval-0.1.1.5a0.tar.gz
python3 bioval/metrics/conditional_evaluation.py
python3 bioval/tests/test_conditional_evaluation.py