
# sh upload_pypi in terminal
# step 1 run this script
# step 2 commit to github
rm dist/*

python3 setup.py sdist bdist_wheel
python3 -m twine upload --repository pypi dist/*
