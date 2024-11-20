
# sh upload_pypi in terminal

# before run this script, please install twine, wheel
# before run this script, check __init__.py
# before run this script, check setup.py

# step 1 run this script
# step 2 commit to github

# after run this script, check commit information


rm dist/*
rm build/*

python3 setup.py sdist bdist_wheel
python3 -m twine upload --repository pypi dist/*
