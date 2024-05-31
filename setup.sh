# python=3.10.12
# pip=24.0

pip install virtualenv

cd LRA

virtualenv -p python3.10 PFC3env

source PFC3env/bin/activate

pip install --upgrade pip==24.0

cd src

pip install -r requirements.txt

exec "$SHELL"
