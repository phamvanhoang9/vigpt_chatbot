## Create virtual environment

*Note*: This step is very important!

```bash
python3.10 -m venv venv
source venv/bin/activate 
```

## Upgrage pip

```bash
python -m pip install --upgrade pip
```

## Install the requirements

```bash
python -m pip install -r requirements.txt
```

## Known Issues

### Issue 1: RuntimeError: Your system has an unsupported version of sqlite3. Chroma requires sqlite3 >= 3.35.0.

If you encounter this error, please following these command lines. I hope it will work :)), because I don't remember exactly how I resolved this issue. Take you time:3 :

*Solution 1*:

1. Upgrade the `sqlite3` library on your system:
```bash 
sudo apt-get install sqlite3 libsqlite3-dev
```
2. Upgrade the `sqlite3` module using `pip`:
```bash
python3.10 -m pip install --upgrade sqlite3
```
3. If you are still encountering the issue, you might need to upgrade the `chromadb` library as well:
```bash
python3.10 -m pip install --upgrade chromadb
```
*Solution 2*:

1. You can try reinstall `pip`:
```bash
python3.10 -m pip uninstall pip
sudo apt-get install python3.10-distutils
sudo apt-get install python3.10-venv 
sudo apt-get install python3-pip
```
2. If `pip` is still not available after activating the virtual environment, you can install it manually by downloading `get-pip.py`:
```bash
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python3.10 get-pip.py
```
3. After reinstalling `pip`, you can attempt to upgrade `sqlite3` again:
```bash
python3.10 -m pip install --upgrade sqlite3
```
4. If you encounter issues during the installation or if the problem persists, you might consider installing `html5lib` separately before upgrading `sqlite3`:
```bash
sudo python3.10 -m pip install --upgrade html5lib
sudo python3.10 -m pip install --upgrade sqlite3
```
5. Finally, you can install or upgrade `chromadb`:
```bash
python -m pip install --upgrade chromadb==0.3.29
```

### Issue 2: AttributeError: module 'openai' has no attribute 'error'.

***Solution***:
```bash
python -m pip install langchain==0.0.316
python -m pip install openai==0.28.1
```