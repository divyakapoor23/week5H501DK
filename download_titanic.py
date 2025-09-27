# Download Titanic dataset locally
import urllib.request

url = 'https://raw.githubusercontent.com/leontoddjohnson/datasets/main/data/titanic.csv'
local_path = 'titanic.csv'

urllib.request.urlretrieve(url, local_path)
print(f"Downloaded Titanic dataset to {local_path}")
