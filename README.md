# Probabilistic Web Crawler

### Installation:
```
virtualenv -p python3 env
source env/bin/activate
pip install lxml
```

### Usage:
##### Find element in all variants:
```
python3 crawl.py train/sample-0-origin.html test/sample-1-evil-gemini.html test/sample-2-container-and-clone.html test/sample-3-the-escape.html  test/sample-4-the-mash.html "make-everything-ok-button"
```
Expected output:
```
'body/div/div/div[3]/div[1]/div/div[2]/a[2]' (score=0.9921029164925812)
'body/div/div/div[3]/div[1]/div/div[2]/div/a' (score=0.9952613484261197)
'body/div/div/div[3]/div[1]/div/div[3]/a' (score=0.989535451948429)
'body/div/div/div[3]/div[1]/div/div[3]/a' (score=0.9940386760307208)
```

##### Find element in one variant:
```
python3 crawl.py train/sample-0-origin.html test/sample-1-evil-gemini.html make-everything-ok-button
```
Expected output:
```
'body/div/div/div[3]/div[1]/div/div[2]/a[2]' (score=0.9921029164925812)
```
