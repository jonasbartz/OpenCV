#from StringIO import StringIO

import io

import requests
from PIL import Image
from bs4 import BeautifulSoup

DCS_IP = "https://10.91.60.60/home.htm"
userauth = ('admin', 'admin')

snapurl = "http://" + DCS_IP + "/top.htm"

r = requests.get(snapurl, auth=userauth)
soup = BeautifulSoup(r.content)

print(soup)
# There are several <img> tags in page, so use border=0 attribute of
# objective <img> to distinguish it
imgtag = soup.find_all("img", attrs={'border':0})

print(imgtag)

imgsrc = BeautifulSoup(str(imgtag[0])).img['src']
imgurl = "http://" + DCS_IP + "/" + imgsrc

img = requests.get(imgurl, auth=userauth)
i = Image.open(io.StringIO(img.content))
i.save("snapshot.png")