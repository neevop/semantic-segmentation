#!/bin/bash

# wget --keep-session-cookies --save-cookies=cookies.txt --post-data 'username={username}&password={passwd}&submit=Login' https://www.cityscapes-dataset.com/login/
wget -c -t 0 --load-cookies cookies.txt  --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=4
unzip index.html?packageID=4 && rm README license.txt index.html?packageID=4
wget -c -t 0 --load-cookies cookies.txt  --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=3
unzip index.html?packageID=3 && rm README license.txt index.html?packageID=3
wget -c -t 0 --load-cookies cookies.txt  --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=2
unzip index.html?packageID=2 && rm README license.txt index.html?packageID=2
wget -c -t 0 --load-cookies cookies.txt  --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=1
unzip index.html?packageID=1 && rm README license.txt index.html?packageID=1
wget -c -t 0 --load-cookies cookies.txt  --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=5
unzip index.html?packageID=5 && rm README license.txt index.html?packageID=5
