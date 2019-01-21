import urllib.request as ur

response = ur.urlopen("http://www.baidu.com")
print(response.read().decode('utf-8'))
