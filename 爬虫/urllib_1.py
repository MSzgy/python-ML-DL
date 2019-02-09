import urllib.request as ur
import http.cookiejar

# response = ur.urlopen("http://www.baidu.com")
# print(response.read().decode('utf-8'))

cookie = http.cookiejar.CookieJar()
handler = ur.HTTPCookieProcessor(cookie)
opener = ur.build_opener(handler)
response = opener.open("http://www.baidu.com")
for item in cookie:
    print(item.name + "=" + item.value)
