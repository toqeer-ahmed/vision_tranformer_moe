import urllib.request
import re
import json

print('Checking Zenodo BUS-BRA...')
try:
    req = urllib.request.Request('https://zenodo.org/records/10695024', headers={'User-Agent': 'Mozilla/5.0'})
    html = urllib.request.urlopen(req).read().decode('utf-8')
    links = re.findall(r'href=[\'\"](.*?\.zip.*?)[\'\"]', html)
    print('Found zip links:', links)
except Exception as e:
    print('Zenodo error:', e)

print('\nChecking Mendeley BUSI...')
try:
    req = urllib.request.Request('https://data.mendeley.com/public-api/datasets/wmy84lzdic/versions/1', headers={'User-Agent': 'Mozilla/5.0'})
    data = urllib.request.urlopen(req).read().decode('utf-8')
    print('Mendeley response:', data[:200])
except Exception as e:
    print('Mendeley error:', e)
