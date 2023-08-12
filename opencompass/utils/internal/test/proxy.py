import os

backup = {}

def proxy_off():
    global backup
    if 'http_proxy' in os.environ:
        backup['http_proxy'] = os.environ['http_proxy']
        del os.environ['http_proxy']
    if 'https_proxy' in os.environ:
        backup['https_proxy'] = os.environ['https_proxy']
        del os.environ['https_proxy']
    if 'HTTP_PROXY' in os.environ:
        backup['HTTP_PROXY'] = os.environ['HTTP_PROXY']
        del os.environ['HTTP_PROXY']
    if 'HTTPS_PROXY' in os.environ:
        backup['HTTPS_PROXY'] = os.environ['HTTPS_PROXY']
        del os.environ['HTTPS_PROXY']

def proxy_on():
    global backup
    if 'http_proxy' in backup:
        os.environ['http_proxy'] = backup['http_proxy']
    if 'https_proxy' in backup:
        os.environ['https_proxy'] = backup['https_proxy']
    if 'HTTP_PROXY' in backup:
        os.environ['HTTP_PROXY'] = backup['HTTP_PROXY']
    if 'HTTPS_PROXY' in backup:
        os.environ['HTTPS_PROXY'] = backup['HTTPS_PROXY']