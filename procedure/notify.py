import requests


def send(msg):
    """ LINEに通知を送る"""
    url = "https://notify-api.line.me/api/notify"
    access_token = 'Access Token'
    headers = {'Authorization': 'Bearer ' + access_token}

    payload = {'message': msg}

    # 送信
    requests.post(url, headers=headers, params=payload)
